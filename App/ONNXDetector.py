import cv2
import numpy as np
import onnxruntime as ort
import time
import supervision as sv


class ONNXDetector:
    def __init__(self, model_path, input_size=(640, 640), conf_threshold=0.4,
                 focal_length=605, real_height=1.7):
        providers = [
            'CUDAExecutionProvider', # Use CUDA if available using ONNX Runtime GPU library
            'DmlExecutionProvider',
            'CPUExecutionProvider'
        ]
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        _, _, self.input_height, self.input_width = self.input_shape
        self.input_size = input_size
        self.conf_threshold = conf_threshold

        self.focal_length = focal_length
        self.real_height = real_height

        self.tracker = sv.ByteTrack()
        self.previous_distances = {}
        self.last_frame_time = None
        self.previous_speeds = {}

    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114)):
        shape = img.shape[:2]
        ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = (int(shape[1] * ratio), int(shape[0] * ratio))
        dw = new_shape[1] - new_unpad[0]
        dh = new_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2

        resized_img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

        padded_img = cv2.copyMakeBorder(resized_img, top, bottom, left, right,
                                        cv2.BORDER_CONSTANT, value=color)
        return padded_img, ratio, dw, dh

    def preprocess(self, frame):
        t0 = time.time()
        img, scale, dw, dh = self.letterbox(frame, new_shape=(self.input_height, self.input_width))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC to CHW
        img = np.expand_dims(img, axis=0)
        t1 = time.time()
        return img, scale, dw, dh, (t1 - t0) * 1000

    def postprocess(self, outputs, scale, dw, dh):
        t0 = time.time()
        pred = outputs[0]
        pred = np.squeeze(pred, axis=0)
        pred = pred.T
        boxes = []
        confidences = []
        class_ids = []
        for row in pred:
            cx, cy, w, h = row[0], row[1], row[2], row[3]
            conf = row[4] # For a single class, this is often the objectness score * class_score

            if conf < self.conf_threshold:
                continue
            
            cx = (cx * self.input_width - dw) / scale
            cy = (cy * self.input_height - dh) / scale
            w = w * self.input_width / scale
            h = h * self.input_height / scale
            x1 = int(cx - w / 2)
            y1 = int(cy - h / 2)
            x2 = int(cx + w / 2)
            y2 = int(cy + h / 2)
            boxes.append([x1, y1, x2, y2])
            confidences.append(float(conf))
            class_ids.append(0)

        t1 = time.time()
        return boxes, confidences, class_ids, (t1 - t0) * 1000

    def __call__(self, frame):
        current_frame_time = time.time()
        time_diff = 0.0
        if self.last_frame_time is not None:
            time_diff = current_frame_time - self.last_frame_time

        img, scale, dw, dh, pre_time = self.preprocess(frame)
        t0 = time.time()
        outputs = self.session.run(None, {self.input_name: img})
        t1 = time.time()
        inf_time = (t1 - t0) * 1000

        # Pass class_ids from postprocess
        boxes, confidences, class_ids, post_time = self.postprocess(outputs, scale, dw, dh)

        # Create detections object BEFORE NMS
        if not boxes:
            detections = sv.Detections(
                xyxy=np.empty((0, 4)),
                confidence=np.empty((0,)),
                class_id=np.empty((0,)).astype(int)
            )
        else:
            xyxy_boxes = np.array(boxes)
            detections = sv.Detections(
                xyxy=xyxy_boxes,
                confidence=np.array(confidences),
                class_id=np.array(class_ids).astype(int)
            )

        # NMS using Supervision
        detections = detections.with_nms(threshold=0.6) # IoU threshold for NMS

        detections = detections[detections.confidence > self.conf_threshold]

        detections = self.tracker.update_with_detections(detections)

        distances = []
        movement_vectors = {}
        current_distances = {}

        for i, box in enumerate(detections.xyxy):
            x1, y1, x2, y2 = box
            pixel_height = y2 - y1
            distance = None
            if pixel_height > 0:
                distance = (self.real_height * self.focal_length) / pixel_height
            distances.append(distance)

            tracker_id = detections.tracker_id[i]
            current_distances[tracker_id] = distance

            approaching_speed_mps = None

            if tracker_id in self.previous_distances and self.previous_distances[tracker_id] is not None and distance is not None:
                raw_distance_change = self.previous_distances[tracker_id] - distance

                if time_diff > 0.05 and abs(raw_distance_change) < 5:  # avoid division spikes
                    raw_speed = raw_distance_change / time_diff
                    previous_speed = self.previous_speeds.get(tracker_id, raw_speed)
                    smoothed_speed = 0.3 * raw_speed + 0.7 * previous_speed
                    self.previous_speeds[tracker_id] = smoothed_speed

                    if smoothed_speed > 0 and smoothed_speed < 20:
                        approaching_speed_mps = smoothed_speed
                    else:
                        approaching_speed_mps = 0.0
                else:
                    approaching_speed_mps = 0.0
            
            movement_vectors[tracker_id] = {
                "distance_change": raw_distance_change if 'raw_distance_change' in locals() else None,
                "time_diff_seconds": time_diff,
                "approaching_speed_mps": approaching_speed_mps
            }

        self.previous_distances = current_distances
        self.last_frame_time = current_frame_time

        times = {
            "preprocess": pre_time,
            "inference": inf_time,
            "postprocess": post_time
        }

        return (
            detections.xyxy.tolist() if detections.xyxy.size > 0 else [],
            detections.confidence.tolist() if detections.confidence.size > 0 else [],
            detections.tracker_id.tolist() if detections.tracker_id.size > 0 else [],
            distances,
            movement_vectors,
            times
        )