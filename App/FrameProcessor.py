import threading
import time
import cv2

class FrameProcessor:
    def __init__(self, camera_feed_obj, onnx_model_obj, show_labels_var, car_deceleration_mps2):
        self.camera_feed = camera_feed_obj
        self.onnx_model = onnx_model_obj
        self.show_labels_var = show_labels_var
        self.car_deceleration_mps2 = car_deceleration_mps2

        self.processed_frame = None
        self.processing_info = {}
        self.running = False
        self.process_lock = threading.Lock()
        self.thread = None

        self.lowest_speeds_history = []
        self.FRAME_AVERAGE_WINDOW = 5
        self.MPS_TO_KPH = 3.6

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._process_frames, args=())
        self.thread.daemon = True
        self.thread.start()

    def _process_frames(self):
        while self.running:
            ret, frame = self.camera_feed.read_latest_raw_frame()
            if not ret or frame is None:
                # If no frame from camera, wait a bit and try again
                time.sleep(0.01)
                continue

            try:
                h, w = frame.shape[:2]

                # Calculate required_braking_distance based on current avg_lowest_speed
                avg_lowest_speed_mps = 0.0
                if self.lowest_speeds_history:
                    effective_speeds_for_avg = [s for s in self.lowest_speeds_history if s > 0]
                    if not effective_speeds_for_avg and self.lowest_speeds_history:
                        effective_speeds_for_avg = self.lowest_speeds_history

                    if effective_speeds_for_avg:
                        avg_lowest_speed_mps = sum(effective_speeds_for_avg) / len(effective_speeds_for_avg)
                    
                required_braking_distance = 0.0
                if avg_lowest_speed_mps > 0 and self.car_deceleration_mps2[0] > 0:
                    required_braking_distance = (avg_lowest_speed_mps ** 2) / (2 * self.car_deceleration_mps2[0])

                boxes, confidences, tracker_ids, distances, movement_vectors, times = self.onnx_model(frame)

                annotated_frame = frame.copy()
                current_frame_speeds = []
                show_labels = self.show_labels_var.get()

                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box
                    x1, y1 = max(0, x1), max(y1, 0)
                    x2, y2 = min(frame.shape[1]-1, x2), min(frame.shape[0]-1, y2)

                    conf = confidences[i]
                    tracker_id = tracker_ids[i]
                    dist = distances[i]

                    color = (0, 255, 0) # Green
                    if dist is not None and required_braking_distance > 0 and dist < required_braking_distance:
                        color = (0, 0, 255) # Red

                    cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

                    movement_info = movement_vectors.get(tracker_id)
                    approaching_speed = None
                    if movement_info:
                        distance_change = movement_info.get("distance_change")
                        approaching_speed = movement_info.get("approaching_speed_mps")

                        if distance_change is not None and abs(distance_change) < 0.5:
                            distance_change = 0.0
                            approaching_speed = 0.0

                        if approaching_speed is not None:
                            speed_kph = approaching_speed * self.MPS_TO_KPH
                            if speed_kph > 100:
                                approaching_speed = 0.0
                    
                    if approaching_speed is not None and approaching_speed >= 0:
                        current_frame_speeds.append(approaching_speed)

                    if show_labels:
                        label_parts = [f"ID: {tracker_id}", f"Conf: {conf:.2f}"]
                        if dist is not None:
                            label_parts.append(f"Dist: {dist:.2f}m")

                        if movement_info:
                            distance_change = movement_info.get("distance_change")
                            if distance_change is not None:
                                label_parts.append(f"Moved: {distance_change:.2f}m")
                        
                        if approaching_speed is not None:
                             label_parts.append(f"Speed: {approaching_speed:.2f}m/s")
                    
                        label = " | ".join(label_parts)
                        
                        cv2.putText(annotated_frame, label, (int(x1), max(int(y1) - 10, 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                if current_frame_speeds:
                    lowest_speed_current = min(current_frame_speeds)
                    self.lowest_speeds_history.append(lowest_speed_current)
                else:
                    self.lowest_speeds_history.append(0.0) 

                if len(self.lowest_speeds_history) > self.FRAME_AVERAGE_WINDOW:
                    self.lowest_speeds_history.pop(0) 

                avg_lowest_speed_kph = avg_lowest_speed_mps * self.MPS_TO_KPH

                with self.process_lock:
                    self.processed_frame = annotated_frame.copy()
                    self.processing_info = {
                        "preprocess_time": times["preprocess"],
                        "inference_time": times["inference"],
                        "postprocess_time": times["postprocess"],
                        "avg_min_speed_kph": avg_lowest_speed_kph,
                        "resolution": f"{w}x{h}"
                    }
            except Exception as e:
                print(f"Error during frame processing: {e}")
                with self.process_lock:
                    self.processed_frame = None # Indicate no valid frame
                    self.processing_info = {}
                time.sleep(0.01) # Small sleep on error to prevent tight loop
            
            # Small delay to ensure the main thread can pick up the processed frame
            time.sleep(0.001)

    def get_latest_processed_frame(self):
        with self.process_lock:
            return self.processed_frame.copy() if self.processed_frame is not None else None, self.processing_info.copy()

    def stop(self):
        self.running = False