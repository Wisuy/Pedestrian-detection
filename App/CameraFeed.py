import cv2
import threading
import time


class CameraFeed:
    def __init__(self, source):
        self.source = source
        self.cap = None
        self.frame = None
        self.ret = False
        self.running = False
        self.read_lock = threading.Lock()
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._update, args=())
        self.thread.daemon = True
        self.thread.start()

    def _update(self):
        # Determine if it's an IP camera or local camera
        is_ip_camera = isinstance(self.source, str) and (self.source.startswith("http"))
        if is_ip_camera:
            self.cap = cv2.VideoCapture(self.source)
            if not self.cap.isOpened():
                print(f"Attempting to open IP Camera with default backend failed. Trying with cv2.CAP_FFMPEG...")
                self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Ensure minimal buffer
        else:
            self.cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW) # Use CAP_DSHOW for local cameras

        if not self.cap.isOpened():
            print(f"Failed to open camera from source: {self.source}. Exiting thread.")
            self.running = False
            return
        
        while self.running:
            ret, frame = self.cap.read()
            with self.read_lock:
                self.ret = ret
                self.frame = frame
            
            # Small delay to prevent burning CPU if frame rate is very high
            time.sleep(0.001)

        if self.cap:
            self.cap.release()

    def read_latest_raw_frame(self):
        with self.read_lock:
            # Return a copy to prevent external modification issues
            return self.ret, self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False