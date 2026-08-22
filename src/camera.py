import cv2
import threading
import time
from typing import Callable, Optional


class VideoStream:
    def __init__(self, src=0):
        self.src = src
        self.cap = cv2.VideoCapture(src)
        self._stopped = False
        self.frame = None
        self.lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None

    def start(self):
        if self.thread is None:
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
        return self

    def _run(self):
        while not self._stopped:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self.lock:
                self.frame = frame

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self._stopped = True
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        try:
            self.cap.release()
        except Exception:
            pass


class ConveyorSimulator:
    """Simple conveyor simulator that yields frames at intervals and marks when a part is 'in position'."""

    def __init__(self, frame_provider: Callable[[], any], part_interval: float = 2.0):
        self.frame_provider = frame_provider
        self.part_interval = part_interval
        self._last = time.time()

    def next(self):
        # returns (frame, part_present:boolean)
        now = time.time()
        frame = self.frame_provider()
        if now - self._last > self.part_interval:
            self._last = now
            return frame, True
        return frame, False
