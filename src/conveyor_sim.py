import threading
import time
import cv2
import numpy as np
from typing import Callable, Optional

from .camera import VideoStream
from src.logging_config import get_logger
from src.metrics import inspections_total, inspections_failed

log = get_logger('conveyor')


class ConveyorLine:
    """Simulate a conveyor line with a photo-eye trigger.

    - Uses a running background model and an ROI (photo-eye) to detect parts.
    - On trigger it captures a frame, runs the provided inspection callback,
      and emits a simulated reject signal via the `reject_callback`.
    """

    def __init__(
        self,
        inspector_callable: Callable[[np.ndarray], tuple],
        src: int = 0,
        photo_eye_roi: Optional[tuple] = None,
        trigger_thresh: float = 100000.0,
        debounce_s: float = 0.5,
        reject_callback: Optional[Callable[[dict], None]] = None,
        net_protocol: Optional[str] = None,
        net_endpoint: Optional[str] = None,
    ):
        self.inspector = inspector_callable
        self.src = src
        self.photo_eye_roi = photo_eye_roi or (0.8, 0.4, 0.95, 0.6)  # normalized x0,y0,x1,y1
        self.trigger_thresh = trigger_thresh
        self.debounce_s = debounce_s
        self.reject_callback = reject_callback
        self.net_protocol = net_protocol
        self.net_endpoint = net_endpoint
        self.net_client = None
        if self.net_protocol is not None:
            try:
                from .industrial_net import create_client
                self.net_client = create_client(self.net_protocol, self.net_endpoint)
            except Exception:
                self.net_client = None

        self.vs = VideoStream(self.src)
        self._stop = False
        self._thread = None

    def start(self):
        self.vs.start()
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop = True
        if self._thread:
            self._thread.join(timeout=1.0)
        try:
            self.vs.stop()
        except Exception:
            pass

    def _roi_rect(self, frame):
        h, w = frame.shape[:2]
        x0 = int(self.photo_eye_roi[0] * w)
        y0 = int(self.photo_eye_roi[1] * h)
        x1 = int(self.photo_eye_roi[2] * w)
        y1 = int(self.photo_eye_roi[3] * h)
        return x0, y0, x1, y1

    def _run(self):
        # running background
        bg = None
        last_trigger = 0.0
        while not self._stop:
            frame = self.vs.read()
            if frame is None:
                time.sleep(0.01)
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if bg is None:
                bg = gray.astype(np.float32)
                time.sleep(0.01)
                continue

            # update running background slowly
            cv2.accumulateWeighted(gray.astype(np.float32), bg, 0.05)
            bg_img = cv2.convertScaleAbs(bg)

            x0, y0, x1, y1 = self._roi_rect(frame)
            roi_cur = gray[y0:y1, x0:x1]
            roi_bg = bg_img[y0:y1, x0:x1]

            diff = cv2.absdiff(roi_cur, roi_bg)
            motion_score = float(np.sum(diff))

            # trigger if motion_score exceeds threshold and debounce elapsed
            if motion_score > self.trigger_thresh and (time.time() - last_trigger) > self.debounce_s:
                last_trigger = time.time()
                # capture full-resolution frame for inspection
                captured = frame.copy()
                try:
                    metrics, ann = self.inspector(captured)
                    try:
                        inspections_total.inc()
                        if metrics.get('status') == 'FAIL':
                            inspections_failed.inc()
                    except Exception:
                        pass
                except Exception as e:
                    metrics = {"status": "ERROR", "reason": str(e)}
                    ann = captured

                # emit simulated signal
                signal = {"timestamp": time.time(), "metrics": metrics}
                if metrics.get("status") == "FAIL":
                    # simulated Modbus/OPC-UA write
                    print(f"[CONVEYOR] REJECT signal emitted for part: {metrics.get('part_id')}")
                    if self.net_client is not None:
                        try:
                            self.net_client.send_reject(signal)
                        except Exception:
                            pass
                    if self.reject_callback:
                        try:
                            self.reject_callback(signal)
                        except Exception:
                            pass
                else:
                    print(f"[CONVEYOR] ACCEPT part: {metrics.get('part_id')} status={metrics.get('status')}")

                # pause briefly to simulate gate action
                time.sleep(0.25)

            time.sleep(0.01)
