import os
import cv2
import time
from src.inspector import PartInspector
from src.conveyor_sim import ConveyorLine


def run_webcam_conveyor():
    if not os.path.exists('sample_data/reference.png'):
        print("Need sample_data/reference.png to run webcam conveyor mode.")
        return

    inspector = PartInspector('sample_data/reference.png')

    def inspector_callable(frame):
        # PartInspector.inspect returns (metrics, ann)
        return inspector.inspect(frame)

    def reject_cb(signal):
        print("[REJECT_CB] Signal:", signal)

    conv = ConveyorLine(inspector_callable, src=0, reject_callback=reject_cb)
    conv.start()

    print("Conveyor simulator running. Press Ctrl+C to stop.")
    try:
        while True:
            # main thread can show the latest frame in a small window
            frame = conv.vs.read()
            if frame is not None:
                cv2.imshow('Conveyor Live', frame)
            if cv2.waitKey(50) & 0xFF == ord('q'):
                break
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        conv.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_webcam_conveyor()
