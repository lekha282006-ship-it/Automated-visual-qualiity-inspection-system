import os
from celery import Celery
from typing import Optional
import time
import json

# Celery configuration via env
CELERY_BROKER = os.environ.get('CELERY_BROKER_URL')
CELERY_BACKEND = os.environ.get('CELERY_RESULT_BACKEND')

celery_app: Optional[Celery] = None
if CELERY_BROKER:
    celery_app = Celery('avq_tasks', broker=CELERY_BROKER, backend=CELERY_BACKEND)


def init_celery():
    global celery_app
    if celery_app is None and CELERY_BROKER:
        celery_app = Celery('avq_tasks', broker=CELERY_BROKER, backend=CELERY_BACKEND)
    return celery_app


def inspect_sync(inspector, image_path: str):
    # simple synchronous wrapper
    start = time.time()
    metrics, ann = inspector.inspect(image_path)
    duration = time.time() - start
    return {'metrics': metrics, 'duration': duration}


if celery_app is not None:
    @celery_app.task(name='avq.inspect_image')
    def inspect_image_task(image_path: str, inspector_pickle: str = ''):
        # A production task should import inspector from a shared location or reconstruct the object
        # For now this task is a placeholder to show wiring. It expects a module-level inspector to be available.
        return {'status': 'queued', 'image': image_path}
