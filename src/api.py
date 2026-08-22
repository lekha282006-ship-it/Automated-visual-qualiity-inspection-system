from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from starlette.responses import JSONResponse
import os
from typing import Dict
from src.tasks import init_celery, inspect_sync
from src.logging_config import get_logger

logger = get_logger('api')

app = FastAPI(title='Automated Visual QC API')


@app.post('/enqueue')
async def enqueue_inspection(background: BackgroundTasks, file: UploadFile = File(...)):
    # save uploaded file
    os.makedirs('tmp', exist_ok=True)
    path = os.path.join('tmp', file.filename)
    with open(path, 'wb') as f:
        f.write(await file.read())

    celery = init_celery()
    if celery is not None:
        # enqueue task (placeholder)
        res = celery.send_task('avq.inspect_image', args=[path])
        logger.info('Enqueued %s -> %s', path, res.id)
        return JSONResponse({'status': 'enqueued', 'task_id': str(res.id)})
    else:
        # run synchronously
        logger.info('Running sync inspection for %s', path)
        # user of API should supply inspector instance or config; here we run a placeholder
        result = inspect_sync(None, path)
        return JSONResponse({'status': 'done', 'result': result})
