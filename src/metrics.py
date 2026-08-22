from prometheus_client import start_http_server, Counter, Histogram
import threading
import os
import time

# Metrics
inspections_total = Counter('avq_inspections_total', 'Total number of inspections')
inspections_failed = Counter('avq_inspections_failed', 'Number of failed inspections')
inspection_duration = Histogram('avq_inspection_seconds', 'Inspection duration seconds')


def _start_server(port: int = 8000):
    # start_http_server blocks; run in background
    start_http_server(port)


def start_metrics_server(port: int = None):
    port = port or int(os.environ.get('METRICS_PORT', '8000'))
    t = threading.Thread(target=_start_server, args=(port,), daemon=True)
    t.start()
    # small sleep to allow server to start
    time.sleep(0.1)
