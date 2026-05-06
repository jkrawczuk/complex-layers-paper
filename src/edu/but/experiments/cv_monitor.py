import os
import resource
import sys
import threading
import time

import psutil


def start_resource_monitor(interval_sec: float, label: str = "cv_run"):
    process = psutil.Process(os.getpid())
    stop_event = threading.Event()
    process.cpu_percent(interval=None)

    def _monitor_loop():
        while not stop_event.wait(interval_sec):
            mem = process.memory_info()
            rss_gb = mem.rss / (1024 ** 3)
            cpu_pct = process.cpu_percent(interval=None)
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            print(
                f"[{timestamp}] [monitor:{label}] "
                f"rss_gb={rss_gb:.2f} cpu_pct={cpu_pct:.1f}"
            )

    thread = threading.Thread(target=_monitor_loop, daemon=True)
    thread.start()
    return stop_event, thread


def max_rss_bytes_from_rusage() -> int:
    ru_maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # On macOS ru_maxrss is bytes; on Linux it is kilobytes.
    if sys.platform == "darwin":
        return int(ru_maxrss)
    return int(ru_maxrss * 1024)
