import time
import functools
import atexit
import logging
import inspect
import random
import statistics
import threading
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class PerformanceAggregator:
    def __init__(self, maxlen=1000):
        self._metrics = defaultdict(lambda: deque(maxlen=maxlen))
        self._lock = threading.Lock()

    def record(self, name: str, duration_ms: float):
        with self._lock:
            self._metrics[name].append(duration_ms)

    def print_summary(self):
        if not self._metrics:
            return
            
        print("\n--- Performance Summary (Latency) ---")
        for name, latencies in self._metrics.items():
            if not latencies:
                continue
            
            avg = statistics.mean(latencies)
            # quantiles requires at least 2 data points. 
            if len(latencies) >= 20:
                p95 = statistics.quantiles(latencies, n=20)[18] 
            elif len(latencies) > 1:
                p95 = statistics.quantiles(latencies, n=10)[8] if len(latencies) >= 10 else max(latencies)
            else:
                p95 = max(latencies)
                
            max_lat = max(latencies)
            count = len(latencies)
            
            print(f"{name}: count={count} avg={avg:.2f}ms p95={p95:.2f}ms max={max_lat:.2f}ms")
        print("-------------------------------------\n")

_aggregator = PerformanceAggregator()

# Register the summary at exit
atexit.register(_aggregator.print_summary)

def profile_latency(sample_rate: float = 1.0):
    """
    Decorator to profile the latency of a function.
    Records duration in an in-memory ring buffer and prints a summary at exit.
    
    Args:
        sample_rate: Float between 0 and 1. Only sample a fraction of calls to reduce overhead.
    """
    def decorator(func):
        is_coroutine = inspect.iscoroutinefunction(func)

        if is_coroutine:
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                if sample_rate < 1.0 and random.random() > sample_rate:
                    return await func(*args, **kwargs)
                    
                start = time.perf_counter()
                try:
                    return await func(*args, **kwargs)
                finally:
                    duration_ms = (time.perf_counter() - start) * 1000
                    _aggregator.record(func.__name__, duration_ms)
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                if sample_rate < 1.0 and random.random() > sample_rate:
                    return func(*args, **kwargs)
                    
                start = time.perf_counter()
                try:
                    return func(*args, **kwargs)
                finally:
                    duration_ms = (time.perf_counter() - start) * 1000
                    _aggregator.record(func.__name__, duration_ms)
            return sync_wrapper

    return decorator
