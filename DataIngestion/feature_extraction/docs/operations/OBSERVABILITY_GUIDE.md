# Connection Pool Observability Guide

This guide documents the observability and back-pressure features added to the thread-safe connection pool implementation.

## Overview

The connection pool now provides comprehensive metrics collection and timeout mechanisms to help monitor pool health, identify bottlenecks, and prevent resource starvation.

## Key Features

### 1. Structured Metrics Collection

The pool automatically tracks:
- **Pool Configuration**: `min_size`, `max_size`
- **Current State**: `total_connections`, `used_connections`, `idle_connections`, `utilization_percent`
- **Performance Counters**: `total_requests`, `successful_acquisitions`, `failed_acquisitions`, `timeouts`, `success_rate`
- **Wait Times**: `total`, `max`, `average`, and `recent_average` wait times
- **Operational Metadata**: `uptime_seconds`, `last_updated`

### 2. Configurable Timeouts

Prevent threads from blocking indefinitely when the pool is exhausted:
```python
from db import configure_acquisition

# Set 30-second timeout for connection acquisition
configure_acquisition(timeout=30.0, use_queue=True)

# Or specify timeout per-request
conn = get_connection(timeout=5.0)
```

### 3. Fair Queuing System

Optional queue ensures fair ordering of connection requests:
```python
# Enable queue (default)
configure_acquisition(use_queue=True)

# Disable for competitive acquisition
configure_acquisition(use_queue=False)
```

### 4. Real-time Monitoring

Access metrics programmatically:
```python
from db import get_pool_metrics, log_pool_summary

# Get current metrics as dictionary
metrics = get_pool_metrics()
print(f"Pool utilization: {metrics['pool.utilization_percent']}%")
print(f"Average wait time: {metrics['pool.wait_time.avg']:.3f}s")

# Log human-readable summary
log_pool_summary()
# Output: Pool metrics: size=3/10 (30.0%), requests=150, success_rate=98.5%, avg_wait=0.025s
```

## Usage Examples

### Basic Monitoring

```python
from db import initialize_pool, get_connection, return_connection, get_pool_metrics

# Initialize pool with metrics enabled
initialize_pool(minconn=5, maxconn=20)

# Use connections
conn = get_connection()
# ... do work ...
return_connection(conn)

# Check pool health
metrics = get_pool_metrics()
if metrics['pool.utilization_percent'] > 80:
    logger.warning("Pool utilization high, consider increasing max_size")
```

### Custom Metrics Listeners

```python
from db import get_metrics_collector

def alert_on_high_wait_time(metrics):
    """Alert when average wait time exceeds threshold."""
    if metrics['pool.wait_time.recent_avg'] > 1.0:
        send_alert(f"High DB wait time: {metrics['pool.wait_time.recent_avg']:.2f}s")

# Register listener
collector = get_metrics_collector()
collector.add_listener(alert_on_high_wait_time)
```

### Timeout Handling

```python
from db import get_connection, configure_acquisition

# Configure aggressive timeout for critical paths
configure_acquisition(timeout=2.0)

try:
    conn = get_connection()
    # Process critical request
except TimeoutError:
    # Fall back to cache or return degraded response
    return cached_response()
finally:
    if conn:
        return_connection(conn)
```

### Performance Tuning

Use metrics to tune pool configuration:

```python
from db import get_pool_metrics, initialize_pool

# Monitor for a period
metrics = get_pool_metrics()

# Analyze patterns
if metrics['pool.timeouts'] > 0:
    # Increase pool size
    close_pool()
    initialize_pool(minconn=10, maxconn=30)
    
elif metrics['pool.utilization_percent'] < 20:
    # Reduce pool size to save resources
    close_pool()
    initialize_pool(minconn=2, maxconn=10)
```

## Metrics Reference

| Metric | Description | Use Case |
|--------|-------------|----------|
| `pool.utilization_percent` | Current usage as % of max_size | Capacity planning |
| `pool.wait_time.avg` | Average time to acquire connection | Performance monitoring |
| `pool.wait_time.max` | Maximum wait time observed | SLA compliance |
| `pool.success_rate` | % of successful acquisitions | Reliability tracking |
| `pool.timeouts` | Number of acquisition timeouts | Back-pressure tuning |

## Best Practices

1. **Set Appropriate Timeouts**: Balance between failing fast and allowing legitimate delays
   ```python
   # Web requests: fail fast
   configure_acquisition(timeout=5.0)
   
   # Batch jobs: more tolerant
   configure_acquisition(timeout=60.0)
   ```

2. **Monitor Key Metrics**: Set up alerts for:
   - Utilization > 80% (capacity issues)
   - Average wait time > 1s (performance degradation)
   - Success rate < 95% (reliability concerns)

3. **Use Queue for Fairness**: Enable queuing to prevent thread starvation
   ```python
   configure_acquisition(use_queue=True)
   ```

4. **Regular Health Checks**: Log summaries periodically
   ```python
   import threading
   
   def periodic_health_check():
       while True:
           time.sleep(300)  # Every 5 minutes
           log_pool_summary()
   
   threading.Thread(target=periodic_health_check, daemon=True).start()
   ```

## Troubleshooting

### High Wait Times
- Check `pool.utilization_percent` - if high, increase `maxconn`
- Look for long-running queries blocking connections
- Consider connection timeout settings in PostgreSQL

### Frequent Timeouts
- Increase acquisition timeout: `configure_acquisition(timeout=30.0)`
- Increase pool size: `initialize_pool(maxconn=50)`
- Check for connection leaks (connections not being returned)

### Queue Growing
- Monitor queue size in logs
- If consistently full, increase `maxconn` or optimize query performance
- Consider disabling queue for non-critical paths

## Integration with Monitoring Systems

Export metrics to external systems:

```python
from db import get_metrics_collector
import prometheus_client

# Prometheus example
pool_utilization = prometheus_client.Gauge(
    'db_pool_utilization_percent',
    'Database connection pool utilization'
)

def export_to_prometheus(metrics):
    pool_utilization.set(metrics['pool.utilization_percent'])

collector = get_metrics_collector()
collector.add_listener(export_to_prometheus)
```

## Migration from Previous Version

The observability features are backward compatible. Existing code continues to work, with metrics collection happening automatically. To leverage new features:

1. Add timeout configuration:
   ```python
   # Old
   conn = get_connection()
   
   # New (with timeout)
   conn = get_connection(timeout=10.0)
   ```

2. Add monitoring:
   ```python
   # After pool initialization
   if get_pool_metrics()['pool.utilization_percent'] > 90:
       logger.warning("Pool near capacity")
   ```

3. Enable periodic logging:
   ```python
   # In application startup
   schedule_periodic_task(log_pool_summary, interval=300)
   ```