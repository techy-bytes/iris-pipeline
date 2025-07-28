# Load Testing with Locust

This directory contains configuration for load testing the Iris ML API using Locust.

## Quick Start

### Install Locust

```bash
pip install locust
```

### Run Load Test Locally

1. Start the Iris API service:
```bash
cd /path/to/iris-pipeline
python src/api.py
```

2. Run Locust in another terminal:
```bash
cd /path/to/iris-pipeline
locust -f locustfile.py --host=http://localhost:8000
```

3. Open your browser to http://localhost:8089 to access the Locust web UI

### Run Load Test Against Kubernetes Deployment

```bash
# Get the external IP of your Kubernetes service
kubectl get service iris-api-service

# Run Locust against the external endpoint
locust -f locustfile.py --host=http://<EXTERNAL-IP>
```

## Load Testing Configuration

The `locustfile.py` contains two user classes:

### IrisAPIUser (Main Load)
- **Weight**: 3 (most users will be of this type)
- **Wait Time**: 1-3 seconds between requests
- **Tasks**:
  - Health check (30% of requests)
  - Single predictions (60% of requests) 
  - Batch predictions (20% of requests)
  - Root endpoint (10% of requests)

### HighLoadUser (Stress Testing)
- **Weight**: 1 (fewer of these users)
- **Wait Time**: 0.1-0.5 seconds between requests
- **Tasks**:
  - Rapid single predictions only

## Performance Optimizations for Load Testing

### Environment Variables

Set `LOAD_TEST_MODE=true` to optimize the API for load testing:

```bash
# For Docker/Kubernetes deployment
export LOAD_TEST_MODE=true

# This will:
# - Reduce logging verbosity to ERROR level only (more aggressive than before)
# - Optimize telemetry batch processing with larger queues and less frequent exports
# - Skip detailed prediction logging during high load
# - Reduce console telemetry output frequency for local development
```

### GCP Integration Optimizations

When running in GCP with `LOAD_TEST_MODE=true`:

- **Logging**: Only ERROR level messages sent to Cloud Logging
- **Telemetry**: Optimized batch processing (4096 queue, 1024 batch size, 5s export interval)
- **Service Identification**: Proper service name "iris-ml-service" for better filtering

### Kubernetes Load Test Deployment

Use the dedicated load testing configuration:

```bash
# Deploy optimized configuration for load testing
envsubst < k8s/deployment-loadtest.yaml | kubectl apply -f -

# This deployment includes:
# - LOAD_TEST_MODE=true by default
# - More replicas (3 instead of 2)
# - Higher resource limits (1Gi memory, 1000m CPU)
# - Optimized for high throughput
```

### Telemetry Settings

The API includes optimized telemetry settings for different environments:

#### Google Cloud Trace (GCP Environment):
- **Standard Mode**: 
  - Queue size: 2048 spans
  - Batch size: 512 spans  
  - Export interval: 1 second
- **Load Test Mode**: 
  - Queue size: 4096 spans
  - Batch size: 1024 spans
  - Export interval: 5 seconds (less frequent for better performance)

#### Console Trace (Local Development):
- **Standard Mode**:
  - Queue size: 1024 spans
  - Batch size: 128 spans
  - Export interval: 2 seconds
- **Load Test Mode**:
  - Queue size: 512 spans
  - Batch size: 64 spans  
  - Export interval: 10 seconds (minimal console output)

#### Service Identification:
- Service name: "iris-ml-service"
- Service version: "1.0.0"
- Enhanced GCP environment detection via metadata server

## Sample Load Test Scenarios

### Light Load Test
```bash
locust -f locustfile.py --host=http://localhost:8000 --users=10 --spawn-rate=2 --run-time=60s --headless
```

### Medium Load Test  
```bash
locust -f locustfile.py --host=http://localhost:8000 --users=50 --spawn-rate=5 --run-time=300s --headless
```

### Heavy Load Test
```bash
locust -f locustfile.py --host=http://localhost:8000 --users=200 --spawn-rate=10 --run-time=600s --headless
```

## Monitoring During Load Tests

### View Telemetry in Google Cloud

1. **Cloud Trace**: https://console.cloud.google.com/traces
   - View distributed traces and latency
   - Analyze performance bottlenecks

2. **Cloud Logging**: https://console.cloud.google.com/logs
   - Filter by `resource.type="k8s_container"`
   - Search for error events

### Key Metrics to Monitor

- **Request Rate**: Requests per second
- **Response Time**: 95th percentile latency should be < 100ms
- **Error Rate**: Should be < 1%
- **Memory Usage**: Monitor pod memory consumption
- **CPU Usage**: Monitor pod CPU utilization

## Troubleshooting

### High Latency
- Check if `LOAD_TEST_MODE=true` is set
- Verify Kubernetes resource limits aren't being hit
- Review Cloud Trace for bottlenecks

### Memory Issues
- Monitor pod memory usage: `kubectl top pods`
- Check for memory leaks in telemetry spans
- Consider increasing pod memory limits

### Connection Errors
- Verify service is accessible: `kubectl get service iris-api-service`
- Check pod health: `kubectl get pods`
- Review pod logs: `kubectl logs -f deployment/iris-api`

## Advanced Configuration

### Custom User Patterns

You can create custom user classes in `locustfile.py`:

```python
class CustomUser(HttpUser):
    wait_time = between(2, 5)
    
    @task
    def custom_workflow(self):
        # Your custom test workflow
        pass
```

### Test Data Variations

The locustfile includes realistic iris data samples for all three species (setosa, versicolor, virginica) to ensure comprehensive testing.