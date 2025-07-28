# Iris ML Service - Logging & Telemetry Demo

This implementation of the Iris ML service includes comprehensive logging and telemetry capabilities as demonstrated by the professor, with enhancements for Google Cloud Platform integration.

## Telemetry Features

### 🔍 OpenTelemetry Tracing
- **Distributed tracing** with Google Cloud Trace integration
- **Trace correlation** across all requests and operations
- **Performance monitoring** with automatic latency measurements
- **Fallback support** to console output for local development

### 📊 Structured Logging
- **Google Cloud Logging** integration when deployed to GKE
- **JSON structured logs** for better searchability and analysis
- **Trace ID correlation** in all log entries for debugging
- **Error tracking** with full exception details

### 🛡️ Exception Handling
- **Comprehensive error handling** with proper HTTP status codes
- **Trace context preservation** in error scenarios
- **Structured error logging** for observability
- **Graceful degradation** when services are unavailable

## Quick Start

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run the service
uvicorn src.api:app --host 0.0.0.0 --port 8000

# Test with telemetry demo
python test_telemetry.py
```

### Google Cloud Deployment
```bash
# Set your project ID
export PROJECT_ID=your-gcp-project-id

# Run the complete deployment script
./deploy.sh
```

The deployment script will:
1. Enable required GCP APIs (Container, Logging, Monitoring, Cloud Trace)
2. Build and push Docker image to Artifact Registry
3. Create GKE cluster with workload identity and observability features
4. Set up service accounts with proper IAM roles
5. Deploy the application with telemetry configuration

### Performance Testing
```bash
# Install wrk (if not already installed)
sudo apt-get install wrk

# Run performance test (replace EXTERNAL_IP with your service IP)
wrk -t4 -c100 -d30s --latency -s k8s/post.lua http://EXTERNAL_IP:80/predict
```

## Architecture

### Telemetry Stack
- **OpenTelemetry SDK** for trace collection and export
- **Google Cloud Trace** for distributed tracing visualization
- **Google Cloud Logging** for centralized log management
- **GKE Workload Identity** for secure GCP service access

### Service Configuration
- **Kubernetes ServiceAccount**: `telemetry-access`
- **GCP Service Account**: `telemetry-access@PROJECT_ID.iam.gserviceaccount.com`
- **Required IAM Roles**: 
  - `roles/logging.logWriter`
  - `roles/cloudtrace.agent`

### Environment Variables
- `GOOGLE_CLOUD_PROJECT` / `GCP_PROJECT`: Enables GCP telemetry features
- `PYTHONUNBUFFERED=1`: Ensures immediate log output

## Monitoring & Observability

When deployed to GKE, you can monitor the service through:

1. **Google Cloud Trace**: View distributed traces and performance
2. **Google Cloud Logging**: Search and analyze structured logs
3. **Google Cloud Monitoring**: Monitor metrics and set up alerts
4. **Kubernetes Dashboard**: View pod health and resource usage

## API Endpoints

- `GET /`: Service health check
- `GET /health`: Detailed health status with model information
- `POST /predict`: Single prediction with full telemetry
- `POST /predict_batch`: Batch predictions with performance metrics

All endpoints include:
- Automatic trace generation
- Performance timing
- Structured logging
- Error handling with trace correlation

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Test telemetry integration
python test_telemetry.py http://your-service-url
```

This implementation provides a production-ready ML service with enterprise-grade observability features suitable for Google Cloud Platform deployment.