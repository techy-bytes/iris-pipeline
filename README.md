# Iris Classification Pipeline

A complete CI/CD pipeline for training and deploying an Iris classification model as a REST API using Docker and Kubernetes on Google Cloud Platform.

## Features

- **Machine Learning Pipeline**: Scikit-learn based Iris classification model
- **REST API**: FastAPI-based service for model predictions
- **Containerization**: Docker support for consistent deployments
- **Kubernetes Deployment**: Scalable deployment on Google Kubernetes Engine (GKE)
- **CI/CD Pipeline**: Automated testing, building, and deployment using GitHub Actions and CML
- **Comprehensive Testing**: Unit tests for data validation, model evaluation, and API endpoints

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   GitHub Repo   │───▶│   GitHub Actions │───▶│   Google Cloud  │
│                 │    │                  │    │                 │
│  - Source Code  │    │  - Run Tests     │    │  - GKE Cluster  │
│  - Tests        │    │  - Build Docker  │    │  - Artifact     │
│  - Dockerfile   │    │  - Deploy to K8s │    │    Registry     │
│  - K8s Manifests│    │  - CML Reports   │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Project Structure

```
iris-pipeline/
├── src/
│   ├── train.py          # Model training script
│   └── api.py            # FastAPI application
├── tests/
│   ├── test_data_validation.py    # Data quality tests
│   ├── test_model_evaluation.py   # Model performance tests
│   └── test_api.py                # API endpoint tests
├── k8s/
│   └── deployment.yaml   # Kubernetes deployment manifests
├── data/
│   └── iris.csv          # Training dataset
├── .github/workflows/
│   └── ci.yml            # CI/CD pipeline configuration
├── Dockerfile            # Container configuration
├── requirements.txt      # Python dependencies
└── Required_Commands.md  # GCP setup instructions
```

## API Endpoints

- `GET /` - Health check endpoint
- `GET /health` - Detailed health check with model status
- `POST /predict` - Single prediction endpoint
- `POST /predict_batch` - Batch prediction endpoint

### Example Usage

```bash
# Health check
curl http://your-service-ip/health

# Single prediction
curl -X POST http://your-service-ip/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  }'

# Batch prediction
curl -X POST http://your-service-ip/predict_batch \
  -H "Content-Type: application/json" \
  -d '[
    {
      "sepal_length": 5.1,
      "sepal_width": 3.5,
      "petal_length": 1.4,
      "petal_width": 0.2
    },
    {
      "sepal_length": 6.0,
      "sepal_width": 3.0,
      "petal_length": 4.5,
      "petal_width": 1.5
    }
  ]'
```

## Local Development

### Prerequisites

- Python 3.10+
- Docker (optional)
- kubectl (for Kubernetes deployment)

### Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run tests:
   ```bash
   pytest
   ```

4. Train the model:
   ```bash
   python src/train.py
   ```

5. Start the API server:
   ```bash
   uvicorn src.api:app --host 0.0.0.0 --port 8000
   ```

### Docker Development

1. Build the image:
   ```bash
   docker build -t iris-api .
   ```

2. Run the container:
   ```bash
   docker run -p 8000:8000 iris-api
   ```

## Deployment

### GitHub Actions CI/CD

The project includes a complete CI/CD pipeline that:

1. **Testing Stage**: Runs all tests and generates reports using CML
2. **Building Stage**: Builds and pushes Docker images to Google Artifact Registry
3. **Deployment Stage**: Deploys to Google Kubernetes Engine (GKE)

### Manual GCP Setup

Follow the instructions in `Required_Commands.md` to set up:

- Google Kubernetes Engine cluster
- Artifact Registry for container images
- Service accounts and IAM permissions
- GitHub secrets for CI/CD

### Kubernetes Deployment

The application is deployed using:

- **Deployment**: 2 replicas with health checks and resource limits
- **Service**: LoadBalancer service for external access
- **HPA**: Horizontal Pod Autoscaler for automatic scaling

## Monitoring and Operations

### Health Checks

- Kubernetes liveness and readiness probes
- Custom health endpoint with model status
- Resource monitoring and alerts

### Scaling

- Horizontal Pod Autoscaler based on CPU and memory usage
- Cluster autoscaling for node management
- Manual scaling commands available

### Logging

```bash
# View application logs
kubectl logs -l app=iris-api

# Follow logs in real-time
kubectl logs -f deployment/iris-api
```

## CI/CD Pipeline Features

### Continuous Integration (GitHub)

- **Automated Testing**: Run data validation, model evaluation, and API tests
- **Code Quality**: Linting and formatting checks
- **CML Reports**: Automated model performance reports on pull requests
- **Docker Build**: Containerization and registry push

### Continuous Deployment (GitHub + GCP)

- **Automated Deployment**: Deploy to GKE on main branch pushes
- **Rolling Updates**: Zero-downtime deployments
- **Deployment Reports**: CML-generated deployment status reports
- **Rollback Capability**: Built-in Kubernetes rollback support

## Security

- Container image vulnerability scanning
- Kubernetes RBAC implementation
- Service account with minimal required permissions
- Secrets management for sensitive configuration

## Performance

- Efficient model loading and caching
- Batch prediction support for high throughput
- Horizontal pod autoscaling
- Optimized Docker image with multi-stage builds

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes and add tests
4. Submit a pull request

The CI/CD pipeline will automatically run tests and generate reports using CML.

## License

This project is licensed under the MIT License.