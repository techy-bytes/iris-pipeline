# Iris Classification Pipeline

A complete **CI/CD pipeline** for training and deploying an Iris **classification** model as a **REST API** using **Docker** and **Kubernetes** on Google Cloud Platform.

> **✅ Project Status**:- Fully functional and ready to use! All tests pass, model training works, API responds correctly.

## 🚀 **[📋 START HERE: Getting Started Checklist](./GETTING_STARTED.md)**

**New to this project?** Follow the step-by-step checklist above to get running in minutes.

## Quick Reference

### Local Development (5 minutes)
```bash
pip install -r requirements.txt
pytest                              # All tests should pass
python src/train.py                 # Train the model  
uvicorn src.api:app --port 8000     # Start API
curl localhost:8000/health          # Test it works
```

### Production Deployment (30 minutes)
1. **Complete GCP setup** → [`Required_Commands.md`](./Required_Commands.md)
2. **Add GitHub secrets** → `GCP_PROJECT_ID` and `GCP_SA_KEY`  
3. **Deploy** → `git push origin main`

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

## 🚧 Production Deployment Guide

### Step-by-Step GCP Deployment

#### Prerequisites
- Google Cloud account with billing enabled
- `gcloud` CLI installed and configured
- `kubectl` installed

#### Step 1: Initial GCP Setup (One-time)
Follow the detailed instructions in [`Required_Commands.md`](./Required_Commands.md) to set up:

1. **Enable APIs**: Container, Artifact Registry, Kubernetes APIs
2. **Create GKE Cluster**: Regional cluster with autoscaling
3. **Create Artifact Registry**: For storing Docker images
4. **Create Service Account**: With necessary permissions for CI/CD
5. **Download Service Account Key**: For GitHub Actions

#### Step 2: Configure GitHub Repository
1. Go to your GitHub repository → Settings → Secrets and variables → Actions
2. Add these secrets:
   - `GCP_PROJECT_ID`: Your Google Cloud project ID
   - `GCP_SA_KEY`: Contents of the service account JSON key file

#### Step 3: Deploy
```bash
# Option A: Automatic deployment (Recommended)
git push origin main  # GitHub Actions will handle everything

# Option B: Manual deployment
# Follow the manual deployment steps in Required_Commands.md
```

#### Step 4: Verify Deployment
```bash
# Check if pods are running
kubectl get pods -l app=iris-api

# Get external IP (may take a few minutes)
kubectl get service iris-api-service

# Test the deployed API
curl http://EXTERNAL_IP/health
```

### CI/CD Pipeline Overview

The automated pipeline triggered on every push to `main`:

1. **Test Stage**: Runs all tests, generates CML reports
2. **Build Stage**: Builds Docker image, pushes to Artifact Registry  
3. **Deploy Stage**: Updates Kubernetes deployment with new image

### Deployment Architecture

- **Kubernetes Deployment**: 2-10 replicas with auto-scaling
- **Load Balancer**: External access via Google Cloud Load Balancer
- **Health Checks**: Liveness and readiness probes
- **Resource Limits**: CPU and memory constraints for stability

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

## 🔧 Troubleshooting

### Common Issues and Solutions

#### "Tests are failing"
```bash
# Check if dependencies are installed
pip install -r requirements.txt

# Run individual test files to isolate issues
pytest tests/test_data_validation.py -v
pytest tests/test_model_evaluation.py -v
pytest tests/test_api.py -v
```

#### "Model not found" or "API won't start"
```bash
# Train the model first
python src/train.py

# Verify model files are created
ls -la *.joblib
# Should see: model.joblib, label_encoder.joblib
```

#### "GitHub Actions deployment failing"
1. **Check secrets**: Ensure `GCP_PROJECT_ID` and `GCP_SA_KEY` are set correctly
2. **Verify GCP setup**: Run through Required_Commands.md steps
3. **Check service account permissions**: Ensure it has Container Developer, Artifact Registry Writer, and Cluster Admin roles

#### "Can't access deployed API"
```bash
# Check if pods are running
kubectl get pods -l app=iris-api

# Check service status
kubectl get service iris-api-service

# If external IP is pending, wait a few minutes and check again
# If still pending, check Google Cloud Console for LoadBalancer issues
```

#### "Local development issues"
```bash
# Reset your environment
pip uninstall -r requirements.txt
pip install -r requirements.txt

# Clean up generated files
rm -f *.joblib *.csv

# Start fresh
python src/train.py
uvicorn src.api:app --reload
```

### Getting Help

1. **Check logs**: 
   ```bash
   # Local API logs
   uvicorn src.api:app --log-level debug
   
   # Kubernetes logs
   kubectl logs -l app=iris-api
   ```

2. **Validate your setup**:
   ```bash
   # Test the pipeline step by step
   pytest                    # Tests should pass
   python src/train.py       # Should create model files
   uvicorn src.api:app       # API should start
   curl localhost:8000/health # Should return healthy status
   ```

## 🎯 What to Do Next

### For Local Development
1. **Start Here**: Run the Quick Start commands above
2. **Make Changes**: Modify the model, API, or add features
3. **Test Everything**: `pytest` before committing
4. **Iterate**: Use `uvicorn src.api:app --reload` for live reloading

### For Production Deployment
1. **Set up GCP**: Complete all steps in `Required_Commands.md`
2. **Configure Secrets**: Add GitHub repository secrets
3. **Deploy**: Push to main branch and watch GitHub Actions
4. **Monitor**: Use `kubectl` commands to check deployment status

### For Team Collaboration  
1. **Fork the repo**: Create your own copy
2. **Create feature branches**: Don't work directly on main
3. **Submit PRs**: GitHub Actions will test your changes automatically
4. **Review CML reports**: Check model performance on every PR

### For Customization
1. **Change the model**: Edit `src/train.py` 
2. **Modify the API**: Update `src/api.py`
3. **Add features**: Create new endpoints or data processing
4. **Update tests**: Keep `tests/` in sync with your changes

### For Production Operations
1. **Monitor health**: Check `/health` endpoint regularly
2. **Scale as needed**: Use `kubectl scale deployment iris-api --replicas=N`
3. **Update the app**: Push to main branch for automatic deployment
4. **View logs**: `kubectl logs -f deployment/iris-api`

## 📚 Additional Resources

- **Detailed GCP Setup**: [`Required_Commands.md`](./Required_Commands.md)
- **CI/CD Pipeline**: [`.github/workflows/ci.yml`](./.github/workflows/ci.yml)
- **Kubernetes Config**: [`k8s/deployment.yaml`](./k8s/deployment.yaml)
- **API Documentation**: Visit `http://your-api-url/docs` for interactive Swagger docs

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes and add tests
4. Submit a pull request

The CI/CD pipeline will automatically run tests and generate reports using CML.

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

## License

This project is licensed under the MIT License.
