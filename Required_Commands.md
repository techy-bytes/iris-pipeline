# Required Commands for GCP Setup

This document outlines the manual steps required to set up the Google Cloud Platform (GCP) infrastructure for deploying the Iris API.

## Prerequisites

1. **Google Cloud Account**: Ensure you have access to a Google Cloud project
2. **gcloud CLI**: Install and configure the Google Cloud CLI on your local machine
3. **kubectl**: Install kubectl for Kubernetes cluster management
4. **Docker**: Install Docker for local testing (optional)

## GCP Setup Steps

### 1. Project Setup

```bash
# Set your project ID
export PROJECT_ID="your-project-id"
gcloud config set project $PROJECT_ID

# Enable required APIs
gcloud services enable container.googleapis.com
gcloud services enable containerregistry.googleapis.com
gcloud services enable artifactregistry.googleapis.com
```

### 2. Create Google Kubernetes Engine (GKE) Cluster

```bash
# Create GKE cluster
gcloud container clusters create iris-cluster \
    --region=us-central1 \
    --node-pool=default-pool \
    --num-nodes=2 \
    --machine-type=e2-medium \
    --disk-size=20GB \
    --enable-autoscaling \
    --min-nodes=1 \
    --max-nodes=5 \
    --enable-autorepair \
    --enable-autoupgrade

# Get cluster credentials
gcloud container clusters get-credentials iris-cluster --region=us-central1
```

### 3. Set Up Artifact Registry

```bash
# Create Artifact Registry repository
gcloud artifacts repositories create iris-api \
    --repository-format=docker \
    --location=us-central1 \
    --description="Iris API Docker repository"

# Configure Docker to use Artifact Registry
gcloud auth configure-docker us-central1-docker.pkg.dev
```

### 4. Create Service Account for CI/CD

```bash
# Create service account
gcloud iam service-accounts create github-actions \
    --display-name="GitHub Actions Service Account" \
    --description="Service account for GitHub Actions CI/CD"

# Grant necessary permissions
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:github-actions@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/container.developer"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:github-actions@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/artifactregistry.writer"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:github-actions@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/container.clusterAdmin"

# Create and download service account key
gcloud iam service-accounts keys create github-actions-key.json \
    --iam-account=github-actions@$PROJECT_ID.iam.gserviceaccount.com
```

### 5. GitHub Secrets Configuration

In your GitHub repository, go to Settings > Secrets and variables > Actions, and add the following secrets:

1. **GCP_PROJECT_ID**: Your Google Cloud project ID
2. **GCP_SA_KEY**: Contents of the `github-actions-key.json` file (the entire JSON content)

### 6. Initial Deployment (Manual)

For the first deployment, you may need to manually apply the Kubernetes manifests:

```bash
# Update the PROJECT_ID in the deployment manifest
sed -i "s|PROJECT_ID|$PROJECT_ID|g" k8s/deployment.yaml

# Apply the Kubernetes manifests
kubectl apply -f k8s/deployment.yaml

# Check deployment status
kubectl get pods -l app=iris-api
kubectl get services
```

### 7. Verify Setup

```bash
# Check if the service is running
kubectl get pods -l app=iris-api

# Get external IP (may take a few minutes)
kubectl get service iris-api-service

# Test the API (replace EXTERNAL_IP with actual IP)
curl http://EXTERNAL_IP/health
```

## Monitoring and Maintenance

### View Logs
```bash
# View application logs
kubectl logs -l app=iris-api

# Follow logs in real-time
kubectl logs -f deployment/iris-api
```

### Scale the Application
```bash
# Scale to 3 replicas
kubectl scale deployment iris-api --replicas=3

# Check scaling status
kubectl get pods -l app=iris-api
```

### Update the Application
The CI/CD pipeline will automatically update the application when changes are pushed to the main branch. You can also manually update:

```bash
# Update deployment image
kubectl set image deployment/iris-api iris-api=us-central1-docker.pkg.dev/$PROJECT_ID/iris-api/iris-api:latest

# Check rollout status
kubectl rollout status deployment/iris-api
```

## Cleanup

To remove all resources when no longer needed:

```bash
# Delete Kubernetes resources
kubectl delete -f k8s/deployment.yaml

# Delete GKE cluster
gcloud container clusters delete iris-cluster --region=us-central1

# Delete Artifact Registry repository
gcloud artifacts repositories delete iris-api --location=us-central1

# Delete service account
gcloud iam service-accounts delete github-actions@$PROJECT_ID.iam.gserviceaccount.com
```

## Troubleshooting

### Common Issues

1. **Authentication Issues**: Make sure the service account has the correct permissions
2. **Image Pull Errors**: Verify that the Artifact Registry is set up correctly and the image exists
3. **Pod Startup Issues**: Check pod logs using `kubectl logs`
4. **Service Not Accessible**: Ensure the LoadBalancer service has an external IP assigned

### Useful Commands

```bash
# Debug pod issues
kubectl describe pods -l app=iris-api

# Check events
kubectl get events --sort-by=.metadata.creationTimestamp

# Port forward for local testing
kubectl port-forward service/iris-api-service 8080:80
```

## Security Considerations

1. **Service Account Permissions**: Use the principle of least privilege
2. **Network Security**: Consider using private GKE clusters and VPC-native networking
3. **Secrets Management**: Use Kubernetes secrets or Google Secret Manager for sensitive data
4. **Image Security**: Regularly scan container images for vulnerabilities
5. **RBAC**: Implement Role-Based Access Control for Kubernetes resources

## Cost Optimization

1. **Node Pool Configuration**: Use appropriate machine types for your workload
2. **Autoscaling**: Enable cluster and pod autoscaling to optimize resource usage
3. **Resource Limits**: Set resource requests and limits for pods
4. **Preemptible Nodes**: Consider using preemptible nodes for cost savings
5. **Monitoring**: Use Google Cloud Monitoring to track resource usage and costs