# 🚀 Google Cloud Platform Setup Guide

This guide walks you through setting up the GCP infrastructure needed to deploy the Iris API to production.

## ⏱️ Time Required
- **First time**: 20-30 minutes
- **Experienced**: 10-15 minutes

## 📋 Before You Start

### Required Tools
```bash
# Install Google Cloud CLI
curl https://sdk.cloud.google.com | bash
exec -l $SHELL  # Restart shell

# Install kubectl
gcloud components install kubectl

# Verify installations
gcloud --version
kubectl version --client
```

### Required Accounts/Access
- Google Cloud account with billing enabled
- Admin access to your Google Cloud project
- Admin access to your GitHub repository (for adding secrets)

## 🏗️ Step-by-Step Setup

### Step 1: Configure Your Project 🎯

```bash
# Replace with your actual project ID
export PROJECT_ID="your-project-id-here"

# Set the project and verify
gcloud config set project $PROJECT_ID
gcloud config get-value project  # Should show your project ID

# Enable required APIs (takes 2-3 minutes)
echo "Enabling required APIs..."
gcloud services enable container.googleapis.com
gcloud services enable containerregistry.googleapis.com
gcloud services enable artifactregistry.googleapis.com

echo "✅ APIs enabled successfully!"
```

### Step 2: Create Kubernetes Cluster ⚙️

```bash
# Create the cluster (takes 5-10 minutes)
echo "Creating GKE cluster (this will take several minutes)..."
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

echo "✅ Cluster created successfully!"
```

### Step 3: Set Up Container Registry 📦

```bash
# Create Docker repository
gcloud artifacts repositories create iris-api \
    --repository-format=docker \
    --location=us-central1 \
    --description="Iris API Docker repository"

# Configure Docker authentication
gcloud auth configure-docker us-central1-docker.pkg.dev

echo "✅ Artifact Registry configured!"
```

### Step 4: Create Service Account 🔐

```bash
# Create service account for GitHub Actions
gcloud iam service-accounts create github-actions \
    --display-name="GitHub Actions Service Account" \
    --description="Service account for GitHub Actions CI/CD"

# Grant required permissions
echo "Granting permissions..."
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:github-actions@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/container.developer"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:github-actions@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/artifactregistry.writer"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:github-actions@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/container.clusterAdmin"

# Create and download the key
gcloud iam service-accounts keys create github-actions-key.json \
    --iam-account=github-actions@$PROJECT_ID.iam.gserviceaccount.com

echo "✅ Service account created! Key saved to github-actions-key.json"
```

### Step 5: Configure GitHub Secrets 🔑

**In your GitHub repository:**

1. Go to **Settings** → **Secrets and variables** → **Actions**
2. Click **New repository secret** and add:

| Secret Name | Value | Where to find it |
|-------------|-------|------------------|
| `GCP_PROJECT_ID` | `your-project-id` | The PROJECT_ID you set above |
| `GCP_SA_KEY` | `{entire JSON content}` | Copy ALL contents of `github-actions-key.json` |

**⚠️ Important**: For `GCP_SA_KEY`, copy the ENTIRE contents of the JSON file, including the outer `{` and `}`.

### Step 6: Test the Setup ✅

```bash
# Update deployment with your project ID
sed -i "s|PROJECT_ID|$PROJECT_ID|g" k8s/deployment.yaml

# Apply the Kubernetes manifests manually (first time only)
kubectl apply -f k8s/deployment.yaml

# Check if everything is working
echo "Checking deployment status..."
kubectl get pods -l app=iris-api
kubectl get services

echo "✅ Setup complete! Push to main branch to trigger automatic deployment."
```

## 🔍 Verification & Testing

### Check Your Deployment
```bash
# 1. Verify pods are running
kubectl get pods -l app=iris-api
# Expected: 2 pods in "Running" status

# 2. Check service status  
kubectl get service iris-api-service
# Expected: External IP assigned (may take 2-5 minutes)

# 3. Test the API
EXTERNAL_IP=$(kubectl get service iris-api-service -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
echo "Testing API at: http://$EXTERNAL_IP"
curl http://$EXTERNAL_IP/health

# Expected response:
# {"status":"healthy","model_loaded":true,"feature_names":[...],"classes":[...]}
```

### View Application Logs
```bash
# Real-time logs from all pods
kubectl logs -f deployment/iris-api

# Logs from specific pod
kubectl logs -l app=iris-api --tail=50
```

## 🔧 Operations & Maintenance

### Scaling
```bash
# Scale to more replicas
kubectl scale deployment iris-api --replicas=5

# Check autoscaling status
kubectl get hpa iris-api-hpa
```

### Updates
After pushing to main branch, GitHub Actions handles updates automatically. Monitor with:
```bash
# Watch deployment rollout
kubectl rollout status deployment/iris-api

# Check deployment history
kubectl rollout history deployment/iris-api
```

### Manual Update (if needed)
```bash
# Update to specific image
kubectl set image deployment/iris-api iris-api=us-central1-docker.pkg.dev/$PROJECT_ID/iris-api/iris-api:latest

# Restart deployment
kubectl rollout restart deployment/iris-api
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

## ❗ Troubleshooting

### Common Setup Issues

#### "Permission denied" errors
```bash
# Make sure you're authenticated
gcloud auth login
gcloud auth application-default login

# Verify your project
gcloud config get-value project
```

#### "Cluster not found" errors
```bash
# Make sure you're connected to the right cluster
gcloud container clusters get-credentials iris-cluster --region=us-central1

# Verify connection
kubectl cluster-info
```

#### "Pods stuck in Pending state"
```bash
# Check node resources
kubectl get nodes
kubectl describe nodes

# Check pod events
kubectl describe pods -l app=iris-api
```

#### "External IP stuck on <pending>"
```bash
# Check service events
kubectl describe service iris-api-service

# This is normal for first 2-5 minutes
# If stuck longer, check Google Cloud Console → Network Services → Load Balancing
```

#### "GitHub Actions failing"
1. **Check secrets**: Go to GitHub repo → Settings → Secrets → Actions
   - `GCP_PROJECT_ID` should be your project ID
   - `GCP_SA_KEY` should be the FULL JSON content (including braces)

2. **Check service account permissions**:
   ```bash
   # Verify service account exists
   gcloud iam service-accounts list | grep github-actions
   
   # Check permissions
   gcloud projects get-iam-policy $PROJECT_ID --flatten="bindings[].members" --filter="bindings.members:github-actions@$PROJECT_ID.iam.gserviceaccount.com"
   ```

### Getting Help
```bash
# Check all resources
kubectl get all -l app=iris-api

# View events
kubectl get events --sort-by=.metadata.creationTimestamp

# Debug pod issues
kubectl describe pods -l app=iris-api
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