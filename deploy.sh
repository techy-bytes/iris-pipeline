#!/bin/bash

# Logging & Telemetry demo deployment script
# Based on professor's demo requirements

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Iris ML Service Telemetry Demo Deployment${NC}"
echo "=============================================="

# Check if PROJECT_ID is set
if [ -z "$PROJECT_ID" ]; then
    PROJECT_ID=$(gcloud config get-value project 2>/dev/null || echo "")
    if [ -z "$PROJECT_ID" ]; then
        echo -e "${RED}❌ PROJECT_ID not set. Please set it or configure gcloud project${NC}"
        exit 1
    fi
fi

echo -e "${YELLOW}📋 Using PROJECT_ID: $PROJECT_ID${NC}"

echo -e "${YELLOW}🔧 Step 1: Enable required APIs${NC}"
gcloud services enable container.googleapis.com \
    logging.googleapis.com \
    monitoring.googleapis.com \
    cloudtrace.googleapis.com

echo -e "${YELLOW}🐳 Step 2: Build and push Docker image${NC}"
docker build -t iris-api .
docker tag iris-api us-central1-docker.pkg.dev/$PROJECT_ID/iris-api/iris-api:latest
docker push us-central1-docker.pkg.dev/$PROJECT_ID/iris-api/iris-api:latest

echo -e "${YELLOW}☸️  Step 3: Create GKE cluster (if needed)${NC}"
if ! gcloud container clusters describe demo-log-ml-cluster --zone=us-central1-a &>/dev/null; then
    gcloud container clusters create demo-log-ml-cluster \
        --zone=us-central1-a \
        --num-nodes=3 \
        --workload-pool=$PROJECT_ID.svc.id.goog \
        --logging=SYSTEM,WORKLOAD \
        --monitoring=SYSTEM
else
    echo "Cluster demo-log-ml-cluster already exists"
fi

echo -e "${YELLOW}🔐 Step 4: Create service account (if needed)${NC}"
if ! gcloud iam service-accounts describe telemetry-access@$PROJECT_ID.iam.gserviceaccount.com &>/dev/null; then
    gcloud iam service-accounts create telemetry-access \
        --display-name "Access for GKE ML service"
    
    gcloud projects add-iam-policy-binding $PROJECT_ID \
        --member="serviceAccount:telemetry-access@$PROJECT_ID.iam.gserviceaccount.com" \
        --role="roles/logging.logWriter"
    
    gcloud projects add-iam-policy-binding $PROJECT_ID \
        --member="serviceAccount:telemetry-access@$PROJECT_ID.iam.gserviceaccount.com" \
        --role="roles/cloudtrace.agent"
else
    echo "Service account telemetry-access already exists"
fi

echo -e "${YELLOW}☸️  Step 5: Configure Kubernetes service account${NC}"
gcloud container clusters get-credentials demo-log-ml-cluster --zone=us-central1-a

kubectl create serviceaccount telemetry-access --namespace default --dry-run=client -o yaml | kubectl apply -f -

kubectl annotate serviceaccount telemetry-access \
    --namespace default \
    iam.gke.io/gcp-service-account=telemetry-access@$PROJECT_ID.iam.gserviceaccount.com \
    --overwrite

gcloud iam service-accounts add-iam-policy-binding telemetry-access@$PROJECT_ID.iam.gserviceaccount.com \
    --role roles/iam.workloadIdentityUser \
    --member "serviceAccount:$PROJECT_ID.svc.id.goog[default/telemetry-access]"

echo -e "${YELLOW}🚀 Step 6: Deploy application${NC}"
# Replace PROJECT_ID in deployment files
envsubst < k8s/deployment.yaml | kubectl apply -f -
kubectl apply -f k8s/hpa.yaml

echo -e "${YELLOW}⏳ Waiting for deployment to be ready...${NC}"
kubectl rollout status deployment iris-api

echo -e "${GREEN}✅ Deployment completed successfully!${NC}"

echo -e "${YELLOW}📊 Getting service information:${NC}"
kubectl get service iris-api-service

echo -e "${YELLOW}🔍 Pod status:${NC}"
kubectl get pods -l app=iris-api

echo -e "${YELLOW}📋 Service account verification:${NC}"
kubectl get serviceaccount telemetry-access -n default
kubectl describe serviceaccount telemetry-access -n default

echo -e "${GREEN}🎉 Deployment complete! Use the external IP to test the service.${NC}"
echo -e "${YELLOW}💡 Test with: curl -X POST http://EXTERNAL_IP:80/predict -H \"Content-Type: application/json\" -d '{\"sepal_length\": 5.1, \"sepal_width\": 3.5, \"petal_length\": 1.4, \"petal_width\": 0.2}'${NC}"
echo -e "${YELLOW}⚡ Performance test: wrk -t4 -c100 -d30s --latency -s k8s/post.lua http://EXTERNAL_IP:80/predict${NC}"