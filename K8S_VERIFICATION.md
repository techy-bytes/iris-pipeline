# 🔍 Kubernetes Integration Verification Guide

This guide helps you verify that your Kubernetes deployment is working correctly after your Docker image has been successfully built and pushed to Google Artifact Registry.

## 📊 Your Current Status

✅ **Docker Image Built**: Your image `iris-api:580f2f508d3a883eeaceec51aac7a449b7794410` is successfully stored in:
- Project: `buoyant-mason-461116-d8`
- Location: `us-central1 (Iowa)` 
- Repository: `iris-api`
- Virtual size: `233.3 MB`
- Status: **Successfully pushed to Google Artifact Registry! 🎉**

**Next Step**: Verify your Kubernetes deployment is running this image correctly.

> 💡 **Success!** Your CI/CD pipeline worked perfectly - your Docker image was built and pushed to Google Artifact Registry. Now let's make sure your Kubernetes deployment picked up this new image and is serving it correctly.

---

## 🚀 Quick Verification Checklist

### Step 1: Connect to Your GKE Cluster
```bash
# Connect to your cluster
gcloud container clusters get-credentials iris-cluster --region=us-central1

# Verify connection
kubectl cluster-info
```
**Expected**: Should show cluster endpoint and core services running.

### Step 2: Check Deployment Status
```bash
# Check if your deployment is running
kubectl get deployments

# Check pods status
kubectl get pods -l app=iris-api

# Check services
kubectl get services
```
**Expected Output**:
```
NAME       READY   UP-TO-DATE   AVAILABLE   AGE
iris-api   2/2     2            2           5m

NAME                        READY   STATUS    RESTARTS   AGE
iris-api-7c8b9d4f6b-abc12   1/1     Running   0          5m
iris-api-7c8b9d4f6b-def34   1/1     Running   0          5m

NAME                TYPE           CLUSTER-IP      EXTERNAL-IP     PORT(S)        AGE
iris-api-service    LoadBalancer   10.96.123.45    35.123.45.67   80:32000/TCP   5m
```

### Step 3: Verify Your Image is Deployed
```bash
# Check which image version is running
kubectl describe deployment iris-api | grep Image

# Or get detailed pod information
kubectl get pods -l app=iris-api -o jsonpath='{.items[*].spec.containers[*].image}'
```
**Expected**: Should show your image with the SHA `580f2f508d3a883eeaceec51aac7a449b7794410`.

### Step 4: Test the Deployed API
```bash
# Get the external IP
EXTERNAL_IP=$(kubectl get service iris-api-service -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
echo "API URL: http://$EXTERNAL_IP"

# Test health endpoint
curl http://$EXTERNAL_IP/health

# Test prediction endpoint
curl -X POST http://$EXTERNAL_IP/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  }'
```
**Expected Responses**:
- Health: `{"status":"healthy","model_loaded":true,...}`
- Predict: `{"prediction":"setosa","confidence":0.99}`

---

## 🔧 Detailed Verification Commands

### Check Resource Usage and Health
```bash
# View resource consumption
kubectl top pods -l app=iris-api
kubectl top nodes

# Check horizontal pod autoscaler status
kubectl get hpa iris-api-hpa

# View deployment details
kubectl describe deployment iris-api
```

### Monitor Application Logs
```bash
# View recent logs from all pods
kubectl logs -l app=iris-api --tail=50

# Follow logs in real-time
kubectl logs -f deployment/iris-api

# Check logs from a specific pod
kubectl logs <pod-name>
```

### Verify Load Balancer and Networking
```bash
# Check service details
kubectl describe service iris-api-service

# Check ingress and networking
kubectl get endpoints iris-api-service

# Test internal connectivity
kubectl run test-pod --image=curlimages/curl --rm -it --restart=Never -- sh
# Inside the pod, run:
# curl http://iris-api-service/health
```

---

## 📈 Performance and Scaling Verification

### Check Autoscaling Behavior
```bash
# View current scaling metrics
kubectl get hpa iris-api-hpa -w

# Generate load to test autoscaling
kubectl run load-generator --image=busybox --rm -it --restart=Never -- sh
# Inside the pod, run:
# while true; do wget -q -O- http://iris-api-service/health; done
```

### Monitor Resource Limits
```bash
# Check if pods are hitting resource limits
kubectl describe pods -l app=iris-api | grep -A 10 "Limits\|Requests"

# View resource usage over time
kubectl top pods -l app=iris-api --containers
```

---

## 🎯 API Testing Suite

### Complete API Functionality Test
```bash
# Store the external IP
EXTERNAL_IP=$(kubectl get service iris-api-service -o jsonpath='{.status.loadBalancer.ingress[0].ip}')

echo "Testing API at: http://$EXTERNAL_IP"

# 1. Root endpoint
echo "1. Testing root endpoint:"
curl http://$EXTERNAL_IP/

# 2. Health check
echo -e "\n2. Testing health endpoint:"
curl http://$EXTERNAL_IP/health

# 3. Single prediction
echo -e "\n3. Testing single prediction:"
curl -X POST http://$EXTERNAL_IP/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  }'

# 4. Batch prediction
echo -e "\n4. Testing batch prediction:"
curl -X POST http://$EXTERNAL_IP/predict_batch \
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

# 5. Interactive API documentation
echo -e "\n5. API documentation available at:"
echo "   http://$EXTERNAL_IP/docs"
echo "   http://$EXTERNAL_IP/redoc"
```

---

## 🚨 Troubleshooting Common Issues

### Issue: Pods Not Running
```bash
# Check pod events
kubectl describe pods -l app=iris-api

# Check deployment events
kubectl describe deployment iris-api

# View recent cluster events
kubectl get events --sort-by=.metadata.creationTimestamp
```

### Issue: External IP Pending
```bash
# Check service status
kubectl describe service iris-api-service

# This is normal for the first 2-5 minutes
# If stuck longer, check Google Cloud Console:
# Navigation menu → Network services → Load balancing
```

### Issue: API Not Responding
```bash
# Check if pods are ready
kubectl get pods -l app=iris-api

# Check pod health probes
kubectl describe pods -l app=iris-api | grep -A 5 "Liveness\|Readiness"

# Test internal connectivity
kubectl port-forward service/iris-api-service 8080:80
# Then test: curl http://localhost:8080/health
```

### Issue: Wrong Image Version
```bash
# Check current image
kubectl describe deployment iris-api | grep Image

# Update to correct image manually if needed
kubectl set image deployment/iris-api iris-api=us-central1-docker.pkg.dev/buoyant-mason-461116-d8/iris-api/iris-api:580f2f508d3a883eeaceec51aac7a449b7794410

# Restart deployment to force pull new image
kubectl rollout restart deployment/iris-api
```

---

## 📊 Monitoring Dashboard

### Create a Simple Monitoring Script
```bash
# Save this as monitor-k8s.sh
cat > monitor-k8s.sh << 'EOF'
#!/bin/bash
echo "=== Kubernetes Iris API Status ==="
echo "Timestamp: $(date)"
echo ""

echo "🏗️  Deployment Status:"
kubectl get deployment iris-api

echo -e "\n🚀 Pod Status:"
kubectl get pods -l app=iris-api

echo -e "\n🌐 Service Status:"
kubectl get service iris-api-service

echo -e "\n📈 HPA Status:"
kubectl get hpa iris-api-hpa

echo -e "\n💾 Resource Usage:"
kubectl top pods -l app=iris-api 2>/dev/null || echo "Metrics not available"

echo -e "\n🔗 API Health Check:"
EXTERNAL_IP=$(kubectl get service iris-api-service -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null)
if [ ! -z "$EXTERNAL_IP" ]; then
    echo "External IP: $EXTERNAL_IP"
    curl -s http://$EXTERNAL_IP/health | jq . 2>/dev/null || curl -s http://$EXTERNAL_IP/health
else
    echo "External IP not ready yet"
fi

echo -e "\n" 
echo "======================="
EOF

chmod +x monitor-k8s.sh

# Run it
./monitor-k8s.sh
```

### Continuous Monitoring
```bash
# Watch pod status continuously
watch kubectl get pods -l app=iris-api

# Monitor logs continuously
kubectl logs -f deployment/iris-api

# Monitor resource usage
watch kubectl top pods -l app=iris-api
```

---

## ✅ Success Indicators

Your Kubernetes integration is working correctly when you see:

1. **✅ Deployment Status**: `2/2 READY`, `2 UP-TO-DATE`, `2 AVAILABLE`
2. **✅ Pod Status**: All pods show `Running` status with `1/1 READY`
3. **✅ Service Status**: LoadBalancer service has `EXTERNAL-IP` assigned
4. **✅ Image Version**: Deployment shows your specific image SHA
5. **✅ Health Check**: `/health` endpoint returns `{"status":"healthy"}`
6. **✅ API Functionality**: All prediction endpoints work correctly
7. **✅ Autoscaling**: HPA shows current metrics and targets
8. **✅ Logs**: Application logs show no errors

## 🎉 What's Next?

Once everything is verified and working:

1. **Set up monitoring alerts** for the `/health` endpoint
2. **Configure log aggregation** using Google Cloud Logging
3. **Set up performance monitoring** using Google Cloud Monitoring
4. **Create backup and disaster recovery** procedures
5. **Document your specific configuration** and external IP for your team

---

## 🔗 Related Documentation

- [README.md](./README.md) - Complete project overview
- [Required_Commands.md](./Required_Commands.md) - GCP setup instructions
- [GETTING_STARTED.md](./GETTING_STARTED.md) - Initial setup checklist
- [GitHub Actions CI/CD](./.github/workflows/ci.yml) - Automated deployment pipeline

---

**🎯 Pro Tip**: Bookmark your external IP and set up a simple cron job to monitor the `/health` endpoint to ensure your service stays healthy!