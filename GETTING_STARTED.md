# 🚀 Getting Started Checklist

Use this checklist to get up and running quickly, whether for local development or production deployment.

## 📝 Choose Your Path

### 🔧 For Local Development (Recommended First)

- [ ] **Clone the repository**
  ```bash
  git clone <your-repo-url>
  cd iris-pipeline
  ```

- [ ] **Install dependencies**
  ```bash
  pip install -r requirements.txt
  ```

- [ ] **Run tests to verify setup**
  ```bash
  pytest
  ```
  Expected: All tests should pass ✅

- [ ] **Train the model**
  ```bash
  python src/train.py
  ```
  Expected: Creates `model.joblib` and `label_encoder.joblib` files

- [ ] **Start the API**
  ```bash
  uvicorn src.api:app --host 0.0.0.0 --port 8000
  ```
  Expected: API starts on http://localhost:8000

- [ ] **Test the API**
  ```bash
  # In another terminal
  curl http://localhost:8000/health
  curl -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
  ```
  Expected: Health check returns "healthy" status, predict returns species

**✅ You're ready for local development!**

---

### 🌩️ For Production Deployment (Google Cloud)

#### Prerequisites Checklist
- [ ] Google Cloud account with billing enabled
- [ ] Admin access to a Google Cloud project
- [ ] `gcloud` CLI installed
- [ ] `kubectl` installed

#### Setup Steps
- [ ] **Complete GCP setup**
  - Follow **all steps** in [`Required_Commands.md`](./Required_Commands.md)
  - This takes 20-30 minutes first time

- [ ] **Configure GitHub secrets**
  - Add `GCP_PROJECT_ID` in GitHub repo settings
  - Add `GCP_SA_KEY` in GitHub repo settings

- [ ] **Deploy to production**
  ```bash
  git push origin main
  ```
  Expected: GitHub Actions runs and deploys to GKE

- [ ] **Verify deployment**
  ```bash
  kubectl get pods -l app=iris-api
  kubectl get service iris-api-service
  ```
  Expected: Pods running, external IP assigned

- [ ] **Test deployed API**
  ```bash
  # Get external IP
  EXTERNAL_IP=$(kubectl get service iris-api-service -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
  curl http://$EXTERNAL_IP/health
  ```
  Expected: API responds from production

- [ ] **Verify K8s integration** 
  - Follow [`K8S_VERIFICATION.md`](./K8S_VERIFICATION.md) for comprehensive verification
  - Use the verification checklist to ensure everything works correctly

**✅ You're live in production!**

---

## 🆘 If Something Goes Wrong

### Local Development Issues
1. **Tests failing?** → Check the [Troubleshooting section in README.md](./README.md#troubleshooting)
2. **API won't start?** → Make sure you ran `python src/train.py` first
3. **Dependencies broken?** → Try `pip install -r requirements.txt --force-reinstall`

### Production Deployment Issues
1. **GitHub Actions failing?** → Check [Required_Commands.md troubleshooting](./Required_Commands.md#troubleshooting)
2. **Pods not starting?** → Run `kubectl describe pods -l app=iris-api`
3. **Can't access API?** → Wait 2-5 minutes for external IP, then check load balancer in Google Cloud Console

## 🎯 What's Next?

### Once Everything Works:
- [ ] **Explore the code**: Check out `src/train.py` and `src/api.py`
- [ ] **Make changes**: Try modifying the model or API
- [ ] **Test your changes**: Always run `pytest` before committing
- [ ] **Deploy updates**: Push to main branch for automatic deployment

### For Team Development:
- [ ] **Create feature branches**: Don't work directly on main
- [ ] **Submit pull requests**: GitHub Actions will test changes automatically
- [ ] **Review CML reports**: Check model performance on every PR

### For Monitoring:
- [ ] **Set up alerts**: Monitor the `/health` endpoint
- [ ] **Check logs**: Use `kubectl logs -f deployment/iris-api`
- [ ] **Scale as needed**: Use HPA or manual scaling

---

**Need help?** Check the full documentation in [README.md](./README.md) or [Required_Commands.md](./Required_Commands.md).