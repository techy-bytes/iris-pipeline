"""Tests for validating GitHub Actions workflow configuration."""

import os
import yaml
import json


def test_github_actions_credentials_not_in_repo():
    """Test that the github-actions-key.json file is not in the repository."""
    key_file_path = "github-actions-key.json"
    assert not os.path.exists(key_file_path), "github-actions-key.json file should not be in repository for security"


def test_gitignore_excludes_credentials():
    """Test that .gitignore excludes credential files."""
    gitignore_path = ".gitignore"
    assert os.path.exists(gitignore_path), ".gitignore file should exist"
    
    with open(gitignore_path, 'r') as f:
        gitignore_content = f.read()
    
    assert "github-actions-key.json" in gitignore_content, ".gitignore should exclude github-actions-key.json"
    assert "*.json.key" in gitignore_content, ".gitignore should exclude *.json.key files"


def test_hpa_yaml_exists():
    """Test that the hpa.yaml file exists and contains valid YAML."""
    hpa_file_path = "k8s/hpa.yaml"
    assert os.path.exists(hpa_file_path), "hpa.yaml file should exist in k8s directory"
    
    with open(hpa_file_path, 'r') as f:
        hpa_content = yaml.safe_load(f)
    
    assert hpa_content is not None, "hpa.yaml should contain valid YAML content"
    assert hpa_content.get('kind') == 'HorizontalPodAutoscaler', "hpa.yaml should contain HPA configuration"
    assert hpa_content.get('metadata', {}).get('name') == 'iris-api-hpa', "HPA should be named iris-api-hpa"


def test_deployment_yaml_no_hpa():
    """Test that deployment.yaml no longer contains HPA configuration."""
    deployment_file_path = "k8s/deployment.yaml"
    assert os.path.exists(deployment_file_path), "deployment.yaml file should exist"
    
    with open(deployment_file_path, 'r') as f:
        deployment_content = f.read()
    
    assert "HorizontalPodAutoscaler" not in deployment_content, "deployment.yaml should not contain HPA configuration"


def test_ci_workflow_applies_hpa():
    """Test that CI workflow includes applying the hpa.yaml file."""
    ci_file_path = ".github/workflows/ci.yml"
    assert os.path.exists(ci_file_path), "ci.yml file should exist"
    
    with open(ci_file_path, 'r') as f:
        ci_content = f.read()
    
    assert "kubectl apply -f k8s/hpa.yaml" in ci_content, "CI workflow should apply hpa.yaml file"


def test_serviceaccount_yaml_exists():
    """Test that the serviceaccount.yaml file exists and contains valid configuration."""
    sa_file_path = "k8s/serviceaccount.yaml"
    assert os.path.exists(sa_file_path), "serviceaccount.yaml file should exist in k8s directory"
    
    with open(sa_file_path, 'r') as f:
        sa_content = yaml.safe_load(f)
    
    assert sa_content is not None, "serviceaccount.yaml should contain valid YAML content"
    assert sa_content.get('kind') == 'ServiceAccount', "serviceaccount.yaml should contain ServiceAccount configuration"
    assert sa_content.get('metadata', {}).get('name') == 'telemetry-access', "ServiceAccount should be named telemetry-access"
    assert sa_content.get('metadata', {}).get('namespace') == 'default', "ServiceAccount should be in default namespace"
    
    # Check that it has the workload identity annotation
    annotations = sa_content.get('metadata', {}).get('annotations', {})
    assert 'iam.gke.io/gcp-service-account' in annotations, "ServiceAccount should have workload identity annotation"
    assert '${PROJECT_ID}' in annotations['iam.gke.io/gcp-service-account'], "ServiceAccount annotation should include PROJECT_ID variable"


def test_deployment_uses_serviceaccount():
    """Test that deployment.yaml specifies the telemetry-access service account."""
    deployment_file_path = "k8s/deployment.yaml"
    assert os.path.exists(deployment_file_path), "deployment.yaml file should exist"
    
    with open(deployment_file_path, 'r') as f:
        deployment_content = f.read()
    
    assert "serviceAccountName: telemetry-access" in deployment_content, "Deployment should use telemetry-access service account"


def test_ci_workflow_applies_serviceaccount():
    """Test that CI workflow includes creating the service account before deployment."""
    ci_file_path = ".github/workflows/ci.yml"
    assert os.path.exists(ci_file_path), "ci.yml file should exist"
    
    with open(ci_file_path, 'r') as f:
        ci_content = f.read()
    
    assert "envsubst < k8s/serviceaccount.yaml | kubectl apply -f -" in ci_content, "CI workflow should apply serviceaccount.yaml file"
    
    # Check that service account is applied before deployment
    sa_line = None
    deploy_line = None
    lines = ci_content.split('\n')
    for i, line in enumerate(lines):
        if "envsubst < k8s/serviceaccount.yaml" in line:
            sa_line = i
        elif "kubectl apply -f k8s/deployment.yaml" in line:
            deploy_line = i
    
    assert sa_line is not None, "CI workflow should include service account creation"
    assert deploy_line is not None, "CI workflow should include deployment"
    assert sa_line < deploy_line, "Service account should be created before deployment"
