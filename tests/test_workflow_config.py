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
