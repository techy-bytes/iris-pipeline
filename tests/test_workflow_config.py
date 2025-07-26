"""Tests for validating GitHub Actions workflow configuration."""

import os
import yaml
import json


def test_github_actions_key_file_exists():
    """Test that the github-actions-key.json file exists."""
    key_file_path = "github-actions-key.json"
    assert os.path.exists(key_file_path), "github-actions-key.json file should exist"


def test_github_actions_key_file_is_valid_json():
    """Test that the github-actions-key.json file contains valid JSON."""
    key_file_path = "github-actions-key.json"
    
    with open(key_file_path, 'r') as f:
        try:
            credentials = json.load(f)
        except json.JSONDecodeError:
            assert False, "github-actions-key.json should contain valid JSON"
    
    # Check that required fields are present
    required_fields = ["type", "project_id", "private_key", "client_email"]
    for field in required_fields:
        assert field in credentials, f"Required field '{field}' should be present in credentials"


def test_workflow_uses_local_credentials_file():
    """Test that the CI workflow is configured to use the local credentials file."""
    workflow_path = ".github/workflows/ci.yml"
    assert os.path.exists(workflow_path), "CI workflow file should exist"
    
    with open(workflow_path, 'r') as f:
        workflow = yaml.safe_load(f)
    
    # Check docker-build job
    docker_build_job = workflow['jobs']['docker-build']
    google_auth_steps = [step for step in docker_build_job['steps'] 
                        if step.get('name') == 'Google Auth']
    
    assert len(google_auth_steps) == 1, "Should have exactly one Google Auth step in docker-build job"
    
    auth_step = google_auth_steps[0]
    assert 'credentials_file' in auth_step['with'], "Should use credentials_file parameter"
    assert auth_step['with']['credentials_file'] == 'github-actions-key.json', \
        "Should reference the local github-actions-key.json file"
    
    # Check deploy job
    deploy_job = workflow['jobs']['deploy']
    google_auth_steps = [step for step in deploy_job['steps'] 
                        if step.get('name') == 'Google Auth']
    
    assert len(google_auth_steps) == 1, "Should have exactly one Google Auth step in deploy job"
    
    auth_step = google_auth_steps[0]
    assert 'credentials_file' in auth_step['with'], "Should use credentials_file parameter"
    assert auth_step['with']['credentials_file'] == 'github-actions-key.json', \
        "Should reference the local github-actions-key.json file"


def test_workflow_docker_auth_configuration():
    """Test that the Docker Auth step is properly configured."""
    workflow_path = ".github/workflows/ci.yml"
    
    with open(workflow_path, 'r') as f:
        workflow = yaml.safe_load(f)
    
    # Check docker-build job
    docker_build_job = workflow['jobs']['docker-build']
    docker_auth_steps = [step for step in docker_build_job['steps'] 
                        if step.get('name') == 'Docker Auth']
    
    assert len(docker_auth_steps) == 1, "Should have exactly one Docker Auth step"
    
    docker_auth_step = docker_auth_steps[0]
    assert docker_auth_step['uses'] == 'docker/login-action@v3', "Should use docker/login-action@v3"
    
    with_params = docker_auth_step['with']
    assert with_params['username'] == 'oauth2accesstoken', "Username should be oauth2accesstoken"
    assert with_params['password'] == '${{ steps.auth.outputs.access_token }}', \
        "Password should reference the auth step output"
    assert with_params['registry'] == '${{ env.GAR_LOCATION }}-docker.pkg.dev', \
        "Registry should use GAR_LOCATION environment variable"