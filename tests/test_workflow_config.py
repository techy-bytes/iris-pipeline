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


def test_workflow_uses_credentials_json():
    """Test that the CI workflow is configured to use credentials_json with GitHub secrets."""
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
    assert 'credentials_json' in auth_step['with'], "Should use credentials_json parameter"
    assert auth_step['with']['credentials_json'] == '${{ secrets.GCP_SA_KEY }}', \
        "Should reference the GCP_SA_KEY secret"
    
    # Check for required configuration parameters
    with_params = auth_step['with']
    assert with_params.get('create_credentials_file') == True, "Should set create_credentials_file to true"
    assert with_params.get('export_environment_variables') == True, "Should set export_environment_variables to true"
    assert with_params.get('universe') == 'googleapis.com', "Should set universe to googleapis.com"
    assert with_params.get('cleanup_credentials') == True, "Should set cleanup_credentials to true"
    assert with_params.get('access_token_lifetime') == '3600s', "Should set access_token_lifetime to 3600s"
    assert with_params.get('access_token_scopes') == 'https://www.googleapis.com/auth/cloud-platform', \
        "Should set access_token_scopes to cloud-platform"
    assert with_params.get('id_token_include_email') == False, "Should set id_token_include_email to false"
    
    # Check deploy job
    deploy_job = workflow['jobs']['deploy']
    google_auth_steps = [step for step in deploy_job['steps'] 
                        if step.get('name') == 'Google Auth']
    
    assert len(google_auth_steps) == 1, "Should have exactly one Google Auth step in deploy job"
    
    auth_step = google_auth_steps[0]
    assert 'credentials_json' in auth_step['with'], "Should use credentials_json parameter"
    assert auth_step['with']['credentials_json'] == '${{ secrets.GCP_SA_KEY }}', \
        "Should reference the GCP_SA_KEY secret"


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