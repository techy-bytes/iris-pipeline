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


