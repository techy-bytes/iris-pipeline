"""
Configuration management for customer segmentation feature engineering.

This module provides configuration classes and utilities to manage
settings for the feature engineering pipeline, replacing hardcoded
values in the original clumsy implementation.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import json
import yaml
import logging

logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Database connection configuration."""
    server: str = ""
    database: str = ""
    username: str = ""
    password: str = ""
    authentication: str = "ActiveDirectoryPassword"
    driver: str = "{ODBC Driver 18 for SQL Server}"
    
    @classmethod
    def from_env(cls) -> 'DatabaseConfig':
        """Create configuration from environment variables."""
        return cls(
            server=os.getenv('DB_SERVER', ''),
            database=os.getenv('DB_DATABASE', ''),
            username=os.getenv('DB_USERNAME', ''),
            password=os.getenv('DB_PASSWORD', ''),
            authentication=os.getenv('DB_AUTHENTICATION', 'ActiveDirectoryPassword'),
            driver=os.getenv('DB_DRIVER', '{ODBC Driver 18 for SQL Server}')
        )
    
    def to_connection_string(self) -> str:
        """Convert to pyodbc connection string."""
        return (
            f"DRIVER={self.driver};"
            f"SERVER={self.server};"
            f"DATABASE={self.database};"
            f"UID={self.username};"
            f"PWD={self.password};"
            f"Authentication={self.authentication};"
            f"Encrypt=yes;"
            f"TrustServerCertificate=no;"
            f"Connection Timeout=30;"
        )


@dataclass 
class QueryConfig:
    """Configuration for SQL queries and customer vector generation."""
    use_simplified_queries: bool = True
    max_customers_per_batch: int = 1000
    query_timeout: int = 300  # 5 minutes
    enable_csv_output: bool = True
    csv_output_path: str = "customer_vectors_debug.csv"


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    enable_file_logging: bool = False
    log_file: str = "customer_vector.log"


class ConfigManager:
    """Configuration manager for the customer vector system."""
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize configuration manager."""
        self.db_config = DatabaseConfig()
        self.query_config = QueryConfig()
        self.logging_config = LoggingConfig()
        
        if config_file and os.path.exists(config_file):
            self.load_from_file(config_file)
    
    def load_from_file(self, config_file: str):
        """Load configuration from YAML file."""
        try:
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Load database config
            if 'database' in config_data:
                db_data = config_data['database']
                self.db_config = DatabaseConfig(**db_data)
            
            # Load query config  
            if 'queries' in config_data:
                query_data = config_data['queries']
                self.query_config = QueryConfig(**query_data)
            
            # Load logging config
            if 'logging' in config_data:
                logging_data = config_data['logging']
                self.logging_config = LoggingConfig(**logging_data)
            
            logger.info(f"Configuration loaded from {config_file}")
            
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_file}: {e}")
            raise
    
    def get_db_connection_string(self) -> str:
        """Get database connection string."""
        return self.db_config.to_connection_string()
    
    def validate(self):
        """Validate configuration settings."""
        if not self.db_config.server:
            raise ValueError("Database server is required")
        if not self.db_config.database:
            raise ValueError("Database name is required")
        if not self.db_config.username:
            raise ValueError("Database username is required")
        if not self.db_config.password:
            raise ValueError("Database password is required")
        
        logger.info("Configuration validation passed")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'database': asdict(self.db_config),
            'queries': asdict(self.query_config),
            'logging': asdict(self.logging_config)
        }
    
    def save_to_file(self, config_file: str):
        """Save current configuration to YAML file."""
        try:
            config_dict = self.to_dict()
            # Remove password for security
            if 'database' in config_dict and 'password' in config_dict['database']:
                config_dict['database']['password'] = "***REDACTED***"
            
            with open(config_file, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
            
            logger.info(f"Configuration saved to {config_file}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration to {config_file}: {e}")
            raise