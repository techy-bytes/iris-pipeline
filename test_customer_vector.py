"""
Test script to verify the customer vector implementation works correctly.
This version can be run without Azure Synapse connection for testing.
"""

from customer_vector_n8n import CustomerVectorGenerator
from config import ConfigManager
import pandas as pd
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


class MockConnection:
    """Mock database connection for testing."""
    
    def close(self):
        pass


def mock_read_sql_query(query: str, conn) -> pd.DataFrame:
    """Mock pandas read_sql_query for testing."""
    if "SELECT TOP" in query and "userId" in query:
        # Mock customer base query
        return pd.DataFrame({'userId': [1001, 1002, 1003]})
    elif "order_counts" in query:
        # Mock order counts
        return pd.DataFrame({
            'userId': [1001, 1002, 1003],
            'order_counts': [15, 23, 8]
        })
    elif "pos_order_counts" in query:
        # Mock POS order counts
        return pd.DataFrame({
            'userId': [1001, 1002],
            'pos_order_counts': [10, 15]
        })
    elif "online_order_counts" in query:
        # Mock online order counts
        return pd.DataFrame({
            'userId': [1002, 1003],
            'online_order_counts': [8, 8]
        })
    elif "total_spent" in query:
        # Mock total spent
        return pd.DataFrame({
            'userId': [1001, 1002, 1003],
            'total_spent': [1250.50, 2100.75, 450.25]
        })
    elif "avg_order_value" in query:
        # Mock average order value
        return pd.DataFrame({
            'userId': [1001, 1002, 1003],
            'avg_order_value': [83.37, 91.34, 56.28]
        })
    else:
        return pd.DataFrame()


def test_customer_vector_generation():
    """Test the customer vector generation with mock data."""
    logger.info("Testing customer vector generation with mock data...")
    
    # Create config manager
    config_manager = ConfigManager()
    config_manager.query_config.enable_csv_output = True
    config_manager.query_config.csv_output_path = "test_vectors.csv"
    
    # Create generator
    generator = CustomerVectorGenerator(config_manager)
    
    # Mock the connection and pandas function
    generator.conn = MockConnection()
    
    # Backup original pandas function
    original_read_sql = pd.read_sql_query
    pd.read_sql_query = mock_read_sql_query
    
    try:
        # Test vector generation
        vectors = generator.generate_vectors(top_n=3)
        
        logger.info(f"Generated {len(vectors)} customer vectors")
        logger.info(f"Feature names: {generator.feature_names}")
        
        # Print vectors
        for i, vector in enumerate(vectors):
            logger.info(f"Customer {i+1} vector: {vector}")
        
        # Verify structure
        assert len(vectors) == 3, f"Expected 3 vectors, got {len(vectors)}"
        assert len(vectors[0]) == 6, f"Expected 6 features, got {len(vectors[0])}"  # userId + 5 features
        
        logger.info("✅ Test passed! Customer vector generation working correctly.")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False
    finally:
        # Restore original function
        pd.read_sql_query = original_read_sql


def test_config_management():
    """Test configuration management."""
    logger.info("Testing configuration management...")
    
    try:
        # Test config creation
        config_manager = ConfigManager()
        
        # Test database config
        assert config_manager.db_config.authentication == "ActiveDirectoryPassword"
        assert config_manager.db_config.driver == "{ODBC Driver 18 for SQL Server}"
        
        # Test query config
        assert config_manager.query_config.use_simplified_queries == True
        assert config_manager.query_config.enable_csv_output == True
        
        # Test connection string generation
        config_manager.db_config.server = "test-server"
        config_manager.db_config.database = "test-db"
        config_manager.db_config.username = "test-user"
        config_manager.db_config.password = "test-pass"
        
        conn_str = config_manager.get_db_connection_string()
        assert "test-server" in conn_str
        assert "test-db" in conn_str
        assert "ActiveDirectoryPassword" in conn_str
        
        logger.info("✅ Configuration management test passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Configuration test failed: {e}")
        return False


if __name__ == "__main__":
    logger.info("Running customer vector system tests...")
    
    config_test = test_config_management()
    vector_test = test_customer_vector_generation()
    
    if config_test and vector_test:
        logger.info("🎉 All tests passed! The implementation is working correctly.")
        logger.info("You can now run 'python run.py' with your Azure Synapse connection.")
    else:
        logger.error("❌ Some tests failed. Please check the implementation.")