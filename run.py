"""
Main execution script for customer vector generation.
"""

import pyodbc
from config import ConfigManager
from customer_vector_n8n import n8n_generate_customer_vectors
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Main execution function."""
    try:
        # Load configuration from YAML file
        config_manager = ConfigManager(config_file='config.yaml')
        config_manager.validate()

        # Test database connection
        logger.info("Testing database connection...")
        connection_string = config_manager.get_db_connection_string()
        
        # Establish database connection
        conn = pyodbc.connect(connection_string)
        logger.info("Database connection successful!")
        
        # Close the test connection
        conn.close()
        
        # Generate vectors for top 3 customers (as per user's example)
        logger.info("Generating customer vectors...")
        vectors = n8n_generate_customer_vectors(top_n=3)
        
        if vectors:
            logger.info(f"Generated {len(vectors)} customers with {len(vectors[0])-1} features each")
            
            # Print first few vectors for verification
            logger.info("Sample vectors:")
            for i, vector in enumerate(vectors[:3]):
                logger.info(f"Customer {i+1}: {vector}")
        else:
            logger.warning("No vectors generated")
    
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise


if __name__ == "__main__":
    main()