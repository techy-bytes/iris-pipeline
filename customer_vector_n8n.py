"""
Customer Vector Generation for N8N Integration with Simplified Queries Approach.

This module provides optimized customer vector generation using individual simplified queries
to avoid performance issues with complex joins and large result sets.
"""

import logging
import time
from typing import List, Dict, Any, Optional, Union
import pandas as pd
import pyodbc
from config import ConfigManager

# Setup logging
logger = logging.getLogger(__name__)


class CustomerVectorGenerator:
    """Optimized customer vector generator using simplified individual queries."""
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize with configuration manager."""
        self.config = config_manager
        self.conn = None
        self.feature_names = []
        
    def connect(self):
        """Establish database connection."""
        try:
            connection_string = self.config.get_db_connection_string()
            self.conn = pyodbc.connect(connection_string)
            logger.info("Database connection established")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    def disconnect(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
            logger.info("Database connection closed")
    
    def get_customer_base(self, top_n: Optional[int] = None, customer_ids: Optional[List[int]] = None) -> pd.DataFrame:
        """Get customer base with proper filtering to avoid loading all customers."""
        if customer_ids:
            # Specific customers requested
            customer_list = ','.join(map(str, customer_ids))
            query = f"""
            SELECT userId 
            FROM [dbo].[customers] 
            WHERE userId IN ({customer_list})
            ORDER BY userId
            """
            logger.info(f"Loading {len(customer_ids)} specific customers")
        elif top_n:
            # Top N customers requested
            query = f"""
            SELECT TOP {top_n} userId 
            FROM [dbo].[customers] 
            ORDER BY userId
            """
            logger.info(f"Loading top {top_n} customers")
        else:
            raise ValueError("Must specify either top_n or customer_ids")
        
        logger.info("Loading customer base...")
        customer_df = pd.read_sql_query(query, self.conn)
        logger.info(f"Found {len(customer_df)} customers")
        return customer_df
    
    def get_feature_for_customers(self, customer_ids: List[int], feature_name: str, query_template: str) -> Dict[int, float]:
        """Get a specific feature value for a list of customers using individual queries."""
        feature_values = {}
        
        logger.info(f"Executing {feature_name} query...")
        
        # Process customers in batches to avoid query length limits
        batch_size = self.config.query_config.max_customers_per_batch
        
        for i in range(0, len(customer_ids), batch_size):
            batch_ids = customer_ids[i:i + batch_size]
            customer_list = ','.join(map(str, batch_ids))
            
            # Replace placeholder in query template
            query = query_template.replace('{customer_ids}', customer_list)
            
            try:
                result_df = pd.read_sql_query(query, self.conn)
                
                # Convert to dictionary for easy lookup
                for _, row in result_df.iterrows():
                    feature_values[row['userId']] = row.get(feature_name, 0.0)
                
                logger.info(f"Processed batch {i//batch_size + 1}/{(len(customer_ids) + batch_size - 1)//batch_size} for {feature_name}")
                
            except Exception as e:
                logger.error(f"Error processing batch for {feature_name}: {e}")
                # Fill with zeros for failed batch
                for customer_id in batch_ids:
                    feature_values[customer_id] = 0.0
        
        return feature_values
    
    def generate_vectors(self, top_n: Optional[int] = None, customer_ids: Optional[List[int]] = None) -> List[List[Union[int, float]]]:
        """Generate customer vectors using simplified individual queries approach."""
        logger.info("Using simplified individual queries approach...")
        
        if not self.conn:
            self.connect()
        
        try:
            # Get customer base (only the customers we need)
            customers_df = self.get_customer_base(top_n=top_n, customer_ids=customer_ids)
            customer_list = customers_df['userId'].tolist()
            
            if not customer_list:
                logger.warning("No customers found")
                return []
            
            # Define simplified feature queries
            feature_queries = {
                'order_counts': """
                    SELECT userId, COUNT(*) as order_counts
                    FROM [dbo].[orders]
                    WHERE userId IN ({customer_ids})
                    GROUP BY userId
                """,
                'pos_order_counts': """
                    SELECT userId, COUNT(*) as pos_order_counts
                    FROM [dbo].[orders]
                    WHERE userId IN ({customer_ids}) AND channel = 'POS'
                    GROUP BY userId
                """,
                'online_order_counts': """
                    SELECT userId, COUNT(*) as online_order_counts
                    FROM [dbo].[orders]
                    WHERE userId IN ({customer_ids}) AND channel = 'Online'
                    GROUP BY userId
                """,
                'total_spent': """
                    SELECT userId, SUM(totalAmount) as total_spent
                    FROM [dbo].[orders]
                    WHERE userId IN ({customer_ids})
                    GROUP BY userId
                """,
                'avg_order_value': """
                    SELECT userId, AVG(totalAmount) as avg_order_value
                    FROM [dbo].[orders]
                    WHERE userId IN ({customer_ids})
                    GROUP BY userId
                """
            }
            
            # Initialize feature names for CSV output
            self.feature_names = ['userId'] + list(feature_queries.keys())
            
            # Collect features for all customers
            customer_features = {}
            
            # Initialize all customers with userId
            for customer_id in customer_list:
                customer_features[customer_id] = {'userId': customer_id}
            
            # Get each feature using individual queries
            for feature_name, query_template in feature_queries.items():
                feature_values = self.get_feature_for_customers(customer_list, feature_name, query_template)
                
                # Add feature values to customer records
                for customer_id in customer_list:
                    customer_features[customer_id][feature_name] = feature_values.get(customer_id, 0.0)
            
            # Convert to list of lists format
            vectors = []
            for customer_id in customer_list:
                feature_row = customer_features[customer_id]
                vector = [feature_row[feature_name] for feature_name in self.feature_names]
                vectors.append(vector)
            
            logger.info(f"Generated vectors for {len(vectors)} customers with {len(self.feature_names)} features each")
            
            # Save to CSV for debugging if enabled
            if self.config.query_config.enable_csv_output:
                self.save_vectors_to_csv(vectors)
            
            return vectors
            
        except Exception as e:
            logger.error(f"Error generating customer vectors: {e}")
            raise
        finally:
            self.disconnect()
    
    def save_vectors_to_csv(self, vectors: List[List[Union[int, float]]]):
        """Save vectors to CSV file for debugging."""
        try:
            # Create DataFrame with feature names as columns
            df = pd.DataFrame(vectors, columns=self.feature_names)
            
            csv_path = self.config.query_config.csv_output_path
            df.to_csv(csv_path, index=False)
            
            logger.info(f"Customer vectors saved to {csv_path} for debugging")
            logger.info(f"CSV contains {len(df)} rows and {len(df.columns)} columns")
            logger.info(f"Feature names: {', '.join(self.feature_names)}")
            
        except Exception as e:
            logger.error(f"Failed to save vectors to CSV: {e}")


def n8n_generate_customer_vectors(top_n: Optional[int] = None, customer_ids: Optional[List[int]] = None, config_file: str = "config.yaml") -> List[List[Union[int, float]]]:
    """
    Main function to generate customer vectors for n8n integration.
    
    Args:
        top_n: Number of top customers to process (mutually exclusive with customer_ids)
        customer_ids: Specific customer IDs to process (mutually exclusive with top_n)
        config_file: Path to configuration file
    
    Returns:
        List of customer vectors, each vector is a list starting with userId followed by feature values
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load configuration
    config_manager = ConfigManager(config_file)
    config_manager.validate()
    
    # Generate vectors
    generator = CustomerVectorGenerator(config_manager)
    return generator.generate_vectors(top_n=top_n, customer_ids=customer_ids)


if __name__ == "__main__":
    # Example usage
    vectors = n8n_generate_customer_vectors(top_n=3)
    print(f"Generated {len(vectors)} customers with {len(vectors[0])-1} features each")