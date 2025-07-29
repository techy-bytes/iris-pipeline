"""
Main script that matches the user's example usage.
This is the optimized version that should run much faster.
"""

from customer_vector_n8n import n8n_generate_customer_vectors
import logging

# Setup logging to match the user's output format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Main function that matches user's example."""
    try:
        # Generate vectors for top 3 customers (uses config.yaml settings)
        print("Starting customer vector generation for top 3 customers...")
        vectors = n8n_generate_customer_vectors(top_n=3)

        # Generate vectors for specific customers (alternative approach)
        # vectors = n8n_generate_customer_vectors(customer_ids=[1234, 5678, 9012])

        print(f"Generated {len(vectors)} customers with {len(vectors[0])-1} features each")
        
        # Show first few vectors for verification
        print("\nSample vectors:")
        for i, vector in enumerate(vectors[:3]):
            print(f"Customer {i+1}: {vector}")
            
    except Exception as e:
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    main()