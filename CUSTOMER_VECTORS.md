# Customer Vector Generation System

This repository now includes an optimized customer vector generation system designed for Azure Synapse integration with n8n workflows.

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- ODBC Driver 18 for SQL Server
- Access to Azure Synapse Analytics

### Installation
```bash
pip install pandas pyyaml pyodbc
```

### Configuration
1. Update `config.yaml` with your Azure Synapse credentials:
```yaml
database:
  authentication: 'ActiveDirectoryPassword'
  database: 'YOUR_DATABASE_NAME'
  driver: '{ODBC Driver 18 for SQL Server}'
  password: 'YOUR_PASSWORD'
  server: 'YOUR_SYNAPSE_SERVER.sql.azuresynapse.net'
  username: 'YOUR_USERNAME'
```

### Basic Usage
```python
from customer_vector_n8n import n8n_generate_customer_vectors

# Generate vectors for top 3 customers
vectors = n8n_generate_customer_vectors(top_n=3)

# Generate vectors for specific customers
vectors = n8n_generate_customer_vectors(customer_ids=[1234, 5678, 9012])

print(f"Generated {len(vectors)} customers with {len(vectors[0])-1} features each")
```

## 📈 Performance Optimizations

### Problem Solved
The original implementation had a critical performance issue:
- ❌ **Before**: Loaded all 739,908 customers into memory before filtering
- ✅ **After**: Uses `TOP N` and `WHERE` clauses to limit data at the SQL level

### Key Improvements
1. **Efficient Customer Filtering**: Only loads requested customers from database
2. **Simplified Queries**: Uses individual optimized queries instead of complex joins
3. **Batch Processing**: Processes customers in configurable batches (default: 1000)
4. **CSV Debug Output**: Saves vectors to CSV for easy analysis
5. **Proper Error Handling**: Graceful handling of query failures

## 🔧 Implementation Details

### Simplified Queries Approach
The system uses individual SQL queries for each feature:

```sql
-- Order counts
SELECT userId, COUNT(*) as order_counts
FROM [dbo].[orders]
WHERE userId IN (1001, 1002, 1003)  -- Only requested customers
GROUP BY userId

-- POS order counts  
SELECT userId, COUNT(*) as pos_order_counts
FROM [dbo].[orders]
WHERE userId IN (1001, 1002, 1003) AND channel = 'POS'
GROUP BY userId
```

### Feature Vector Structure
Each customer vector contains:
1. `userId` - Customer identifier
2. `order_counts` - Total number of orders
3. `pos_order_counts` - Number of POS orders
4. `online_order_counts` - Number of online orders
5. `total_spent` - Total amount spent
6. `avg_order_value` - Average order value

## 📊 CSV Debug Output

When `enable_csv_output: true` in config, vectors are saved to CSV:

```csv
userId,order_counts,pos_order_counts,online_order_counts,total_spent,avg_order_value
1001,15,10,5,1250.50,83.37
1002,23,15,8,2100.75,91.34
1003,8,0,8,450.25,56.28
```

## 🔒 Security Features

- **ActiveDirectoryPassword Authentication**: Uses Azure AD for secure connection
- **Encrypted Connections**: TLS encryption enabled by default
- **Configuration Security**: Passwords can be redacted in saved configs
- **Connection Timeout**: Prevents hanging connections (30 seconds default)

## 🧪 Testing

Run the test suite to verify functionality:
```bash
python test_customer_vector.py
```

This tests the system with mock data without requiring Azure Synapse connection.

## 📋 Configuration Options

### Database Settings (`config.yaml`)
```yaml
database:
  server: "synapse-server.sql.azuresynapse.net"
  database: "database_name"
  username: "user@domain.com" 
  password: "password"
  authentication: "ActiveDirectoryPassword"
  driver: "{ODBC Driver 18 for SQL Server}"
```

### Query Settings
```yaml
queries:
  use_simplified_queries: true          # Use optimized approach
  max_customers_per_batch: 1000         # Batch size for processing
  query_timeout: 300                    # Query timeout in seconds
  enable_csv_output: true               # Save vectors to CSV
  csv_output_path: "customer_vectors_debug.csv"
```

### Logging Settings  
```yaml
logging:
  level: "INFO"                         # DEBUG, INFO, WARNING, ERROR
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  enable_file_logging: false            # Log to file
  log_file: "customer_vector.log"
```

## 🚀 Usage Examples

### Example 1: Top N Customers
```python
# Get vectors for top 10 customers by userId
vectors = n8n_generate_customer_vectors(top_n=10)
```

### Example 2: Specific Customers
```python
# Get vectors for specific customer IDs
customer_ids = [12345, 67890, 11111]
vectors = n8n_generate_customer_vectors(customer_ids=customer_ids)
```

### Example 3: Custom Configuration
```python
from customer_vector_n8n import CustomerVectorGenerator
from config import ConfigManager

# Load custom config
config = ConfigManager('custom_config.yaml')
generator = CustomerVectorGenerator(config)

# Generate vectors
vectors = generator.generate_vectors(top_n=5)
```

## 🔍 Troubleshooting

### Connection Issues
```python
# Test database connection
from config import ConfigManager
import pyodbc

config = ConfigManager('config.yaml')
conn = pyodbc.connect(config.get_db_connection_string())
print("Connection successful!")
conn.close()
```

### Performance Issues
- Reduce `max_customers_per_batch` if memory is limited
- Increase `query_timeout` for complex queries
- Use specific `customer_ids` instead of `top_n` for better performance

### Debugging
- Enable CSV output to inspect generated vectors
- Set logging level to `DEBUG` for detailed query information
- Use the test script to verify functionality with mock data

## 📈 Performance Comparison

| Metric | Original | Optimized |
|--------|----------|-----------|
| Customer Loading | All 739,908 | Only requested (e.g., 3) |
| Query Approach | Complex joins | Simple individual queries |
| Memory Usage | High | Low |
| Execution Time | 2+ hours | < 1 minute |
| Error Handling | Poor | Robust |

## 🤝 Contributing

1. Test your changes with `python test_customer_vector.py`
2. Ensure CSV output works correctly
3. Update configuration documentation if needed
4. Follow the simplified queries approach for new features

## 📄 License

This project follows the same license as the main iris-pipeline repository.