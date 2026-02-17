# Scalable User Churn Prediction (Big Data ML)

## Project Overview
This project implements a scalable user churn prediction system using Apache Spark for big data processing and machine learning.

## Week 1: Local Spark Cluster Setup

### Objectives
- Set up local Spark cluster (Master + 2 Workers) using Docker
- Implement PySpark schema validation
- Validate row counts and data quality metrics

### Architecture
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Spark Master  │────│  Spark Worker 1 │────│  Spark Worker 2 │
│   (Port 8080)   │    │   (2G Memory)   │    │   (2G Memory)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Setup Instructions

### Prerequisites
- Docker installed and running
- Python 3.8+
- Git

### Quick Start

1. **Start the Spark Cluster**
   ```bash
   cd /Users/tarandeepsingh/Desktop/internship\ project\ 4/SparkScale-churn
   docker-compose up -d
   ```

2. **Verify Cluster Status**
   - Master Web UI: http://localhost:8080
   - Check cluster status: `docker-compose ps`

3. **Install Python Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run Validation Tests**
   ```bash
   python test_schema_validation.py
   ```

## Project Structure

```
SparkScale-churn/
├── docker-compose.yml          # Spark cluster configuration
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── src/
│   ├── schema_validation.py    # Schema validation logic
│   └── data_processor.py       # Data processing utilities
├── data/                       # Data directory (mounted in containers)
├── models/                     # Model storage
├── docs/                       # Documentation
└── test_schema_validation.py   # Test suite
```

## Week 1 Features

### Schema Validation
- Validates data structure against expected schema
- Checks for missing/extra fields
- Type validation
- Null value analysis

### Data Quality Metrics
- Row count validation
- Duplicate detection
- Negative value identification
- Data quality scoring (0-100)

### Key Components

#### ChurnDataValidator
```python
validator = ChurnDataValidator(spark)
report = validator.generate_validation_report(df, "churn_data")
```

#### ChurnDataProcessor
```python
processor = ChurnDataProcessor(spark)
stats = processor.get_basic_statistics(df)
report = processor.validate_and_report(df, "dataset_name")
```

## Expected Schema

| Field | Type | Nullable | Description |
|-------|------|----------|-------------|
| customer_id | String | No | Unique customer identifier |
| age | Integer | Yes | Customer age |
| gender | String | Yes | Customer gender |
| tenure_months | Integer | Yes | Months with company |
| monthly_charges | Double | Yes | Monthly bill amount |
| total_charges | Double | Yes | Total charges to date |
| contract_type | String | Yes | Contract type |
| payment_method | String | Yes | Payment method |
| internet_service | String | Yes | Internet service type |
| online_security | String | Yes | Online security service |
| tech_support | String | Yes | Technical support |
| churn | Boolean | No | Churn status |
| signup_date | Date | Yes | Signup date |
| last_interaction_date | Date | Yes | Last interaction |

## Monitoring

### Spark Master UI
- URL: http://localhost:8080
- Monitor cluster health
- Track job execution
- Resource utilization

### Logging
- Application logs: INFO level
- Error tracking and validation reports

## Next Steps (Week 2)
- Ingest real customer data
- Implement advanced data profiling
- Set up automated data quality checks
- Performance optimization

## Troubleshooting

### Docker Issues
```bash
# Stop all containers
docker-compose down

# Rebuild and start
docker-compose up --build -d

# Check logs
docker-compose logs spark-master
```

### Spark Connection Issues
- Ensure all containers are running
- Check network connectivity
- Verify port availability (8080, 7077)

## Performance Considerations
- Memory allocation: 2GB per worker
- CPU cores: 2 per worker
- Adaptive query execution enabled
- Partition coalescing configured
