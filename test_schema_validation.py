"""
Test script for Schema Validation Module
Week 1: Testing schema validation and row counts
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, BooleanType
from schema_validation import ChurnDataValidator, create_spark_session
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_data(spark):
    """Create test data for validation"""
    
    # Valid test data
    valid_data = [
        ("cust001", 35, "Male", 12, 75.50, 906.00, "Month-to-month", "Electronic check", "Fiber optic", "No", "No", True, None, None),
        ("cust002", 28, "Female", 24, 95.75, 2298.00, "One year", "Mailed check", "DSL", "Yes", "Yes", False, None, None),
        ("cust003", 45, "Male", 6, 55.25, 331.50, "Month-to-month", "Credit card", "No", "No internet service", "No", True, None, None),
        ("cust004", 32, "Female", 36, 105.00, 3780.00, "Two year", "Bank transfer", "Fiber optic", "Yes", "Yes", False, None, None),
        ("cust005", 51, "Male", 18, 85.50, 1539.00, "One year", "Electronic check", "DSL", "No", "No", False, None, None)
    ]
    
    schema = StructType([
        StructField("customer_id", StringType(), nullable=False),
        StructField("age", IntegerType(), nullable=True),
        StructField("gender", StringType(), nullable=True),
        StructField("tenure_months", IntegerType(), nullable=True),
        StructField("monthly_charges", DoubleType(), nullable=True),
        StructField("total_charges", DoubleType(), nullable=True),
        StructField("contract_type", StringType(), nullable=True),
        StructField("payment_method", StringType(), nullable=True),
        StructField("internet_service", StringType(), nullable=True),
        StructField("online_security", StringType(), nullable=True),
        StructField("tech_support", StringType(), nullable=True),
        StructField("churn", BooleanType(), nullable=False),
        StructField("signup_date", StringType(), nullable=True),  # Using String for simplicity in test
        StructField("last_interaction_date", StringType(), nullable=True)
    ])
    
    return spark.createDataFrame(valid_data, schema)

def create_invalid_data(spark):
    """Create invalid test data for validation testing"""
    
    invalid_data = [
        ("cust006", -5, "Other", 12, 75.50, 906.00, "Month-to-month", "Electronic check", "Fiber optic", "No", "No", True, None, None),  # Negative age
        (None, 28, "Female", 24, 95.75, 2298.00, "One year", "Mailed check", "DSL", "Yes", "Yes", False, None, None),  # Null customer_id
        ("cust008", 45, "Male", 6, 55.25, 331.50, "Month-to-month", "Credit card", "No", "No internet service", "No", None, None, None),  # Null churn
        ("cust009", 32, "Female", 36, 105.00, 3780.00, "Two year", "Bank transfer", "Fiber optic", "Yes", "Yes", False, None, None),
        ("cust010", 51, "Male", 18, 85.50, 1539.00, "One year", "Electronic check", "DSL", "No", "No", False, None, None)
    ]
    
    schema = StructType([
        StructField("customer_id", StringType(), nullable=True),  # Made nullable to test null validation
        StructField("age", IntegerType(), nullable=True),
        StructField("gender", StringType(), nullable=True),
        StructField("tenure_months", IntegerType(), nullable=True),
        StructField("monthly_charges", DoubleType(), nullable=True),
        StructField("total_charges", DoubleType(), nullable=True),
        StructField("contract_type", StringType(), nullable=True),
        StructField("payment_method", StringType(), nullable=True),
        StructField("internet_service", StringType(), nullable=True),
        StructField("online_security", StringType(), nullable=True),
        StructField("tech_support", StringType(), nullable=True),
        StructField("churn", BooleanType(), nullable=True),  # Made nullable to test null validation
        StructField("signup_date", StringType(), nullable=True),
        StructField("last_interaction_date", StringType(), nullable=True)
    ])
    
    return spark.createDataFrame(invalid_data, schema)

def test_schema_validation():
    """Test schema validation functionality"""
    logger.info("Starting schema validation tests...")
    
    spark = create_spark_session("SchemaValidationTest")
    validator = ChurnDataValidator(spark)
    
    try:
        # Test with valid data
        logger.info("Testing with valid data...")
        valid_df = create_test_data(spark)
        valid_report = validator.generate_validation_report(valid_df, "valid_test_data")
        
        logger.info(f"Valid data - Schema match: {valid_report['schema_validation']['schema_match']}")
        logger.info(f"Valid data - Total rows: {valid_report['data_quality']['total_rows']}")
        logger.info(f"Valid data - Quality score: {valid_report['data_quality']['data_quality_score']}")
        
        # Test with invalid data
        logger.info("Testing with invalid data...")
        invalid_df = create_invalid_data(spark)
        invalid_report = validator.generate_validation_report(invalid_df, "invalid_test_data")
        
        logger.info(f"Invalid data - Schema match: {invalid_report['schema_validation']['schema_match']}")
        logger.info(f"Invalid data - Total rows: {invalid_report['data_quality']['total_rows']}")
        logger.info(f"Invalid data - Quality score: {invalid_report['data_quality']['data_quality_score']}")
        logger.info(f"Invalid data - Null customer_id: {invalid_report['data_quality']['null_customer_id']}")
        logger.info(f"Invalid data - Null churn: {invalid_report['data_quality']['null_churn']}")
        logger.info(f"Invalid data - Negative values: {invalid_report['data_quality']['negative_values']}")
        
        logger.info("Schema validation tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in schema validation tests: {str(e)}")
        raise
    finally:
        spark.stop()

def test_row_count_validation():
    """Test row count and basic statistics"""
    logger.info("Starting row count validation tests...")
    
    spark = create_spark_session("RowCountTest")
    from data_processor import ChurnDataProcessor
    
    try:
        processor = ChurnDataProcessor(spark)
        
        # Test with valid data
        valid_df = create_test_data(spark)
        stats = processor.get_basic_statistics(valid_df)
        
        logger.info(f"Row count test - Total rows: {stats['total_rows']}")
        logger.info(f"Row count test - Total columns: {stats['total_columns']}")
        
        for col_stat in stats['column_statistics']:
            logger.info(f"Column {col_stat['column']}: {col_stat['non_null_count']} non-null, {col_stat['null_count']} null ({col_stat['null_percentage']}%)")
        
        # Test comprehensive validation
        report = processor.validate_and_report(valid_df, "row_count_test")
        logger.info(f"Comprehensive report generated for {report['dataset_name']}")
        
        logger.info("Row count validation tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in row count validation tests: {str(e)}")
        raise
    finally:
        spark.stop()

if __name__ == "__main__":
    logger.info("Starting Week 1 validation tests...")
    
    # Run tests
    test_schema_validation()
    test_row_count_validation()
    
    logger.info("All Week 1 tests completed successfully!")
