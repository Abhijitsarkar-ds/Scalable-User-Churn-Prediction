"""
Simple test for Week 1: Schema validation and row counts
Uses pandas for demonstration since Spark has Java compatibility issues
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleChurnValidator:
    """Simple validator for churn data using pandas"""
    
    def __init__(self):
        self.expected_schema = {
            'customer_id': str,
            'age': int,
            'gender': str,
            'tenure_months': int,
            'monthly_charges': float,
            'total_charges': float,
            'contract_type': str,
            'payment_method': str,
            'internet_service': str,
            'online_security': str,
            'tech_support': str,
            'churn': bool,
            'signup_date': str,
            'last_interaction_date': str
        }
    
    def create_sample_data(self) -> pd.DataFrame:
        """Create sample churn data for testing"""
        np.random.seed(42)
        n_customers = 1000
        
        data = {
            'customer_id': [f'cust_{i:06d}' for i in range(1, n_customers + 1)],
            'age': np.random.randint(18, 80, n_customers),
            'gender': np.random.choice(['Male', 'Female'], n_customers),
            'tenure_months': np.random.randint(1, 72, n_customers),
            'monthly_charges': np.random.uniform(20.0, 200.0, n_customers).round(2),
            'total_charges': None,  # Will be calculated
            'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_customers),
            'payment_method': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_customers),
            'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], n_customers),
            'online_security': np.random.choice(['Yes', 'No', 'No internet service'], n_customers),
            'tech_support': np.random.choice(['Yes', 'No', 'No internet service'], n_customers),
            'churn': np.random.choice([True, False], n_customers, p=[0.3, 0.7]),
            'signup_date': pd.date_range('2020-01-01', '2023-12-31', periods=n_customers).strftime('%Y-%m-%d'),
            'last_interaction_date': pd.date_range('2023-01-01', '2023-12-31', periods=n_customers).strftime('%Y-%m-%d')
        }
        
        df = pd.DataFrame(data)
        # Calculate total charges
        df['total_charges'] = (df['monthly_charges'] * df['tenure_months']).round(2)
        
        return df
    
    def validate_schema(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate DataFrame schema"""
        validation_results = {
            'schema_match': True,
            'missing_fields': [],
            'extra_fields': [],
            'type_mismatches': []
        }
        
        actual_fields = set(df.columns)
        expected_fields = set(self.expected_schema.keys())
        
        # Check for missing fields
        missing = expected_fields - actual_fields
        if missing:
            validation_results['missing_fields'] = list(missing)
            validation_results['schema_match'] = False
        
        # Check for extra fields
        extra = actual_fields - expected_fields
        if extra:
            validation_results['extra_fields'] = list(extra)
        
        # Check type mismatches
        for field, expected_type in self.expected_schema.items():
            if field in df.columns:
                actual_type = df[field].dtype
                # Simple type checking (can be enhanced)
                if expected_type == int and actual_type not in ['int64', 'int32']:
                    validation_results['type_mismatches'].append({
                        'field': field,
                        'expected': str(expected_type),
                        'actual': str(actual_type)
                    })
                    validation_results['schema_match'] = False
                elif expected_type == float and actual_type not in ['float64', 'float32']:
                    validation_results['type_mismatches'].append({
                        'field': field,
                        'expected': str(expected_type),
                        'actual': str(actual_type)
                    })
                    validation_results['schema_match'] = False
        
        return validation_results
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate data quality metrics"""
        total_rows = len(df)
        
        # Check for null values
        null_counts = df.isnull().sum()
        
        # Check for duplicate customer IDs
        duplicate_customers = df['customer_id'].duplicated().sum()
        
        # Check for negative values in numeric fields
        numeric_fields = ['age', 'tenure_months', 'monthly_charges', 'total_charges']
        negative_counts = {}
        for field in numeric_fields:
            if field in df.columns:
                negative_counts[field] = (df[field] < 0).sum()
        
        total_negatives = sum(negative_counts.values())
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(
            total_rows, null_counts, duplicate_customers, total_negatives
        )
        
        return {
            'total_rows': total_rows,
            'null_counts': null_counts.to_dict(),
            'duplicate_customers': duplicate_customers,
            'negative_counts': negative_counts,
            'total_negatives': total_negatives,
            'data_quality_score': quality_score
        }
    
    def _calculate_quality_score(self, total_rows: int, null_counts: pd.Series, 
                              duplicate_customers: int, negative_values: int) -> float:
        """Calculate overall data quality score (0-100)"""
        if total_rows == 0:
            return 0
        
        # Deductions for quality issues
        deductions = 0
        
        # Critical fields with nulls
        critical_nulls = null_counts.get('customer_id', 0) + null_counts.get('churn', 0)
        deductions += (critical_nulls / total_rows) * 40
        
        # Duplicates
        deductions += (duplicate_customers / total_rows) * 30
        
        # Negative values
        deductions += (negative_values / total_rows) * 30
        
        score = max(0, 100 - (deductions * 100))
        return round(score, 2)
    
    def generate_validation_report(self, df: pd.DataFrame, dataset_name: str = "churn_data") -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        logger.info(f"Starting validation for dataset: {dataset_name}")
        
        # Schema validation
        schema_results = self.validate_schema(df)
        
        # Data quality validation
        quality_results = self.validate_data_quality(df)
        
        # Generate report
        report = {
            'dataset': dataset_name,
            'validation_timestamp': pd.Timestamp.now().isoformat(),
            'schema_validation': schema_results,
            'data_quality': quality_results
        }
        
        # Log summary
        logger.info(f"Validation completed for {dataset_name}")
        logger.info(f"Total rows: {quality_results['total_rows']}")
        logger.info(f"Schema valid: {schema_results['schema_match']}")
        logger.info(f"Data quality score: {quality_results['data_quality_score']}")
        
        return report

def main():
    """Main function to run Week 1 validation tests"""
    logger.info("Starting Week 1: Schema Validation and Row Count Tests")
    
    # Initialize validator
    validator = SimpleChurnValidator()
    
    # Create sample data
    logger.info("Creating sample churn data...")
    df = validator.create_sample_data()
    
    # Run validation
    logger.info("Running schema validation...")
    schema_report = validator.validate_schema(df)
    
    logger.info("Running data quality validation...")
    quality_report = validator.validate_data_quality(df)
    
    # Generate comprehensive report
    logger.info("Generating validation report...")
    full_report = validator.generate_validation_report(df, "sample_churn_data")
    
    # Print summary
    print("\n" + "="*60)
    print("WEEK 1 VALIDATION SUMMARY")
    print("="*60)
    print(f"Dataset: {full_report['dataset']}")
    print(f"Total Rows: {full_report['data_quality']['total_rows']}")
    print(f"Schema Valid: {full_report['schema_validation']['schema_match']}")
    print(f"Data Quality Score: {full_report['data_quality']['data_quality_score']}")
    
    if full_report['schema_validation']['missing_fields']:
        print(f"Missing Fields: {full_report['schema_validation']['missing_fields']}")
    
    if full_report['schema_validation']['extra_fields']:
        print(f"Extra Fields: {full_report['schema_validation']['extra_fields']}")
    
    print(f"Duplicate Customers: {full_report['data_quality']['duplicate_customers']}")
    print(f"Negative Values: {full_report['data_quality']['total_negatives']}")
    
    # Show sample of data
    print("\nSample Data (first 5 rows):")
    print(df.head().to_string())
    
    print("\n" + "="*60)
    print("Week 1 validation completed successfully!")
    print("Ready for Week 2: Data Ingestion and Advanced Profiling")
    print("="*60)

if __name__ == "__main__":
    main()
