"""
Test data processor with pandas (Week 1 alternative)
Tests data loading and basic statistics
"""

import pandas as pd
import logging
import os
from simple_test import SimpleChurnValidator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleDataProcessor:
    """Simple data processor using pandas"""
    
    def __init__(self):
        self.validator = SimpleChurnValidator()
    
    def load_data(self, file_path: str, file_format: str = "csv") -> pd.DataFrame:
        """Load data from file"""
        try:
            if file_format.lower() == "csv":
                df = pd.read_csv(file_path)
            elif file_format.lower() == "json":
                df = pd.read_json(file_path)
            elif file_format.lower() == "parquet":
                df = pd.read_parquet(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_format}")
            
            logger.info(f"Successfully loaded data from {file_path}")
            logger.info(f"Data shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {str(e)}")
            raise
    
    def get_basic_statistics(self, df: pd.DataFrame) -> dict:
        """Get basic statistics and row counts"""
        try:
            total_rows = len(df)
            total_columns = len(df.columns)
            
            # Get column-wise statistics
            column_stats = []
            for col_name in df.columns:
                non_null_count = df[col_name].count()
                null_count = total_rows - non_null_count
                null_percentage = round((null_count / total_rows) * 100, 2) if total_rows > 0 else 0
                
                # Get data type
                dtype = str(df[col_name].dtype)
                
                column_stats.append({
                    'column': col_name,
                    'data_type': dtype,
                    'non_null_count': non_null_count,
                    'null_count': null_count,
                    'null_percentage': null_percentage
                })
            
            # Numeric statistics
            numeric_cols = df.select_dtypes(include=['number']).columns
            numeric_stats = {}
            for col in numeric_cols:
                numeric_stats[col] = {
                    'mean': round(df[col].mean(), 2),
                    'std': round(df[col].std(), 2),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'median': df[col].median()
                }
            
            return {
                'total_rows': total_rows,
                'total_columns': total_columns,
                'column_statistics': column_stats,
                'numeric_statistics': numeric_stats
            }
            
        except Exception as e:
            logger.error(f"Error getting basic statistics: {str(e)}")
            raise
    
    def validate_and_report(self, df: pd.DataFrame, dataset_name: str = "churn_data") -> dict:
        """Run comprehensive validation and generate report"""
        try:
            # Get basic statistics
            basic_stats = self.get_basic_statistics(df)
            
            # Run schema validation
            validation_report = self.validator.generate_validation_report(df, dataset_name)
            
            # Combine reports
            comprehensive_report = {
                'dataset_name': dataset_name,
                'basic_statistics': basic_stats,
                'validation_report': validation_report
            }
            
            logger.info(f"Validation and reporting completed for {dataset_name}")
            return comprehensive_report
            
        except Exception as e:
            logger.error(f"Error in validation and reporting: {str(e)}")
            raise
    
    def save_validation_report(self, report: dict, output_path: str):
        """Save validation report as JSON"""
        try:
            import json
            
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Validation report saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving validation report: {str(e)}")
            raise

def main():
    """Main function to test data processor"""
    logger.info("Testing Data Processor - Week 1")
    
    # Initialize processor
    processor = SimpleDataProcessor()
    
    # Test data loading
    data_path = "/Users/tarandeepsingh/Desktop/internship project 4/SparkScale-churn/data/sample_churn_data.csv"
    
    if os.path.exists(data_path):
        logger.info(f"Loading data from {data_path}")
        df = processor.load_data(data_path, "csv")
        
        # Get basic statistics
        logger.info("Getting basic statistics...")
        stats = processor.get_basic_statistics(df)
        
        # Run validation
        logger.info("Running validation...")
        report = processor.validate_and_report(df, "sample_churn_data")
        
        # Save report
        output_path = "/Users/tarandeepsingh/Desktop/internship project 4/SparkScale-churn/validation_report.json"
        processor.save_validation_report(report, output_path)
        
        # Print summary
        print("\n" + "="*60)
        print("DATA PROCESSOR TEST RESULTS")
        print("="*60)
        print(f"Dataset: {report['dataset_name']}")
        print(f"Total Rows: {report['basic_statistics']['total_rows']}")
        print(f"Total Columns: {report['basic_statistics']['total_columns']}")
        print(f"Schema Valid: {report['validation_report']['schema_validation']['schema_match']}")
        print(f"Data Quality Score: {report['validation_report']['data_quality']['data_quality_score']}")
        
        print("\nColumn Statistics:")
        for col_stat in report['basic_statistics']['column_statistics']:
            print(f"  {col_stat['column']}: {col_stat['data_type']} - {col_stat['null_count']} nulls ({col_stat['null_percentage']}%)")
        
        print("\nNumeric Statistics:")
        for col, stats in report['basic_statistics']['numeric_statistics'].items():
            print(f"  {col}: mean={stats['mean']}, std={stats['std']}, min={stats['min']}, max={stats['max']}")
        
        print(f"\nValidation report saved to: {output_path}")
        print("="*60)
        
    else:
        logger.error(f"Data file not found: {data_path}")
        print("Please ensure the sample data file exists before running this test.")

if __name__ == "__main__":
    main()
