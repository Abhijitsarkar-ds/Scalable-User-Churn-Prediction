"""
Train churn model using pandas and scikit-learn
Alternative implementation due to PySpark Java compatibility issues
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChurnModelTrainer:
    """Train churn prediction models using scikit-learn"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.models = {}
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load and preprocess data"""
        try:
            # Load the IBM telecom data
            df = pd.read_csv(file_path)
            logger.info(f"Loaded data with shape: {df.shape}")
            
            # Basic cleaning
            df = df.dropna()
            logger.info(f"After dropping nulls: {df.shape}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data for modeling"""
        try:
            # Make a copy to avoid modifying original
            df_processed = df.copy()
            
            # Handle categorical variables
            categorical_columns = df_processed.select_dtypes(include=['object']).columns
            
            for col in categorical_columns:
                if col != 'customer_id':  # Skip ID column
                    le = LabelEncoder()
                    df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                    self.label_encoders[col] = le
            
            # Convert target variable if it's not numeric
            if 'churn' in df_processed.columns:
                if df_processed['churn'].dtype == 'object':
                    df_processed['churn'] = df_processed['churn'].map({'Yes': 1, 'No': 0, True: 1, False: 0})
            
            # Select numeric features
            numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
            feature_columns = [col for col in numeric_columns if col not in ['churn', 'customer_id']]
            
            logger.info(f"Feature columns: {feature_columns}")
            
            return df_processed, feature_columns
            
        except Exception as e:
            logger.error(f"Error preprocessing data: {str(e)}")
            raise
    
    def train_models(self, X_train, y_train):
        """Train multiple models"""
        try:
            # Logistic Regression
            lr = LogisticRegression(random_state=42, max_iter=1000)
            lr.fit(X_train, y_train)
            self.models['logistic_regression'] = lr
            
            # Random Forest
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)
            self.models['random_forest'] = rf
            
            logger.info("Models trained successfully")
            
        except Exception as e:
            logger.error(f"Error training models: {str(e)}")
            raise
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all trained models"""
        results = {}
        
        for name, model in self.models.items():
            try:
                # Predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # Metrics
                accuracy = accuracy_score(y_test, y_pred)
                auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
                
                results[name] = {
                    'accuracy': accuracy,
                    'auc': auc,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba
                }
                
                logger.info(f"{name} - Accuracy: {accuracy:.4f}, AUC: {auc:.4f if auc else 'N/A'}")
                
            except Exception as e:
                logger.error(f"Error evaluating {name}: {str(e)}")
        
        return results
    
    def save_models(self, model_dir: str = "models"):
        """Save trained models"""
        try:
            os.makedirs(model_dir, exist_ok=True)
            
            # Save models
            for name, model in self.models.items():
                model_path = os.path.join(model_dir, f"{name}_model.pkl")
                joblib.dump(model, model_path)
                logger.info(f"Saved {name} model to {model_path}")
            
            # Save preprocessors
            joblib.dump(self.scaler, os.path.join(model_dir, "scaler.pkl"))
            joblib.dump(self.label_encoders, os.path.join(model_dir, "label_encoders.pkl"))
            
            logger.info(f"All models saved to {model_dir}")
            
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")
            raise

def main():
    """Main training pipeline"""
    logger.info("Starting Churn Model Training")
    
    # Initialize trainer
    trainer = ChurnModelTrainer()
    
    # Load data
    data_path = "/Users/tarandeepsingh/Desktop/internship project 4/SparkScale-churn/data/raw/ibm_telco_churn.csv"
    
    if os.path.exists(data_path):
        df = trainer.load_data(data_path)
        
        # Display basic info
        print("\n" + "="*60)
        print("DATA OVERVIEW")
        print("="*60)
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        if 'churn' in df.columns:
            print(f"\nChurn Distribution:")
            print(df['churn'].value_counts())
        
        print("\nFirst few rows:")
        print(df.head())
        
        # Preprocess
        df_processed, feature_columns = trainer.preprocess_data(df)
        
        # Prepare features and target
        X = df_processed[feature_columns]
        y = df_processed['churn'] if 'churn' in df_processed.columns else df_processed.iloc[:, -1]  # Assume last column is target
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = trainer.scaler.fit_transform(X_train)
        X_test_scaled = trainer.scaler.transform(X_test)
        
        print(f"\nTrain set: {X_train_scaled.shape}")
        print(f"Test set: {X_test_scaled.shape}")
        
        # Train models
        trainer.train_models(X_train_scaled, y_train)
        
        # Evaluate models
        results = trainer.evaluate_models(X_test_scaled, y_test)
        
        # Display results
        print("\n" + "="*60)
        print("MODEL EVALUATION RESULTS")
        print("="*60)
        
        for name, metrics in results.items():
            print(f"\n{name.upper()}:")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            if metrics['auc']:
                print(f"  AUC: {metrics['auc']:.4f}")
            
            # Classification report
            y_pred = metrics['predictions']
            print(f"  Classification Report:")
            print(classification_report(y_test, y_pred))
        
        # Save models
        trainer.save_models()
        
        print("\n" + "="*60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("Models saved to: models/")
        print("Files created:")
        print("  - logistic_regression_model.pkl")
        print("  - random_forest_model.pkl")
        print("  - scaler.pkl")
        print("  - label_encoders.pkl")
        
    else:
        logger.error(f"Data file not found: {data_path}")
        print("Please ensure the IBM telecom dataset exists at the specified path.")

if __name__ == "__main__":
    main()
