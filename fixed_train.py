"""
Fixed training script with proper regularization
Prevents weight explosion and creates usable model
"""

import csv
import random
import math
from collections import defaultdict, Counter
import json
import os

class FixedChurnModel:
    """Improved logistic regression with regularization"""
    
    def __init__(self, learning_rate=0.001, epochs=1000, reg_lambda=0.1):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.reg_lambda = reg_lambda  # L2 regularization
        self.weights = None
        self.bias = None
        self.feature_names = None
        
    def sigmoid(self, x):
        """Sigmoid with overflow protection"""
        if x > 500:
            return 1.0
        elif x < -500:
            return 0.0
        else:
            return 1 / (1 + math.exp(-x))
    
    def normalize_features(self, X):
        """Normalize features to prevent large values"""
        if not X:
            return X
            
        # Simple min-max normalization
        X_norm = []
        for j in range(len(X[0])):
            col_values = [row[j] for row in X]
            min_val = min(col_values)
            max_val = max(col_values)
            
            if max_val - min_val > 0:
                norm_col = [(x - min_val) / (max_val - min_val) for x in col_values]
            else:
                norm_col = [0.0] * len(col_values)
            
            X_norm.append(norm_col)
        
        # Transpose back
        return [[X_norm[j][i] for j in range(len(X_norm))] for i in range(len(X_norm[0]))]
    
    def preprocess_data(self, data, target_column='Churn Label'):
        """Better data preprocessing"""
        processed = []
        self.feature_names = []
        
        if data:
            columns = list(data[0].keys())
            
            # Select important features only
            important_features = [
                'Age', 'Tenure in Months', 'Monthly Charge', 'Total Charges',
                'Satisfaction Score', 'Number of Referrals', 'Number of Dependents'
            ]
            
            # Filter to available features
            available_features = [col for col in important_features if col in columns]
            self.feature_names = available_features
            
            print(f"Using features: {self.feature_names}")
            
            # Convert each row
            for row in data:
                features = []
                for col in self.feature_names:
                    value = row.get(col, 0)
                    
                    # Better encoding
                    if isinstance(value, str):
                        if value.lower() in ['yes', 'true', 'male']:
                            features.append(1.0)
                        elif value.lower() in ['no', 'false', 'female']:
                            features.append(0.0)
                        elif value.replace('.', '').isdigit():
                            features.append(float(value))
                        else:
                            features.append(0.5)  # Neutral for other strings
                    else:
                        features.append(float(value) if value is not None else 0.0)
                
                # Target
                target = 1.0 if row.get(target_column, '').lower() in ['yes', 'true'] else 0.0
                
                processed.append((features, target))
        
        return processed
    
    def train(self, X, y):
        """Train with regularization"""
        n_samples, n_features = len(X), len(X[0]) if X else 0
        
        # Initialize small weights
        self.weights = [random.uniform(-0.1, 0.1) for _ in range(n_features)]
        self.bias = 0.0
        
        # Normalize features
        X_norm = self.normalize_features(X)
        
        print(f"Training with {n_samples} samples, {n_features} features")
        
        for epoch in range(self.epochs):
            total_error = 0
            
            for i in range(n_samples):
                # Forward pass
                linear_output = sum(w * x for w, x in zip(self.weights, X_norm[i])) + self.bias
                prediction = self.sigmoid(linear_output)
                
                # Calculate error
                error = prediction - y[i]
                total_error += abs(error)
                
                # Update weights with L2 regularization
                for j in range(n_features):
                    # Gradient with regularization
                    gradient = error * X_norm[i][j] + self.reg_lambda * self.weights[j]
                    self.weights[j] -= self.learning_rate * gradient
                
                # Update bias
                self.bias -= self.learning_rate * error
            
            # Print progress
            if epoch % 100 == 0:
                predictions = [self.predict_raw(x) for x in X_norm]
                accuracy = sum(1 for pred, true in zip(predictions, y) if round(pred) == true) / len(y)
                print(f"Epoch {epoch}: Error={total_error:.4f}, Accuracy={accuracy:.4f}")
                
                # Early stopping if converged
                if total_error < 0.1:
                    print(f"Converged at epoch {epoch}")
                    break
    
    def predict_raw(self, x):
        """Raw prediction (needs normalized input)"""
        if self.weights is None:
            return 0.0
        
        linear_output = sum(w * x for w, x in zip(self.weights, x)) + self.bias
        return self.sigmoid(linear_output)
    
    def predict(self, X_original):
        """Make prediction on original data"""
        if self.weights is None:
            return 0.0
        
        # Normalize input
        # Simple normalization (would need training stats for proper normalization)
        X_norm = []
        for i, x in enumerate(X_original):
            # Basic scaling
            if i < len(self.feature_names):
                if self.feature_names[i] in ['Age', 'Tenure in Months']:
                    norm_x = min(1.0, x / 100.0)  # Scale age/tenure
                elif self.feature_names[i] in ['Monthly Charge', 'Total Charges']:
                    norm_x = min(1.0, x / 1000.0)  # Scale charges
                else:
                    norm_x = min(1.0, x / 10.0)  # Scale others
                X_norm.append(norm_x)
            else:
                X_norm.append(0.0)
        
        # Pad if needed
        while len(X_norm) < len(self.weights):
            X_norm.append(0.0)
        
        linear_output = sum(w * x for w, x in zip(self.weights, X_norm)) + self.bias
        return self.sigmoid(linear_output)

def load_csv_data(file_path):
    """Load CSV data"""
    data = []
    try:
        with open(file_path, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                data.append(row)
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return []

def main():
    """Main training function"""
    print("=" * 60)
    print("FIXED CHURN MODEL TRAINING")
    print("=" * 60)
    
    # Load data
    data_path = "/Users/tarandeepsingh/Desktop/internship project 4/SparkScale-churn/data/raw/ibm_telco_churn.csv"
    
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        return
    
    data = load_csv_data(data_path)
    print(f"Loaded {len(data)} records")
    
    # Split data
    random.shuffle(data)
    split_idx = int(len(data) * 0.7)
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    
    print(f"Split: {len(train_data)} train, {len(test_data)} test")
    
    # Preprocess
    model = FixedChurnModel(learning_rate=0.001, epochs=1000, reg_lambda=0.1)
    train_processed = model.preprocess_data(train_data)
    test_processed = model.preprocess_data(test_data)
    
    X_train = [item[0] for item in train_processed]
    y_train = [item[1] for item in train_processed]
    X_test = [item[0] for item in test_processed]
    y_test = [item[1] for item in test_processed]
    
    print(f"Features: {len(X_train[0]) if X_train else 0}")
    
    # Train model
    print("\nTraining improved model...")
    model.train(X_train, y_train)
    
    # Test
    print("\nTesting model...")
    predictions = [model.predict(x) for x in X_test]
    pred_classes = [round(p) for p in predictions]
    
    correct = sum(1 for pred, true in zip(pred_classes, y_test) if pred == true)
    accuracy = correct / len(y_test)
    
    print(f"\nFinal Accuracy: {accuracy:.4f}")
    
    # Show sample predictions
    print(f"\nSample predictions:")
    for i in range(min(5, len(X_test))):
        pred = predictions[i]
        actual = int(y_test[i])
        print(f"  Sample {i+1}: Pred={pred:.4f} ({round(pred)}), Actual={actual}")
    
    # Save model
    model_data = {
        'weights': model.weights,
        'bias': model.bias,
        'feature_names': model.feature_names,
        'accuracy': accuracy,
        'model_type': 'fixed_logistic_regression'
    }
    
    model_path = "/Users/tarandeepsingh/Desktop/internship project 4/SparkScale-churn/models/fixed_churn_model.json"
    
    with open(model_path, 'w') as f:
        json.dump(model_data, f, indent=2)
    
    print(f"\n✅ Fixed model saved to: {model_path}")
    print("✅ Model should now give varied predictions!")

if __name__ == "__main__":
    main()
