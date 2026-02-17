"""
Simple model training using basic Python libraries
Trains churn prediction model on IBM telecom data
"""

import csv
import random
import math
from collections import defaultdict, Counter
import json
import os

class SimpleChurnModel:
    """Simple logistic regression implementation for churn prediction"""
    
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.feature_names = None
        
    def sigmoid(self, x):
        """Sigmoid activation function"""
        try:
            return 1 / (1 + math.exp(-x))
        except:
            return 0.5
    
    def preprocess_data(self, data, target_column='Churn Label'):
        """Convert categorical data to numerical"""
        processed = []
        self.feature_names = []
        
        # Get all column names
        if data:
            columns = list(data[0].keys())
            
            # Remove non-numeric and target columns for simplicity
            exclude_cols = ['Customer ID', target_column, 'Churn Category', 'Churn Reason', 'Customer Status']
            feature_cols = [col for col in columns if col not in exclude_cols]
            
            # Simple encoding for categorical variables
            for col in feature_cols:
                if col not in self.feature_names:
                    self.feature_names.append(col)
            
            # Convert each row
            for row in data:
                features = []
                for col in self.feature_names:
                    value = row.get(col, 0)
                    
                    # Simple encoding
                    if isinstance(value, str):
                        if value.lower() in ['yes', 'true', 'male']:
                            features.append(1.0)
                        elif value.lower() in ['no', 'false', 'female']:
                            features.append(0.0)
                        elif value.replace('.', '').replace('-', '').isdigit():
                            features.append(float(value))
                        else:
                            # Hash other strings
                            features.append(float(hash(value) % 1000) / 1000.0)
                    else:
                        features.append(float(value) if value is not None else 0.0)
                
                # Target
                target = 1.0 if row.get(target_column, '').lower() in ['yes', 'true'] else 0.0
                
                processed.append((features, target))
        
        return processed
    
    def train(self, X, y):
        """Train the model"""
        n_samples, n_features = len(X), len(X[0]) if X else 0
        
        # Initialize weights and bias
        self.weights = [0.0] * n_features
        self.bias = 0.0
        
        # Gradient descent
        for epoch in range(self.epochs):
            for i in range(n_samples):
                # Forward pass
                linear_output = sum(w * x for w, x in zip(self.weights, X[i])) + self.bias
                prediction = self.sigmoid(linear_output)
                
                # Calculate error
                error = prediction - y[i]
                
                # Update weights
                for j in range(n_features):
                    self.weights[j] -= self.learning_rate * error * X[i][j]
                self.bias -= self.learning_rate * error
            
            # Print progress
            if epoch % 100 == 0:
                predictions = [self.predict(x) for x in X]
                accuracy = sum(1 for pred, true in zip(predictions, y) if round(pred) == true) / len(y)
                print(f"Epoch {epoch}: Accuracy = {accuracy:.4f}")
    
    def predict(self, X):
        """Make prediction"""
        if self.weights is None:
            return 0.0
        
        linear_output = sum(w * x for w, x in zip(self.weights, X)) + self.bias
        return self.sigmoid(linear_output)
    
    def predict_proba(self, X):
        """Get prediction probability"""
        pred = self.predict(X)
        return [1 - pred, pred]  # [class_0_prob, class_1_prob]

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

def split_data(data, test_size=0.3):
    """Split data into train and test sets"""
    random.shuffle(data)
    split_idx = int(len(data) * (1 - test_size))
    return data[:split_idx], data[split_idx:]

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    predictions = [model.predict(x) for x in X_test]
    pred_classes = [round(p) for p in predictions]
    
    # Calculate metrics
    correct = sum(1 for pred, true in zip(pred_classes, y_test) if pred == true)
    accuracy = correct / len(y_test)
    
    # Calculate AUC (simplified)
    try:
        from collections import defaultdict
        pos_scores = [p for p, t in zip(predictions, y_test) if t == 1]
        neg_scores = [p for p, t in zip(predictions, y_test) if t == 0]
        
        if pos_scores and neg_scores:
            auc = sum(1 for pos in pos_scores for neg in neg_scores if pos > neg) / (len(pos_scores) * len(neg_scores))
        else:
            auc = 0.5
    except:
        auc = 0.5
    
    return {
        'accuracy': accuracy,
        'auc': auc,
        'predictions': pred_classes,
        'probabilities': predictions
    }

def main():
    """Main training function"""
    print("=" * 60)
    print("CHURN PREDICTION MODEL TRAINING")
    print("=" * 60)
    
    # Load data
    data_path = "/Users/tarandeepsingh/Desktop/internship project 4/SparkScale-churn/data/raw/ibm_telco_churn.csv"
    
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        return
    
    print(f"Loading data from: {data_path}")
    data = load_csv_data(data_path)
    print(f"Loaded {len(data)} records")
    
    if not data:
        print("No data loaded!")
        return
    
    # Show data sample
    print("\nSample data:")
    for i, row in enumerate(data[:3]):
        print(f"Row {i+1}: {dict(list(row.items())[:5])}...")
    
    # Split data
    train_data, test_data = split_data(data, test_size=0.3)
    print(f"\nData split: {len(train_data)} train, {len(test_data)} test")
    
    # Preprocess
    model = SimpleChurnModel(learning_rate=0.01, epochs=500)
    
    print("\nPreprocessing data...")
    train_processed = model.preprocess_data(train_data)
    test_processed = model.preprocess_data(test_data)
    
    X_train = [item[0] for item in train_processed]
    y_train = [item[1] for item in train_processed]
    X_test = [item[0] for item in test_processed]
    y_test = [item[1] for item in test_processed]
    
    print(f"Features: {len(X_train[0]) if X_train else 0}")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # Train model
    print("\nTraining model...")
    model.train(X_train, y_train)
    
    # Evaluate
    print("\nEvaluating model...")
    results = evaluate_model(model, X_test, y_test)
    
    print("\n" + "=" * 60)
    print("MODEL EVALUATION RESULTS")
    print("=" * 60)
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"AUC: {results['auc']:.4f}")
    
    # Show some predictions
    print(f"\nSample predictions (first 10):")
    for i in range(min(10, len(X_test))):
        pred = results['predictions'][i]
        prob = results['probabilities'][i]
        actual = int(y_test[i])
        print(f"  Sample {i+1}: Predicted={pred}, Probability={prob:.4f}, Actual={actual}")
    
    # Save model
    model_data = {
        'weights': model.weights,
        'bias': model.bias,
        'feature_names': model.feature_names,
        'accuracy': results['accuracy'],
        'auc': results['auc']
    }
    
    model_path = "/Users/tarandeepsingh/Desktop/internship project 4/SparkScale-churn/models/simple_churn_model.json"
    
    try:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, 'w') as f:
            json.dump(model_data, f, indent=2)
        print(f"\n✅ Model saved to: {model_path}")
    except Exception as e:
        print(f"Error saving model: {e}")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    main()
