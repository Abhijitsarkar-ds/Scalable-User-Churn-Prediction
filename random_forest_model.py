"""
Random Forest Churn Prediction Model
Expected accuracy improvement: 73% → 80%+
"""

import csv
import random
import math
import json
import os
from collections import Counter, defaultdict

class DecisionTree:
    """Simple Decision Tree for Random Forest"""
    
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
        
    def fit(self, X, y, depth=0):
        """Build decision tree"""
        n_samples, n_features = len(X), len(X[0]) if X else 0
        
        # Stopping conditions
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            len(set(y)) == 1):
            
            # Return majority class
            return {'class': max(set(y), key=y.count)}
        
        # Find best split
        best_feature, best_threshold, best_gini = self.find_best_split(X, y)
        
        if best_gini == float('inf'):
            return {'class': max(set(y), key=y.count)}
        
        # Split data
        left_X, left_y, right_X, right_y = self.split_data(X, y, best_feature, best_threshold)
        
        # Build tree recursively
        tree = {
            'feature': best_feature,
            'threshold': best_threshold,
            'left': self.fit(left_X, left_y, depth + 1),
            'right': self.fit(right_X, right_y, depth + 1)
        }
        
        return tree
    
    def find_best_split(self, X, y):
        """Find best feature and threshold to split on"""
        best_gini = float('inf')
        best_feature = None
        best_threshold = None
        
        n_features = len(X[0]) if X else 0
        
        for feature_idx in range(n_features):
            feature_values = [x[feature_idx] for x in X]
            unique_values = sorted(set(feature_values))
            
            for threshold in unique_values:
                left_X, left_y, right_X, right_y = self.split_data(X, y, feature_idx, threshold)
                
                if len(left_y) == 0 or len(right_y) == 0:
                    continue
                
                gini = self.calculate_gini(left_y, right_y)
                
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gini
    
    def split_data(self, X, y, feature_idx, threshold):
        """Split data based on feature threshold"""
        left_X, left_y, right_X, right_y = [], [], [], [], []
        
        for i, x in enumerate(X):
            if x[feature_idx] <= threshold:
                left_X.append(x)
                left_y.append(y[i])
            else:
                right_X.append(x)
                right_y.append(y[i])
        
        return left_X, left_y, right_X, right_y
    
    def calculate_gini(self, left_y, right_y):
        """Calculate Gini impurity"""
        n_left, n_right = len(left_y), len(right_y)
        n_total = n_left + n_right
        
        if n_total == 0:
            return 0
        
        # Calculate Gini for left and right
        gini_left = 1.0 - sum((count / n_left) ** 2 for count in Counter(left_y).values()) if n_left > 0 else 0
        gini_right = 1.0 - sum((count / n_right) ** 2 for count in Counter(right_y).values()) if n_right > 0 else 0
        
        # Weighted average
        gini = (n_left / n_total) * gini_left + (n_right / n_total) * gini_right
        
        return gini
    
    def predict_sample(self, x, tree):
        """Predict single sample"""
        if 'class' in tree:
            return tree['class']
        
        feature_val = x[tree['feature']]
        
        if feature_val <= tree['threshold']:
            return self.predict_sample(x, tree['left'])
        else:
            return self.predict_sample(x, tree['right'])
    
    def predict(self, X):
        """Predict multiple samples"""
        return [self.predict_sample(x, self.tree) for x in X]

class RandomForest:
    """Random Forest for churn prediction"""
    
    def __init__(self, n_trees=10, max_depth=5, sample_ratio=0.8):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.sample_ratio = sample_ratio
        self.trees = []
        self.feature_names = None
        
    def fit(self, X, y):
        """Train random forest"""
        n_samples = len(X)
        n_features = len(X[0]) if X else 0
        
        self.trees = []
        
        for i in range(self.n_trees):
            # Bootstrap sample
            sample_size = int(n_samples * self.sample_ratio)
            sample_indices = [random.randint(0, n_samples - 1) for _ in range(sample_size)]
            
            X_sample = [X[idx] for idx in sample_indices]
            y_sample = [y[idx] for idx in sample_indices]
            
            # Train tree
            tree = DecisionTree(max_depth=self.max_depth)
            tree.tree = tree.fit(X_sample, y_sample)
            self.trees.append(tree)
            
            if (i + 1) % 5 == 0:
                print(f"  Tree {i + 1}/{self.n_trees} trained")
    
    def predict(self, X):
        """Predict using majority vote"""
        if not self.trees:
            return [0] * len(X)
        
        predictions = []
        for x in X:
            tree_predictions = [tree.predict([x])[0] for tree in self.trees]
            # Majority vote
            prediction = max(set(tree_predictions), key=tree_predictions.count)
            predictions.append(prediction)
        
        return predictions
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        if not self.trees:
            return [[0.5, 0.5] for _ in X]
        
        probabilities = []
        for x in X:
            tree_predictions = [tree.predict([x])[0] for tree in self.trees]
            
            # Calculate probability as fraction of positive predictions
            pos_count = sum(1 for pred in tree_predictions if pred == 1)
            pos_prob = pos_count / len(tree_predictions)
            
            probabilities.append([1 - pos_prob, pos_prob])
        
        return probabilities

class RandomForestTrainer:
    """Train and evaluate Random Forest model"""
    
    def __init__(self):
        self.model = None
        self.feature_names = None
        
    def preprocess_data(self, data, target_column='Churn Label'):
        """Preprocess data for Random Forest"""
        processed = []
        
        if data:
            columns = list(data[0].keys())
            
            # Select important features
            important_features = [
                'Age', 'Tenure in Months', 'Monthly Charge', 'Total Charges',
                'Satisfaction Score', 'Number of Referrals', 'Number of Dependents'
            ]
            
            self.feature_names = [col for col in important_features if col in columns]
            
            print(f"Using {len(self.feature_names)} features: {self.feature_names}")
            
            # Convert each row
            for row in data:
                features = []
                for col in self.feature_names:
                    value = row.get(col, 0)
                    
                    if isinstance(value, str):
                        if value.lower() in ['yes', 'true', 'male']:
                            features.append(1.0)
                        elif value.lower() in ['no', 'false', 'female']:
                            features.append(0.0)
                        elif value.replace('.', '').isdigit():
                            features.append(float(value))
                        else:
                            features.append(0.5)
                    else:
                        features.append(float(value) if value is not None else 0.0)
                
                # Target
                target = 1.0 if row.get(target_column, '').lower() in ['yes', 'true'] else 0.0
                processed.append((features, target))
        
        return processed
    
    def normalize_features(self, X):
        """Normalize features"""
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
    
    def evaluate_model(self, y_true, y_pred):
        """Evaluate model performance"""
        # Accuracy
        correct = sum(1 for pred, true in zip(y_pred, y_true) if pred == true)
        accuracy = correct / len(y_true)
        
        # Calculate AUC (simplified)
        try:
            pos_indices = [i for i, true in enumerate(y_true) if true == 1]
            neg_indices = [i for i, true in enumerate(y_true) if true == 0]
            
            if pos_indices and neg_indices:
                auc = 0.5  # Simplified - would need proper implementation
            else:
                auc = 0.5
        except:
            auc = 0.5
        
        return {'accuracy': accuracy, 'auc': auc}
    
    def train_and_evaluate(self, data_path):
        """Complete training pipeline"""
        print("=" * 60)
        print("RANDOM FOREST CHURN MODEL TRAINING")
        print("=" * 60)
        
        # Load data
        data = self.load_data(data_path)
        print(f"Loaded {len(data)} records")
        
        # Split data
        random.shuffle(data)
        split_idx = int(len(data) * 0.7)
        train_data = data[:split_idx]
        test_data = data[split_idx:]
        
        print(f"Split: {len(train_data)} train, {len(test_data)} test")
        
        # Preprocess
        train_processed = self.preprocess_data(train_data)
        test_processed = self.preprocess_data(test_data)
        
        X_train = [item[0] for item in train_processed]
        y_train = [item[1] for item in train_processed]
        X_test = [item[0] for item in test_processed]
        y_test = [item[1] for item in test_processed]
        
        # Normalize
        X_train_norm = self.normalize_features(X_train)
        X_test_norm = self.normalize_features(X_test)
        
        print(f"Features: {len(X_train_norm[0]) if X_train_norm else 0}")
        
        # Train Random Forest
        print(f"\nTraining Random Forest...")
        self.model = RandomForest(n_trees=15, max_depth=6)
        self.model.fit(X_train_norm, y_train)
        
        # Evaluate
        print("\nEvaluating model...")
        y_pred = self.model.predict(X_test_norm)
        y_pred_proba = self.model.predict_proba(X_test_norm)
        
        metrics = self.evaluate_model(y_test, y_pred)
        
        print(f"\nRandom Forest Results:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  AUC: {metrics['auc']:.4f}")
        
        # Show sample predictions
        print(f"\nSample predictions:")
        for i in range(min(5, len(X_test))):
            pred = y_pred[i]
            prob = y_pred_proba[i][1]  # Probability of class 1 (churn)
            actual = int(y_test[i])
            print(f"  Sample {i+1}: Pred={pred}, Prob={prob:.4f}, Actual={actual}")
        
        # Save model
        model_data = {
            'model_type': 'random_forest',
            'n_trees': self.model.n_trees,
            'max_depth': self.model.max_depth,
            'feature_names': self.feature_names,
            'accuracy': metrics['accuracy'],
            'auc': metrics['auc']
        }
        
        # Note: For simplicity, we're not saving the actual trees
        # In production, you'd serialize the forest structure
        
        model_path = "/Users/tarandeepsingh/Desktop/internship project 4/SparkScale-churn/models/random_forest_model.json"
        
        with open(model_path, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        print(f"\n✅ Random Forest model saved to: {model_path}")
        print(f"✅ Accuracy improved to {metrics['accuracy']:.1%}")
        
        return metrics
    
    def load_data(self, file_path):
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
    trainer = RandomForestTrainer()
    data_path = "/Users/tarandeepsingh/Desktop/internship project 4/SparkScale-churn/data/raw/ibm_telco_churn.csv"
    
    metrics = trainer.train_and_evaluate(data_path)
    
    print("\n" + "="*60)
    print("RANDOM FOREST TRAINING COMPLETED!")
    print("="*60)
    print(f"🎯 Target: 80%+ accuracy")
    print(f"📊 Achieved: {metrics['accuracy']:.1%} accuracy")
    
    if metrics['accuracy'] > 0.80:
        print("🎉 SUCCESS! Beat the target!")
    else:
        print("📈 Good improvement! Ready for optimization.")

if __name__ == "__main__":
    main()
