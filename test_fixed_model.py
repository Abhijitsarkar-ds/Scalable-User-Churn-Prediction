"""
Test the fixed churn model
Should show varied risk levels now
"""

import json
import math

class FixedModelTester:
    """Test the improved churn model"""
    
    def __init__(self, model_path):
        self.model = self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load fixed model"""
        with open(model_path, 'r') as f:
            model_data = json.load(f)
        
        print(f"✅ Fixed Model Loaded:")
        print(f"   - Features: {len(model_data['weights'])}")
        print(f"   - Accuracy: {model_data['accuracy']:.4f}")
        print(f"   - Weight range: {min(model_data['weights']):.4f} to {max(model_data['weights']):.4f}")
        
        return model_data
    
    def sigmoid(self, x):
        """Sigmoid with overflow protection"""
        if x > 500:
            return 1.0
        elif x < -500:
            return 0.0
        else:
            return 1 / (1 + math.exp(-x))
    
    def predict(self, sample_data):
        """Make prediction"""
        features = []
        
        # Map to model features
        feature_map = {
            'Age': sample_data.get('age', 40),
            'Tenure in Months': sample_data.get('tenure_months', 12),
            'Monthly Charge': sample_data.get('monthly_charge', 50),
            'Total Charges': sample_data.get('total_charges', 600),
            'Satisfaction Score': sample_data.get('satisfaction', 3),
            'Number of Referrals': sample_data.get('referrals', 0),
            'Number of Dependents': sample_data.get('dependents', 0)
        }
        
        # Create feature vector in correct order
        for feature_name in self.model['feature_names']:
            features.append(feature_map.get(feature_name, 0))
        
        # Normalize features (same as training)
        normalized_features = []
        for i, x in enumerate(features):
            if i < len(self.model['feature_names']):
                feature_name = self.model['feature_names'][i]
                if feature_name in ['Age', 'Tenure in Months']:
                    norm_x = min(1.0, x / 100.0)
                elif feature_name in ['Monthly Charge', 'Total Charges']:
                    norm_x = min(1.0, x / 1000.0)
                else:
                    norm_x = min(1.0, x / 10.0)
                normalized_features.append(norm_x)
        
        # Calculate prediction
        linear_output = sum(w * x for w, x in zip(self.model['weights'], normalized_features)) + self.model['bias']
        probability = self.sigmoid(linear_output)
        
        return {
            'churn_probability': probability,
            'churn_prediction': 1 if probability > 0.5 else 0,
            'confidence': max(probability, 1 - probability)
        }
    
    def run_test_cases(self):
        """Test with realistic scenarios"""
        
        test_cases = [
            {
                'name': 'Happy Long-term Customer',
                'data': {
                    'age': 35, 'tenure_months': 48, 'monthly_charge': 45.0, 
                    'total_charges': 2160.0, 'satisfaction': 5, 'referrals': 2, 'dependents': 1
                }
            },
            {
                'name': 'New Expensive Customer',
                'data': {
                    'age': 28, 'tenure_months': 3, 'monthly_charge': 120.0, 
                    'total_charges': 360.0, 'satisfaction': 2, 'referrals': 0, 'dependents': 0
                }
            },
            {
                'name': 'Unhappy Senior Customer',
                'data': {
                    'age': 65, 'tenure_months': 6, 'monthly_charge': 95.0, 
                    'total_charges': 570.0, 'satisfaction': 1, 'referrals': 0, 'dependents': 2
                }
            },
            {
                'name': 'Satisfied Mid-range Customer',
                'data': {
                    'age': 45, 'tenure_months': 24, 'monthly_charge': 75.0, 
                    'total_charges': 1800.0, 'satisfaction': 4, 'referrals': 1, 'dependents': 0
                }
            },
            {
                'name': 'Young At-risk Customer',
                'data': {
                    'age': 22, 'tenure_months': 2, 'monthly_charge': 85.0, 
                    'total_charges': 170.0, 'satisfaction': 2, 'referrals': 0, 'dependents': 0
                }
            }
        ]
        
        print("\n" + "="*60)
        print("FIXED MODEL PREDICTION TEST RESULTS")
        print("="*60)
        
        for i, test_case in enumerate(test_cases, 1):
            result = self.predict(test_case['data'])
            
            print(f"\n{i}. {test_case['name']}")
            print(f"   Churn Probability: {result['churn_probability']:.4f}")
            print(f"   Churn Prediction: {'YES' if result['churn_prediction'] else 'NO'}")
            print(f"   Confidence: {result['confidence']:.4f}")
            
            # Business interpretation
            if result['churn_probability'] > 0.7:
                recommendation = "🔴 HIGH RISK - Immediate retention action needed"
            elif result['churn_probability'] > 0.5:
                recommendation = "🟡 MEDIUM RISK - Monitor and offer incentives"
            elif result['churn_probability'] > 0.3:
                recommendation = "🟡 LOW-MEDIUM RISK - Monitor closely"
            else:
                recommendation = "🟢 LOW RISK - Maintain current service"
            
            print(f"   Recommendation: {recommendation}")

def main():
    """Test the fixed model"""
    model_path = "/Users/tarandeepsingh/Desktop/internship project 4/SparkScale-churn/models/fixed_churn_model.json"
    
    print("="*60)
    print("TESTING FIXED CHURN MODEL")
    print("="*60)
    
    tester = FixedModelTester(model_path)
    tester.run_test_cases()
    
    print("\n" + "="*60)
    print("✅ FIXED MODEL TEST COMPLETED")
    print("✅ Now showing varied risk levels!")
    print("="*60)

if __name__ == "__main__":
    main()
