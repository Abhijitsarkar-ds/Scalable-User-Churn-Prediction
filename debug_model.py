"""
Debug the model to understand why all predictions are 0.0000
"""

import json
import math

def debug_model():
    """Debug model predictions"""
    
    # Load model
    model_path = "/Users/tarandeepsingh/Desktop/internship project 4/SparkScale-churn/models/simple_churn_model.json"
    
    with open(model_path, 'r') as f:
        model = json.load(f)
    
    print("MODEL DEBUG INFO:")
    print(f"Number of weights: {len(model['weights'])}")
    print(f"Bias: {model['bias']}")
    print(f"First 10 weights: {model['weights'][:10]}")
    print(f"Weight range: {min(model['weights'])} to {max(model['weights'])}")
    
    # Test with simple features
    test_features = [35, 48, 45, 2160, 5] + [0] * 40  # Pad to 45 features
    
    print(f"\nTEST FEATURES:")
    print(f"Length: {len(test_features)}")
    print(f"First 5: {test_features[:5]}")
    
    # Calculate linear output
    linear_output = sum(w * x for w, x in zip(model['weights'], test_features)) + model['bias']
    
    print(f"\nCALCULATION:")
    print(f"Linear output: {linear_output}")
    print(f"This is a HUGE number - causing sigmoid saturation!")
    
    # Test sigmoid
    def safe_sigmoid(x):
        if x > 500:
            return 1.0
        elif x < -500:
            return 0.0
        else:
            return 1 / (1 + math.exp(-x))
    
    sigmoid_result = safe_sigmoid(linear_output)
    print(f"Sigmoid result: {sigmoid_result}")
    
    # Test with normalized weights
    print(f"\nTESTING WITH NORMALIZED WEIGHTS:")
    normalized_weights = [w / 1000000 for w in model['weights']]  # Divide by 1M
    normalized_bias = model['bias'] / 1000000
    
    linear_output_norm = sum(w * x for w, x in zip(normalized_weights, test_features)) + normalized_bias
    sigmoid_result_norm = safe_sigmoid(linear_output_norm)
    
    print(f"Normalized linear output: {linear_output_norm}")
    print(f"Normalized sigmoid result: {sigmoid_result_norm}")
    
    return linear_output, sigmoid_result

if __name__ == "__main__":
    debug_model()
