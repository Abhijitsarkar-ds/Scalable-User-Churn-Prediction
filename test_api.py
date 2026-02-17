"""
Test the deployed Churn Prediction API
Verify all endpoints work correctly
"""

import requests
import json
import time

class APITester:
    """Test the churn prediction API"""
    
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
    
    def test_endpoint(self, endpoint, method="GET", data=None):
        """Test API endpoint"""
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method == "GET":
                response = requests.get(url, timeout=10)
            elif method == "POST":
                headers = {"Content-Type": "application/json"}
                response = requests.post(url, json=data, headers=headers, timeout=10)
            
            return {
                'endpoint': endpoint,
                'method': method,
                'status_code': response.status_code,
                'response': response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text,
                'success': response.status_code == 200
            }
        except Exception as e:
            return {
                'endpoint': endpoint,
                'method': method,
                'status_code': 'ERROR',
                'response': str(e),
                'success': False
            }
    
    def test_health_endpoints(self):
        """Test health and info endpoints"""
        print("🔍 Testing Health Endpoints...")
        
        endpoints = [
            {'path': '/', 'name': 'Home'},
            {'path': '/health', 'name': 'Health Check'},
            {'path': '/docs', 'name': 'Documentation'}
        ]
        
        for endpoint in endpoints:
            result = self.test_endpoint(endpoint['path'])
            status = "✅" if result['success'] else "❌"
            print(f"  {status} {endpoint['name']}: {result['status_code']}")
            
            if result['success'] and endpoint['name'] == 'Health Check':
                print(f"     Model Status: {result['response'].get('model_loaded', 'Unknown')}")
    
    def test_prediction_endpoint(self):
        """Test single prediction endpoint"""
        print("\n🎯 Testing Prediction Endpoint...")
        
        # Test case 1: Low risk customer
        test_customer_1 = {
            'age': 35,
            'tenure_months': 48,
            'monthly_charge': 45.0,
            'total_charges': 2160.0,
            'satisfaction': 5,
            'referrals': 2,
            'dependents': 1,
            'customer_id': 'test_low_risk_001'
        }
        
        result1 = self.test_endpoint('/predict', 'POST', test_customer_1)
        status1 = "✅" if result1['success'] else "❌"
        print(f"  {status1} Low Risk Customer: {result1['status_code']}")
        
        if result1['success']:
            pred = result1['response'].get('prediction', {})
            print(f"     Churn Probability: {pred.get('churn_probability', 'N/A')}")
            print(f"     Risk Level: {pred.get('risk_level', 'N/A')}")
            print(f"     Recommendation: {pred.get('recommendation', 'N/A')}")
        
        # Test case 2: High risk customer
        test_customer_2 = {
            'age': 22,
            'tenure_months': 2,
            'monthly_charge': 120.0,
            'total_charges': 240.0,
            'satisfaction': 1,
            'referrals': 0,
            'dependents': 0,
            'customer_id': 'test_high_risk_002'
        }
        
        result2 = self.test_endpoint('/predict', 'POST', test_customer_2)
        status2 = "✅" if result2['success'] else "❌"
        print(f"  {status2} High Risk Customer: {result2['status_code']}")
        
        if result2['success']:
            pred = result2['response'].get('prediction', {})
            print(f"     Churn Probability: {pred.get('churn_probability', 'N/A')}")
            print(f"     Risk Level: {pred.get('risk_level', 'N/A')}")
            print(f"     Recommendation: {pred.get('recommendation', 'N/A')}")
    
    def test_batch_prediction(self):
        """Test batch prediction endpoint"""
        print("\n📦 Testing Batch Prediction...")
        
        batch_data = {
            'customers': [
                {
                    'age': 45, 'tenure_months': 24, 'monthly_charge': 75.0,
                    'total_charges': 1800.0, 'satisfaction': 4, 'referrals': 1, 'dependents': 0,
                    'customer_id': 'batch_001'
                },
                {
                    'age': 28, 'tenure_months': 3, 'monthly_charge': 95.0,
                    'total_charges': 285.0, 'satisfaction': 2, 'referrals': 0, 'dependents': 0,
                    'customer_id': 'batch_002'
                },
                {
                    'age': 65, 'tenure_months': 6, 'monthly_charge': 110.0,
                    'total_charges': 660.0, 'satisfaction': 1, 'referrals': 0, 'dependents': 2,
                    'customer_id': 'batch_003'
                }
            ]
        }
        
        result = self.test_endpoint('/batch_predict', 'POST', batch_data)
        status = "✅" if result['success'] else "❌"
        print(f"  {status} Batch Prediction: {result['status_code']}")
        
        if result['success']:
            response = result['response']
            print(f"     Total Predictions: {response.get('total_predictions', 0)}")
            
            predictions = response.get('predictions', [])
            for i, pred in enumerate(predictions[:2]):  # Show first 2
                cust_id = pred.get('customer_id', f'customer_{i+1}')
                prob = pred.get('prediction', {}).get('churn_probability', 'N/A')
                risk = pred.get('prediction', {}).get('risk_level', 'N/A')
                print(f"     {cust_id}: Prob={prob}, Risk={risk}")
    
    def test_error_handling(self):
        """Test error handling"""
        print("\n⚠️  Testing Error Handling...")
        
        # Test missing fields
        incomplete_data = {'age': 35, 'tenure_months': 12}  # Missing required fields
        result = self.test_endpoint('/predict', 'POST', incomplete_data)
        status = "✅" if result['status_code'] == 400 else "❌"
        print(f"  {status} Missing Fields Error: {result['status_code']}")
        
        # Test invalid JSON
        try:
            response = requests.post(f"{self.base_url}/predict", 
                               data="invalid json", 
                               headers={"Content-Type": "application/json"}, 
                               timeout=5)
            status = "✅" if response.status_code == 400 else "❌"
            print(f"  {status} Invalid JSON Error: {response.status_code}")
        except:
            print("  ❌ Invalid JSON Error: Connection failed")
    
    def run_all_tests(self):
        """Run complete API test suite"""
        print("=" * 60)
        print("CHURN PREDICTION API TEST SUITE")
        print("=" * 60)
        print(f"Testing API at: {self.base_url}")
        
        # Test health endpoints
        self.test_health_endpoints()
        
        # Test prediction endpoints
        self.test_prediction_endpoint()
        self.test_batch_prediction()
        
        # Test error handling
        self.test_error_handling()
        
        print("\n" + "=" * 60)
        print("API TESTING COMPLETED!")
        print("=" * 60)

def main():
    """Main test function"""
    print("🚀 Starting API Tests...")
    print("⚠️  Make sure API is running: python deploy_churn_api.py")
    print("⏳ Waiting 3 seconds for API to start...")
    
    time.sleep(3)
    
    tester = APITester()
    tester.run_all_tests()

if __name__ == "__main__":
    main()
