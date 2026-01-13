#!/usr/bin/env python3
"""
Test script for FaceFetch authentication system
Tests all authentication endpoints and workflows
"""

import requests
import json
import time
from urllib.parse import urljoin

class FaceFetchAuthTester:
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.test_results = []
        
    def log(self, test_name, passed, message=""):
        """Log test result"""
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
        if message:
            print(f"  → {message}")
        self.test_results.append((test_name, passed))
    
    def test_register(self, email="test_user@example.com", password="TestPassword123"):
        """Test user registration"""
        print("\n=== Testing Registration ===")
        
        data = {
            "firstName": "Test",
            "lastName": "User",
            "email": email,
            "password": password,
            "company": "Test Company"
        }
        
        try:
            response = self.session.post(
                urljoin(self.base_url, "/register"),
                data=data
            )
            
            if response.status_code == 201:
                self.log("Registration successful", True, "User created")
                return True
            elif response.status_code == 409:
                self.log("Registration with existing email", True, "Email already registered (expected if re-running)")
                return False
            else:
                self.log("Registration", False, f"Status: {response.status_code}, Response: {response.text}")
                return False
                
        except Exception as e:
            self.log("Registration", False, str(e))
            return False
    
    def test_login(self, email="test_user@example.com", password="TestPassword123"):
        """Test user login"""
        print("\n=== Testing Login ===")
        
        data = {
            "email": email,
            "password": password,
            "rememberMe": "true"
        }
        
        try:
            response = self.session.post(
                urljoin(self.base_url, "/login"),
                data=data,
                allow_redirects=True
            )
            
            if response.status_code == 200 or response.url.endswith('/'):
                self.log("Login successful", True, "Session established")
                return True
            elif response.status_code == 401:
                self.log("Login with invalid credentials", True, "Correctly rejected (expected if wrong password)")
                return False
            else:
                self.log("Login", False, f"Status: {response.status_code}")
                return False
                
        except Exception as e:
            self.log("Login", False, str(e))
            return False
    
    def test_auth_status(self):
        """Test authentication status endpoint"""
        print("\n=== Testing Auth Status ===")
        
        try:
            response = self.session.get(
                urljoin(self.base_url, "/api/auth/status")
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("authenticated"):
                    self.log("Auth status check (authenticated)", True, 
                           f"User: {data.get('user_name')} ({data.get('user_email')})")
                    return True
                else:
                    self.log("Auth status check (not authenticated)", True, 
                           "User is not logged in")
                    return False
            else:
                self.log("Auth status", False, f"Status: {response.status_code}")
                return False
                
        except Exception as e:
            self.log("Auth status", False, str(e))
            return False
    
    def test_user_profile(self):
        """Test user profile endpoint"""
        print("\n=== Testing User Profile ===")
        
        try:
            response = self.session.get(
                urljoin(self.base_url, "/api/user/profile")
            )
            
            if response.status_code == 200:
                data = response.json()
                self.log("User profile retrieval", True,
                       f"Email: {data.get('email')}, Name: {data.get('first_name')} {data.get('last_name')}")
                return True
            elif response.status_code == 401:
                self.log("User profile (not authenticated)", True, "Correctly requires authentication")
                return False
            else:
                self.log("User profile", False, f"Status: {response.status_code}")
                return False
                
        except Exception as e:
            self.log("User profile", False, str(e))
            return False
    
    def test_api_endpoints(self):
        """Test protected API endpoints"""
        print("\n=== Testing API Endpoints ===")
        
        endpoints = [
            ("/api/status", "System status"),
            ("/api/history", "Detection history"),
            ("/api/detections", "Current detections"),
            ("/api/alerts", "Security alerts"),
        ]
        
        for endpoint, description in endpoints:
            try:
                response = self.session.get(
                    urljoin(self.base_url, endpoint)
                )
                
                if response.status_code == 200:
                    self.log(f"API: {description}", True)
                elif response.status_code == 401:
                    self.log(f"API: {description} (not authenticated)", True, "Correctly requires authentication")
                else:
                    self.log(f"API: {description}", False, f"Status: {response.status_code}")
                    
            except Exception as e:
                self.log(f"API: {description}", False, str(e))
    
    def test_logout(self):
        """Test logout"""
        print("\n=== Testing Logout ===")
        
        try:
            response = self.session.get(
                urljoin(self.base_url, "/logout"),
                allow_redirects=True
            )
            
            if response.status_code == 200:
                self.log("Logout", True, "Session cleared, redirected to login")
                return True
            else:
                self.log("Logout", False, f"Status: {response.status_code}")
                return False
                
        except Exception as e:
            self.log("Logout", False, str(e))
            return False
    
    def test_invalid_login(self):
        """Test login with invalid credentials"""
        print("\n=== Testing Invalid Login ===")
        
        data = {
            "email": "nonexistent@example.com",
            "password": "WrongPassword123"
        }
        
        try:
            response = self.session.post(
                urljoin(self.base_url, "/login"),
                data=data,
                allow_redirects=False
            )
            
            if response.status_code == 401:
                self.log("Invalid login rejection", True, "Correctly rejected")
                return True
            else:
                self.log("Invalid login rejection", False, f"Status: {response.status_code}")
                return False
                
        except Exception as e:
            self.log("Invalid login rejection", False, str(e))
            return False
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*50)
        print("TEST SUMMARY")
        print("="*50)
        
        passed = sum(1 for _, result in self.test_results if result)
        total = len(self.test_results)
        
        print(f"\nTests Passed: {passed}/{total}")
        print(f"Success Rate: {(passed/total)*100:.1f}%")
        
        if passed == total:
            print("\n✓ All tests passed!")
        else:
            print(f"\n✗ {total - passed} test(s) failed")
            print("\nFailed tests:")
            for test_name, result in self.test_results:
                if not result:
                    print(f"  - {test_name}")

def main():
    """Run all tests"""
    print("\n" + "="*50)
    print("FaceFetch Authentication System Test Suite")
    print("="*50)
    
    tester = FaceFetchAuthTester()
    
    # Test workflow
    print("\nAttempting to register new user...")
    registration_ok = tester.test_register()
    
    print("\nAttempting to login...")
    if tester.test_login():
        print("\nTesting authenticated endpoints...")
        tester.test_auth_status()
        tester.test_user_profile()
        tester.test_api_endpoints()
        
        print("\nTesting logout...")
        tester.test_logout()
    
    print("\nTesting invalid login...")
    tester.test_invalid_login()
    
    # Print summary
    tester.print_summary()

if __name__ == "__main__":
    main()
