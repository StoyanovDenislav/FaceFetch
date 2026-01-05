"""
Example of how to integrate facial recognition with a backend API
"""
import json
from facial_recognition_fixed import FaceRecognition

# Example: Get detection results and send to backend
def process_frame_for_api():
    """
    Example function showing how to use the detection results
    """
    fr = FaceRecognition()
    
    # In a real implementation, you would:
    # 1. Capture a frame from camera
    # 2. Process it with the facial recognition
    # 3. Get the structured results
    # 4. Send to your backend
    
    results = fr.get_detection_results()
    
    # Convert to JSON for API
    json_results = json.dumps(results, indent=2)
    print("Detection Results (JSON format for API):")
    print(json_results)
    
    # Example of what you get:
    """
    {
      "timestamp": 1704484123.456,
      "total_faces": 2,
      "faces": [
        {
          "face_id": 0,
          "location": {
            "top": 120,
            "right": 340,
            "bottom": 280,
            "left": 180
          },
          "state": "known",
          "name": "denkata.png (95.5%)",
          "confidence": "95.5%",
          "is_live": true
        },
        {
          "face_id": 1,
          "location": {
            "top": 150,
            "right": 400,
            "bottom": 310,
            "left": 240
          },
          "state": "spoof",
          "name": "PHOTO/SCREEN DETECTED!",
          "confidence": null,
          "is_live": false
        }
      ]
    }
    """
    
    # You can now send this to your backend:
    # import requests
    # response = requests.post('http://your-backend.com/api/face-detection', json=results)
    
    return results


# State meanings:
# 'known' - Recognized person from your database
# 'unknown' - Live person but not in database
# 'spoof' - Photo/screen detected
# 'pending_verification' - Waiting for motion to confirm liveness

if __name__ == '__main__':
    print("This is an example of API integration.")
    print("The actual facial_recognition_fixed.py now exposes structured data.")
    print("\nExample structure:")
    example = {
        "timestamp": 1704484123.456,
        "total_faces": 1,
        "faces": [{
            "face_id": 0,
            "location": {"top": 120, "right": 340, "bottom": 280, "left": 180},
            "state": "known",  # or 'unknown', 'spoof', 'pending_verification'
            "name": "John Doe (95.5%)",
            "confidence": "95.5%",
            "is_live": True
        }]
    }
    print(json.dumps(example, indent=2))
