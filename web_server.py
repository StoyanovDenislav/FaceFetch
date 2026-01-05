"""
Flask Web Server for Facial Recognition with Live Feed
Supports both regular webcam and Raspberry Pi camera
Uses the FaceRecognition class from facial_recognition_fixed.py
"""
from flask import Flask, render_template, Response, jsonify
import cv2
import sys
import time

# Import the FaceRecognition class from facial_recognition_fixed.py
from facial_recognition_fixed import FaceRecognition

# Try to import picamera for Raspberry Pi support
try:
    from picamera2 import Picamera2
    RASPBERRY_PI = True
except ImportError:
    RASPBERRY_PI = False


class WebCamera:
    """
    Camera wrapper that works with USB webcam or Raspberry Pi camera
    Integrates with FaceRecognition class for face detection
    """
    def __init__(self, face_recognition_instance, use_pi_camera=False):
        self.face_recognition = face_recognition_instance
        self.use_pi_camera = use_pi_camera and RASPBERRY_PI
        
        # Initialize camera
        if self.use_pi_camera:
            print("Initializing Raspberry Pi Camera...")
            self.picam = Picamera2()
            config = self.picam.create_preview_configuration(main={"size": (640, 480)})
            self.picam.configure(config)
            self.picam.start()
            time.sleep(2)
            self.video_capture = None
        else:
            print("Initializing USB/Webcam...")
            self.video_capture = cv2.VideoCapture(0)
            self.picam = None
    
    def get_frame(self):
        """Capture frame from camera"""
        if self.use_pi_camera:
            # Raspberry Pi camera
            frame = self.picam.capture_array()
            # Convert from RGB to BGR for OpenCV
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            # USB/Webcam
            ret, frame = self.video_capture.read()
            if not ret:
                return None
        return frame
    
    def generate_frames(self):
        """Generate frames for video streaming"""
        while True:
            frame = self.get_frame()
            if frame is None:
                break
            
            # Process frame using FaceRecognition class
            frame = self.face_recognition.process_frame(frame, draw_annotations=True)
            
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    def __del__(self):
        if hasattr(self, 'video_capture') and self.video_capture:
            self.video_capture.release()
        if hasattr(self, 'picam') and self.picam:
            self.picam.stop()


# Flask app
app = Flask(__name__)
face_recognition = None
camera = None
detection_history = []  # Store detection history
MAX_HISTORY = 50  # Maximum history entries to keep

@app.route('/')
def index():
    """Home page with video feed"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(camera.generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/detections')
def get_detections():
    """API endpoint for detection results"""
    global detection_history
    
    results = face_recognition.get_detection_results()
    
    # Add to history if there are faces detected
    if results['total_faces'] > 0:
        for face in results['faces']:
            # Create history entry
            history_entry = {
                'timestamp': results['timestamp'],
                'face_id': face['face_id'],
                'state': face['state'],
                'name': face['name'],
                'confidence': face['confidence'],
                'is_live': face['is_live']
            }
            
            # Add to history (avoid duplicates from same timestamp)
            if not detection_history or detection_history[-1]['timestamp'] != results['timestamp']:
                detection_history.append(history_entry)
        
        # Keep only last MAX_HISTORY entries
        if len(detection_history) > MAX_HISTORY:
            detection_history = detection_history[-MAX_HISTORY:]
    
    return jsonify(results)

@app.route('/api/history')
def get_history():
    """API endpoint for detection history"""
    return jsonify({
        'total_entries': len(detection_history),
        'history': list(reversed(detection_history))  # Most recent first
    })

@app.route('/api/history/clear', methods=['POST'])
def clear_history():
    """Clear detection history"""
    global detection_history
    detection_history = []
    return jsonify({'status': 'success', 'message': 'History cleared'})

@app.route('/api/status')
def get_status():
    """API endpoint for system status"""
    return jsonify({
        'status': 'running',
        'camera_type': 'Raspberry Pi Camera' if camera.use_pi_camera else 'USB/Webcam',
        'known_faces': len(face_recognition.known_face_names),
        'faces_loaded': face_recognition.known_face_names
    })

if __name__ == '__main__':
    # Check if we should use Pi camera
    use_pi = '--pi' in sys.argv or (RASPBERRY_PI and '--no-pi' not in sys.argv)
    
    # Initialize FaceRecognition instance
    face_recognition = FaceRecognition(known_faces_dir='faces')
    
    # Initialize camera with FaceRecognition instance
    camera = WebCamera(face_recognition, use_pi_camera=use_pi)
    
    print("\n" + "="*50)
    print("Facial Recognition Web Server")
    print("="*50)
    print(f"Camera: {'Raspberry Pi Camera' if camera.use_pi_camera else 'USB/Webcam'}")
    print(f"Known faces loaded: {len(face_recognition.known_face_names)}")
    print(f"Faces: {face_recognition.known_face_names}")
    print("\nAccess the web interface at:")
    print("  http://localhost:5000")
    print("  http://0.0.0.0:5000  (for network access)")
    print("\nAPI Endpoints:")
    print("  GET /api/detections - Get current face detection results")
    print("  GET /api/status - Get system status")
    print("\nPress Ctrl+C to stop")
    print("="*50 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
