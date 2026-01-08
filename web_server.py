"""
Flask Web Server for Facial Recognition with Live Feed
Supports both regular webcam and Raspberry Pi camera
Uses the FaceRecognition class from facial_recognition_fixed.py
Optimized with multithreading for concurrent request handling
"""
from flask import Flask, render_template, Response, jsonify, request
import cv2
import sys
import time
import threading
from queue import Queue, Empty
from collections import deque

# Import the FaceRecognition class from facial_recognition_fixed.py
from facial_recognition_fixed import FaceRecognition

# Try to import picamera for Raspberry Pi support
try:
    from picamera2 import Picamera2
    RASPBERRY_PI = True
except ImportError:
    RASPBERRY_PI = False


class FrameBuffer:
    """Thread-safe circular buffer for frames with lock-free reads"""
    def __init__(self, maxsize=2):
        self.buffer = deque(maxlen=maxsize)
        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_processed = None
    
    def put(self, frame):
        """Add frame to buffer (non-blocking)"""
        with self.lock:
            self.buffer.append(frame)
            self.latest_frame = frame
    
    def get_latest(self):
        """Get latest frame (non-blocking)"""
        return self.latest_frame
    
    def set_processed(self, frame):
        """Store latest processed frame"""
        with self.lock:
            self.latest_processed = frame
    
    def get_processed(self):
        """Get latest processed frame"""
        return self.latest_processed


class WebCamera:
    """
    Camera wrapper that works with USB webcam, Raspberry Pi camera, or network stream
    Integrates with FaceRecognition class for face detection
    Optimized with separate threads for capture and processing
    """
    def __init__(self, face_recognition_instance, use_pi_camera=False, network_stream_url=None, stream_token=None):
        self.face_recognition = face_recognition_instance
        self.use_pi_camera = use_pi_camera and RASPBERRY_PI
        self.network_stream_url = network_stream_url
        self.stream_token = stream_token
        
        # Thread-safe frame buffer
        self.frame_buffer = FrameBuffer(maxsize=2)
        self.running = False
        self.capture_thread = None
        self.process_thread = None
        self.fps = 0
        self.last_fps_time = time.time()
        self.frame_count = 0
        
        # Add token to stream URL if provided
        if self.network_stream_url and self.stream_token:
            separator = '&' if '?' in self.network_stream_url else '?'
            self.network_stream_url = f"{self.network_stream_url}{separator}token={self.stream_token}"
        
        # Initialize camera
        if self.network_stream_url:
            print(f"Connecting to network stream: {self.network_stream_url.split('?')[0]}...")  # Don't print token
            self.video_capture = cv2.VideoCapture(self.network_stream_url)
            self.picam = None
        elif self.use_pi_camera:
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
        
        # Start background threads
        self.start_threads()
    
    def start_threads(self):
        """Start capture and processing threads"""
        self.running = True
        
        # Thread 1: Capture frames from camera (fast)
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        
        # Thread 2: Process frames with face recognition (slower)
        self.process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.process_thread.start()
        
        print("✓ Camera threads started (capture + processing)")
    
    def _capture_loop(self):
        """Background thread: continuously capture frames"""
        while self.running:
            frame = self._get_frame_internal()
            if frame is not None:
                self.frame_buffer.put(frame)
                self.frame_count += 1
                
                # Calculate FPS
                current_time = time.time()
                if current_time - self.last_fps_time >= 1.0:
                    self.fps = self.frame_count / (current_time - self.last_fps_time)
                    self.frame_count = 0
                    self.last_fps_time = current_time
            else:
                time.sleep(0.01)  # Avoid busy waiting on error
    
    def _process_loop(self):
        """Background thread: process frames with face recognition"""
        while self.running:
            frame = self.frame_buffer.get_latest()
            if frame is not None:
                # Process frame (this is the slow part)
                processed_frame = self.face_recognition.process_frame(frame.copy(), draw_annotations=True)
                self.frame_buffer.set_processed(processed_frame)
            else:
                time.sleep(0.01)
    
    def _get_frame_internal(self):
        """Internal method to capture frame from camera"""
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
    
    def get_frame(self):
        """Get latest processed frame (non-blocking)"""
        frame = self.frame_buffer.get_processed()
        if frame is None:
            # Fallback to raw frame if no processed frame available yet
            frame = self.frame_buffer.get_latest()
        return frame
    
    def generate_frames(self):
        """Generate frames for video streaming (non-blocking)"""
        last_frame = None
        
        while True:
            frame = self.get_frame()
            
            # Only send new frames (avoid duplicates)
            if frame is None or (last_frame is not None and id(frame) == id(last_frame)):
                time.sleep(0.01)  # Small delay to avoid busy waiting
                continue
            
            last_frame = frame
            
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ret:
                continue
            
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    def stop(self):
        """Stop background threads"""
        self.running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=2)
        if self.process_thread:
            self.process_thread.join(timeout=2)
    
    def __del__(self):
        self.stop()
        if hasattr(self, 'video_capture') and self.video_capture:
            self.video_capture.release()
        if hasattr(self, 'picam') and self.picam:
            self.picam.stop()


# Flask app with optimizations
app = Flask(__name__)

# Enable threading and optimizations
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching for live feed
app.config['THREADED'] = True

face_recognition = None
camera = None
detection_history = []  # Store detection history
MAX_HISTORY = 50  # Maximum history entries to keep
history_lock = threading.Lock()  # Thread-safe history access

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
    """API endpoint for detection results (thread-safe)"""
    global detection_history
    
    results = face_recognition.get_detection_results()
    
    # Add to history if there are faces detected
    if results['total_faces'] > 0:
        with history_lock:
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
    """API endpoint for detection history (thread-safe)"""
    with history_lock:
        return jsonify({
            'total_entries': len(detection_history),
            'history': list(reversed(detection_history))  # Most recent first
        })

@app.route('/api/history/clear', methods=['POST'])
def clear_history():
    """Clear detection history (thread-safe)"""
    global detection_history
    with history_lock:
        detection_history = []
    return jsonify({'status': 'success', 'message': 'History cleared'})

@app.route('/api/status')
def get_status():
    """API endpoint for system status"""
    return jsonify({
        'status': 'running',
        'camera_type': 'Raspberry Pi Camera' if camera.use_pi_camera else 'USB/Webcam',
        'camera_fps': round(camera.fps, 1),
        'known_faces': len(face_recognition.known_face_names),
        'faces_loaded': face_recognition.known_face_names,
        'threads_active': camera.running
    })

@app.route('/api/alerts')
def get_alerts():
    """API endpoint for security alerts"""
    unacknowledged_only = request.args.get('unacknowledged', 'false').lower() == 'true'
    alerts = face_recognition.get_alerts(unacknowledged_only=unacknowledged_only)
    return jsonify({
        'total_alerts': len(alerts),
        'alerts': alerts
    })

@app.route('/api/alerts/<int:alert_id>/acknowledge', methods=['POST'])
def acknowledge_alert(alert_id):
    """Acknowledge a specific alert"""
    face_recognition.acknowledge_alert(alert_id)
    return jsonify({'status': 'success', 'message': f'Alert {alert_id} acknowledged'})

@app.route('/api/alerts/clear', methods=['POST'])
def clear_alerts():
    """Clear all alerts"""
    face_recognition.clear_alerts()
    return jsonify({'status': 'success', 'message': 'All alerts cleared'})

if __name__ == '__main__':
    import os
    
    # Check command line arguments
    use_pi = '--pi' in sys.argv or (RASPBERRY_PI and '--no-pi' not in sys.argv)
    network_stream = None
    stream_token = None
    
    # Check for network stream URL
    for arg in sys.argv:
        if arg.startswith('--stream='):
            network_stream = arg.split('=', 1)[1]
            print(f"Using network stream: {network_stream}")
            break
    
    # Check environment variable for Docker
    if not network_stream:
        network_stream = os.environ.get('CAMERA_STREAM_URL')
        if network_stream:
            print(f"Using camera stream from environment: {network_stream}")
    
    # Get authentication token for network streams
    stream_token = os.environ.get('CAMERA_STREAM_TOKEN')
    if network_stream and stream_token:
        print("Using authenticated camera stream")
    elif network_stream and not stream_token:
        print("⚠️  WARNING: No CAMERA_STREAM_TOKEN set - stream may require authentication")
    
    # Get number of CPU cores for optimal thread count
    cpu_count = os.cpu_count() or 4
    # Use half of available cores for face processing, min 2, max 8
    max_workers = max(2, min(8, cpu_count // 2))
    
    # Initialize FaceRecognition instance with multithreading
    face_recognition = FaceRecognition(known_faces_dir='faces', max_workers=max_workers)
    
    # Initialize camera with FaceRecognition instance
    camera = WebCamera(face_recognition, use_pi_camera=use_pi, network_stream_url=network_stream, stream_token=stream_token)
    
    print("\n" + "="*50)
    print("Facial Recognition Web Server (Optimized)")
    print("="*50)
    print(f"CPU Cores: {cpu_count} (using {max_workers} worker threads)")
    print(f"Architecture: Multi-threaded (capture + processing)")
    
    if camera.network_stream_url:
        print(f"Camera: Network Stream (authenticated)")
    elif camera.use_pi_camera:
        print(f"Camera: Raspberry Pi Camera")
    else:
        print(f"Camera: USB/Webcam")
    
    print(f"Known faces loaded: {len(face_recognition.known_face_names)}")
    print(f"Faces: {face_recognition.known_face_names}")
    print("\nOptimizations:")
    print("  ✓ Separate capture and processing threads")
    print("  ✓ Non-blocking frame buffer")
    print("  ✓ Thread-safe API endpoints")
    print("  ✓ Parallel face detection/encoding")
    print("  ✓ JPEG compression optimization")
    print("\nAccess the web interface at:")
    print("  http://localhost:5000")
    print("  http://0.0.0.0:5000  (for network access)")
    print("\nAPI Endpoints:")
    print("  GET /api/detections - Get current face detection results")
    print("  GET /api/status - Get system status (including FPS)")
    print("  GET /api/history - Get detection history")
    print("  GET /api/alerts - Get security alerts")
    print("\nPress Ctrl+C to stop")
    print("="*50 + "\n")
    
    try:
        # Use waitress for production-grade WSGI server (if available)
        try:
            from waitress import serve
            print("Using Waitress production server")
            serve(app, host='0.0.0.0', port=5000, threads=max_workers * 2)
        except ImportError:
            # Fallback to Flask development server with threading
            print("Using Flask development server (install waitress for production)")
            app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    finally:
        camera.stop()
        face_recognition.cleanup()
        print("Cleaned up resources")
