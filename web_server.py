"""
Flask Web Server for Facial Recognition with Live Feed
Connects to camera stream URL (never accesses USB directly)
"""
from flask import Flask, render_template, Response, jsonify, request
import cv2
import sys
import time
import threading
from queue import Queue, Empty
from collections import deque
from dotenv import load_dotenv
import os

# Load .env configuration
load_dotenv()

# Import the FaceRecognition class
from facial_recognition_fixed import FaceRecognition

# Camera stream configuration from .env
# If CAMERA_STREAM_URL is set in .env, use it. Otherwise, default to localhost:1337
CAMERA_STREAM_URL = os.getenv('CAMERA_STREAM_URL', 'http://127.0.0.1:1337/camera/usb_0/stream')
CAMERA_STREAM_TOKEN = os.getenv('CAMERA_STREAM_TOKEN', '').strip('\'"')  # Strip quotes if present

print(f"\n📋 Camera Configuration from .env:")
print(f"   Stream URL: {CAMERA_STREAM_URL}")
print(f"   Token: {CAMERA_STREAM_TOKEN[:8]}...{CAMERA_STREAM_TOKEN[-4:] if CAMERA_STREAM_TOKEN else 'NOT SET'}")
print(f"   Full URL: {CAMERA_STREAM_URL}{'?' if '?' not in CAMERA_STREAM_URL else '&'}token={CAMERA_STREAM_TOKEN[:8]}..." if CAMERA_STREAM_TOKEN else f"   Full URL: {CAMERA_STREAM_URL} (NO AUTH)")


class FrameBuffer:
    """Thread-safe circular buffer for frames"""
    def __init__(self, maxsize=2):
        self.buffer = deque(maxlen=maxsize)
        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_processed = None
    
    def put(self, frame):
        with self.lock:
            self.buffer.append(frame)
            self.latest_frame = frame
    
    def get_latest(self):
        return self.latest_frame
    
    def set_processed(self, frame):
        with self.lock:
            self.latest_processed = frame
    
    def get_processed(self):
        return self.latest_processed


class WebCamera:
    """
    Camera wrapper that connects to camera stream URL ONLY
    Never accesses USB camera directly - only web addresses
    Integrates with FaceRecognition for face detection
    """
    def __init__(self, face_recognition_instance):
        self.face_recognition = face_recognition_instance
        
        # Thread-safe frame buffer
        self.frame_buffer = FrameBuffer(maxsize=2)
        self.running = False
        self.capture_thread = None
        self.process_thread = None
        self.connection_thread = None
        self.fps = 0
        self.last_fps_time = time.time()
        self.frame_count = 0
        self.video_capture = None
        self.connected = False
        self.connection_error = None
        
        # Start connection in background (non-blocking)
        self.connection_thread = threading.Thread(target=self._connect_to_stream, daemon=True)
        self.connection_thread.start()
    
    def _connect_to_stream(self):
        """Connect to camera stream in background thread"""
        # Build stream URL with token
        stream_url = CAMERA_STREAM_URL
        if CAMERA_STREAM_TOKEN:
            separator = '&' if '?' in stream_url else '?'
            stream_url = f"{stream_url}{separator}token={CAMERA_STREAM_TOKEN}"
            print(f"\n🔗 Connecting to: {CAMERA_STREAM_URL}")
            print(f"   With token: {CAMERA_STREAM_TOKEN[:8]}...{CAMERA_STREAM_TOKEN[-4:]}")
            print(f"   Full URL: {stream_url[:50]}...")
        else:
            print(f"\n🔗 Connecting to: {stream_url} (no auth)")
        
        print(f"\n🔍 Debug info:")
        print(f"   Attempting cv2.VideoCapture({stream_url[:60]}...)")
        print(f"   OpenCV version: {cv2.__version__}")
        
        # Connect to camera stream - retry until success
        max_retries = 60
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                self.video_capture = cv2.VideoCapture(stream_url)
                print(f"   Attempt {attempt + 1}: VideoCapture created, isOpened={self.video_capture.isOpened()}")
                
                if self.video_capture.isOpened():
                    # Test read a frame
                    ret, frame = self.video_capture.read()
                    print(f"   Attempt {attempt + 1}: Frame read, ret={ret}, frame shape={frame.shape if ret else 'None'}")
                    
                    if ret and frame is not None:
                        print(f"✅ Connected successfully on attempt {attempt + 1}!")
                        print(f"   Camera found: usb_0 at localhost:1337")
                        self.connected = True
                        self.start_threads()
                        return
                    else:
                        print(f"   Attempt {attempt + 1}: Failed to read frame")
                        self.video_capture.release()
                        self.video_capture = None
                else:
                    print(f"   Attempt {attempt + 1}: VideoCapture not opened")
                
                if attempt == 0:
                    print(f"\n⚠️  First attempt failed - is camera_streamer.py running on port 1337?")
                    print(f"⚠️  Web server is running, will keep trying to connect...\n")
                
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                
            except Exception as e:
                print(f"   Attempt {attempt + 1}: Exception - {type(e).__name__}: {e}")
                if attempt == 0:
                    print(f"\n⚠️  Connection error: {e}")
                    print(f"⚠️  Make sure camera_streamer.py is running!\n")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
        
        self.connection_error = f"Could not connect to {CAMERA_STREAM_URL} after {max_retries} attempts"
        print(f"❌ {self.connection_error}")
    
    def start_threads(self):
        """Start capture and processing threads"""
        self.running = True
        
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        
        self.process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.process_thread.start()
        
        print("✅ Camera threads started")
    
    def _capture_loop(self):
        """Background thread: continuously capture frames from stream"""
        while self.running:
            ret, frame = self.video_capture.read()
            if ret and frame is not None:
                self.frame_buffer.put(frame)
                self.frame_count += 1
                
                # Calculate FPS
                current_time = time.time()
                if current_time - self.last_fps_time >= 1.0:
                    self.fps = self.frame_count / (current_time - self.last_fps_time)
                    self.frame_count = 0
                    self.last_fps_time = current_time
            else:
                time.sleep(0.01)
    
    def _process_loop(self):
        """Background thread: process frames with face recognition"""
        while self.running:
            frame = self.frame_buffer.get_latest()
            if frame is not None:
                processed_frame = self.face_recognition.process_frame(frame.copy(), draw_annotations=True)
                self.frame_buffer.set_processed(processed_frame)
            else:
                time.sleep(0.01)
    
    def get_frame(self):
        """Get latest processed frame"""
        if not self.connected:
            return None
        
        frame = self.frame_buffer.get_processed()
        if frame is None:
            frame = self.frame_buffer.get_latest()
        return frame
    
    def generate_frames(self):
        """Generate frames for video streaming"""
        last_frame = None
        
        while True:
            if not self.connected:
                # Show "connecting" message
                placeholder = self._create_placeholder_frame("Connecting to camera stream...")
                ret, buffer = cv2.imencode('.jpg', placeholder, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(0.5)
                continue
            
            frame = self.get_frame()
            
            if frame is None or (last_frame is not None and id(frame) == id(last_frame)):
                time.sleep(0.01)
                continue
            
            last_frame = frame
            
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ret:
                continue
            
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    def _create_placeholder_frame(self, text):
        """Create a placeholder frame with text"""
        import numpy as np
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, text, (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.8, (255, 255, 255), 2, cv2.LINE_AA)
        return frame
    
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


# Flask app
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['THREADED'] = True

face_recognition = None
camera = None
detection_history = []
MAX_HISTORY = 50
history_lock = threading.Lock()

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
        'camera_type': 'Network Stream',
        'camera_connected': camera.connected,
        'camera_fps': round(camera.fps, 1) if camera.connected else 0,
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
    cpu_count = os.cpu_count() or 4
    face_workers = max(2, min(8, cpu_count // 2))
    backend_threads = max(2, cpu_count // 2)
    
    print("\n" + "="*50)
    print("Facial Recognition Web Server")
    print("="*50)
    print(f"🚀 Starting Flask server first...")
    print(f"   Will connect to camera in background")
    print(f"   Access at: http://localhost:5000")
    print("="*50 + "\n")
    
    # Initialize FaceRecognition
    face_recognition = FaceRecognition(known_faces_dir='faces', max_workers=face_workers)
    
    # Initialize camera (connects in background, non-blocking)
    camera = WebCamera(face_recognition)
    
    print(f"CPU Cores: {cpu_count}")
    print(f"  Face processing: {face_workers} threads")
    print(f"  Backend: {backend_threads} threads")
    print(f"Known faces: {len(face_recognition.known_face_names)}")
    print(f"Faces: {face_recognition.known_face_names}")
    print(f"\n💡 Camera will connect in background while server runs")
    print("Press Ctrl+C to stop\n")
    
    try:
        try:
            from waitress import serve
            serve(app, host='0.0.0.0', port=5000, threads=backend_threads)
        except ImportError:
            app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    finally:
        camera.stop()
        face_recognition.cleanup()
        print("Cleaned up")
