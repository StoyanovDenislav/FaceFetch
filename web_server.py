"""
Flask Web Server for Facial Recognition with Live Feed
Connects to camera stream URL (never accesses USB directly)
"""
from flask import Flask, render_template, Response, jsonify
import pymysql
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

# Import database from models
from models import db, User, FaceProfile, Session, DetectionEvent, Command

DB_USER = os.environ.get('DB_USER', 'root')
DB_PASSWORD = os.environ.get('DB_PASSWORD', '')
DB_HOST = os.environ.get('DB_HOST', 'localhost')
DB_PORT = int(os.environ.get('DB_PORT', 3306))
DB_NAME = os.environ.get('DB_NAME', 'facefetch')

def ensure_database_exists():
    """Ensure the database exists before connecting"""
    try:
        conn = pymysql.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            port=DB_PORT
        )
        conn.autocommit(True)
        cursor = conn.cursor()
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{DB_NAME}`")
        cursor.close()
        conn.close()
        print(f"✅ Database '{DB_NAME}' ensured.")
    except Exception as e:
        print(f"❌ Database connection error: {e}")
        raise

# Camera stream configuration from .env
CAMERA_SOURCE = os.getenv('CAMERA_SOURCE', 'network').lower()  # 'usb' or 'network'
CAMERA_STREAM_URL = os.getenv('CAMERA_STREAM_URL', 'http://127.0.0.1:1337/camera/usb_0/stream')
CAMERA_STREAM_TOKEN = os.getenv('CAMERA_STREAM_TOKEN', '').strip('\'"')  # Strip quotes if present

print(f"\n📋 Camera Configuration from .env:")
print(f"   Camera Source: {CAMERA_SOURCE.upper()}")
if CAMERA_SOURCE == 'network':
    print(f"   Stream URL: {CAMERA_STREAM_URL}")
    print(f"   Token: {CAMERA_STREAM_TOKEN[:8]}...{CAMERA_STREAM_TOKEN[-4:] if CAMERA_STREAM_TOKEN else 'NOT SET'}")
    print(f"   Full URL: {CAMERA_STREAM_URL}{'?' if '?' not in CAMERA_STREAM_URL else '&'}token={CAMERA_STREAM_TOKEN[:8]}..." if CAMERA_STREAM_TOKEN else f"   Full URL: {CAMERA_STREAM_URL} (NO AUTH)")
else:
    print(f"   USB Camera: Index 0 (Direct)")



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
    Camera wrapper that connects to camera stream URL with fallback to local USB camera
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
        self.camera_source = None  # 'network' or 'usb'
        self.paused = False  # For pausing detection during registration
        
        # Start connection in background (non-blocking)
        self.connection_thread = threading.Thread(target=self._connect_to_stream, daemon=True)
        self.connection_thread.start()
    
    def _connect_to_stream(self):
        """Connect to camera stream in background thread with USB fallback"""
        # Check if user wants to use USB camera directly
        if CAMERA_SOURCE == 'usb':
            print("\n📸 CAMERA_SOURCE set to 'usb' - connecting directly to USB camera...")
            if self._connect_to_usb():
                return
            else:
                self.connection_error = "Could not connect to USB camera"
                print(f"❌ {self.connection_error}")
                return
        
        # Try network stream first
        stream_url = CAMERA_STREAM_URL
        if CAMERA_STREAM_TOKEN:
            separator = '&' if '?' in stream_url else '?'
            stream_url = f"{stream_url}{separator}token={CAMERA_STREAM_TOKEN}"
            print(f"\n🔗 Connecting to network camera: {CAMERA_STREAM_URL}")
            print(f"   With token: {CAMERA_STREAM_TOKEN[:8]}...{CAMERA_STREAM_TOKEN[-4:]}")
            print(f"   Full URL: {stream_url[:50]}...")
        else:
            print(f"\n🔗 Connecting to network camera: {stream_url} (no auth)")
        
        print(f"\n🔍 Debug info:")
        print(f"   Attempting cv2.VideoCapture({stream_url[:60]}...)")
        print(f"   OpenCV version: {cv2.__version__}")
        
        # Try network camera - retry indefinitely until found
        print(f"\n🔄 Connecting to network camera... (Will retry until connected)")
        retry_delay = 2
        attempt = 0
        
        while True:
            attempt += 1
            try:
                self.video_capture = cv2.VideoCapture(stream_url)
                
                if self.video_capture.isOpened():
                    # Test read a frame
                    ret, frame = self.video_capture.read()
                    
                    if ret and frame is not None:
                        print(f"✅ Connected to network camera on attempt {attempt}!")
                        print(f"   Network camera: {CAMERA_STREAM_URL}")
                        self.connected = True
                        self.camera_source = 'network'
                        self.start_threads()
                        return
                    else:
                        print(f"   Attempt {attempt}: Failed to read frame")
                        self.video_capture.release()
                        self.video_capture = None
                else:
                    print(f"   Attempt {attempt}: VideoCapture not opened")
                
                if attempt == 1:
                    print(f"⚠️  Network camera not found yet. Keeping background connection valid. Retrying every {retry_delay}s...")
                
                # Sleep before retry
                time.sleep(retry_delay)
                
            except Exception as e:
                print(f"   Attempt {attempt}: Exception - {type(e).__name__}: {e}")
                time.sleep(retry_delay)
    
    def _connect_to_usb(self):
        """Try to connect to USB camera (with retries for release race conditions)"""
        print("🔍 Attempting to connect to USB camera (index 0)...")
        
        max_retries = 5
        for attempt in range(max_retries):
            try:
                self.video_capture = cv2.VideoCapture(0)
                
                if self.video_capture.isOpened():
                    # Test read a frame
                    ret, frame = self.video_capture.read()
                    
                    if ret and frame is not None:
                        print(f"✅ Connected to USB camera successfully!")
                        print(f"   Camera: USB Camera 0 (Direct)")
                        print(f"   Resolution: {frame.shape[1]}x{frame.shape[0]}")
                        self.connected = True
                        self.camera_source = 'usb'
                        self.start_threads()
                        return True
                    else:
                        print(f"   Attempt {attempt+1}: Failed to read frame from USB camera")
                        self.video_capture.release()
                        self.video_capture = None
                else:
                    print(f"   Attempt {attempt+1}: USB camera not opened")
                    
            except Exception as e:
                print(f"   Attempt {attempt+1}: USB camera error: {type(e).__name__}: {e}")
            
            # Wait before retrying (camera might be busy releasing from browser)
            if attempt < max_retries - 1:
                print(f"   ⏳ Waiting for camera release... ({attempt+1}/{max_retries})")
                time.sleep(0.5)
        
        print("❌ Failed to connect to USB camera after retries")
        return False
    
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
            if self.paused:
                # When paused, just pass through raw frames without processing
                frame = self.frame_buffer.get_latest()
                if frame is not None:
                    self.frame_buffer.set_processed(frame)
                time.sleep(0.05)
                continue
                
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
    
    def pause(self):
        """Pause face detection processing"""
        self.paused = True
        print("⏸️  Face detection paused")
    
    def resume(self):
        """Resume face detection processing"""
        self.paused = False
        print("▶️  Face detection resumed")
    
    def release_camera(self):
        """Release the camera so browser can access it"""
        print("📷 Releasing camera for browser access...")
        
        # Stop threads first
        self.running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=2)
        if self.process_thread:
            self.process_thread.join(timeout=2)
        
        # Release the video capture
        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None
        
        self.connected = False
        print("✅ Camera released - browser can now access it")
    
    def reconnect_camera(self):
        """Reconnect to the camera after browser releases it"""
        print("🔄 Reconnecting to camera...")
        
        # Reset state
        self.connected = False
        self.video_capture = None
        
        # Reconnect using the same logic as initial connection
        import threading
        self.connection_thread = threading.Thread(target=self._connect_to_stream, daemon=True)
        self.connection_thread.start()
        
        # Wait a bit for connection (non-blocking)
        import time
        time.sleep(2)
        
        if self.connected:
            print("✅ Camera reconnected successfully")
        else:
            print("⚠️  Camera reconnection in progress...")
    
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


# Initialize Flask app with database
def create_app():
    """Create and configure Flask application"""
    ensure_database_exists()
    
    app = Flask(__name__)
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', os.urandom(24))
    app.config['SQLALCHEMY_DATABASE_URI'] = f'mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
    app.config['THREADED'] = True
    
    # Initialize database
    db.init_app(app)
    
    with app.app_context():
        db.create_all()
        print("✅ Database tables created/verified")
    
    return app

# Create Flask app
app = create_app()

# Global instances
face_recognition = None
camera = None

# Register API blueprints
from api.routes.detections import detections_bp, init_detections
from api.routes.alerts import alerts_bp, init_alerts
from api.routes.status import status_bp, init_status
from api.routes.registration import registration_bp, init_registration

app.register_blueprint(detections_bp, url_prefix='/api')
app.register_blueprint(alerts_bp, url_prefix='/api')
app.register_blueprint(status_bp, url_prefix='/api')
app.register_blueprint(registration_bp, url_prefix='/api')

@app.route('/')
def index():
    """Home page with video feed"""
    return render_template('index.html')

@app.route('/register')
def register():
    """Registration page"""
    return render_template('register.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(camera.generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/control/enter_registration', methods=['POST'])
def enter_registration_mode():
    """Stop face recognition module and release camera for browser access"""
    global face_recognition
    try:
        if face_recognition:
            print("🛑 Stopping face recognition module for registration...")
            face_recognition.cleanup()
            face_recognition = None
        
        # Release camera so browser can access it
        camera.release_camera()
        
        return jsonify({
            'status': 'success',
            'message': 'Entered registration mode - camera released for browser'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/control/exit_registration', methods=['POST'])
def exit_registration_mode():
    """Reinstantiate face recognition module and reconnect camera (Full Start)"""
    global face_recognition
    try:
        print("🔄 Performing full re-initiation of facial recognition...")

        # 1. Instantiate new Face Recognition module first
        cpu_count = os.cpu_count() or 4
        face_workers = max(2, min(8, cpu_count // 2))
        
        face_recognition = FaceRecognition(app=app, db=db, max_workers=face_workers)
        
        # 2. Update API routes with new FR instance
        from api.routes.detections import init_detections
        from api.routes.alerts import init_alerts
        from api.routes.registration import init_registration
        from api.routes.status import init_status
        
        init_detections(face_recognition)
        init_alerts(face_recognition)
        init_registration(face_recognition, db)
        init_status(face_recognition, camera)
        
        # 3. Bind the new FR instance to the existing camera object
        if camera:
            camera.face_recognition = face_recognition
            print("✅ Camera rebound to new face recognition instance")
            
            # 4. NOW reconnect the camera (ensures FR is ready when frames start arriving)
            camera.reconnect_camera()
        
        print(f"✅ Full start complete. Loaded {len(face_recognition.known_face_names)} known faces")
        
        return jsonify({
            'status': 'success',
            'message': 'Full re-initiation complete',
            'faces_loaded': len(face_recognition.known_face_names),
            'faces': face_recognition.known_face_names
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/control/pause', methods=['POST'])
def pause_detection():
    """Pause face detection during registration (legacy - use enter_registration instead)"""
    camera.pause()
    return jsonify({'status': 'success', 'message': 'Face detection paused'})

@app.route('/api/control/resume', methods=['POST'])
def resume_detection():
    """Resume face detection after registration (legacy - use exit_registration instead)"""
    camera.resume()
    return jsonify({'status': 'success', 'message': 'Face detection resumed'})

@app.route('/api/control/reload', methods=['POST'])
def reload_faces():
    """Reload face recognition from database"""
    try:
        # Clear existing faces
        face_recognition.known_face_encodings = []
        face_recognition.known_face_names = []
        
        # Reload from database
        face_recognition.encode_faces(app=app, db=db)
        
        return jsonify({
            'status': 'success',
            'message': 'Face recognition reloaded',
            'faces_loaded': len(face_recognition.known_face_names),
            'faces': face_recognition.known_face_names
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


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
    
    # Initialize FaceRecognition with database access
    face_recognition = FaceRecognition(app=app, db=db, max_workers=face_workers)
    
    # Initialize camera (connects in background, non-blocking)
    camera = WebCamera(face_recognition)
    
    print(f"CPU Cores: {cpu_count}")
    print(f"  Face processing: {face_workers} threads")
    print(f"  Backend: {backend_threads} threads")
    print(f"Known faces: {len(face_recognition.known_face_names)}")
    print(f"Faces: {face_recognition.known_face_names}")
    print(f"\n💡 Camera will connect in background while server runs")
    print("Press Ctrl+C to stop\n")
    
    # Initialize API routes with shared instances
    init_detections(face_recognition)
    init_alerts(face_recognition)
    init_status(face_recognition, camera)
    init_registration(face_recognition, db)
    
    try:
        try:
            from waitress import serve
            serve(app, host='0.0.0.0', port=5000, threads=backend_threads)
        except ImportError:
            app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    finally:
        camera.stop()
        if face_recognition:
            face_recognition.cleanup()
        print("Cleaned up")