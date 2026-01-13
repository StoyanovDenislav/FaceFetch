"""
Universal Camera Streamer
Discovers all available cameras (USB, Raspberry Pi, etc.) and streams them over HTTP as MJPEG
Run this on your host machine to make cameras accessible to Docker containers or remote clients
"""
import cv2
import sys
import time
import os
import secrets
from flask import Flask, render_template, Response, jsonify, request, abort
from threading import Thread, Lock
from functools import wraps
import socket
from pathlib import Path
from dotenv import load_dotenv, set_key, find_dotenv

# USB camera support only (no Raspberry Pi)


def generate_and_save_token(env_path='.env'):
    """Generate a new token and save it to .env file"""
    token = secrets.token_urlsafe(32)
    env_file = Path(env_path)
    
    # Create .env if it doesn't exist
    if not env_file.exists():
        with open(env_file, 'w') as f:
            f.write('# Camera Streamer Configuration\n')
            f.write(f'CAMERA_STREAM_TOKEN={token}\n')
            f.write('CAMERA_STREAM_AUTH=true\n')
    else:
        # Update existing .env
        set_key(env_path, 'CAMERA_STREAM_TOKEN', token)
    
    print(f"✓ Generated new token and saved to {env_path}")
    return token


class CameraStream:
    """Manages a single camera stream"""
    def __init__(self, camera_id, camera_type="usb"):
        self.camera_id = camera_id
        self.camera_type = camera_type
        self.capture = None
        self.picam = None
        self.frame = None
        self.lock = Lock()
        self.running = False
        self.last_access = time.time()
        
    def start(self):
        """Initialize and start the camera"""
        try:
            print(f"Starting USB Camera {self.camera_id}...")
            self.capture = cv2.VideoCapture(self.camera_id)
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            # Test if camera is accessible
            if not self.capture.isOpened():
                raise Exception(f"Cannot open camera {self.camera_id}")
            
            ret, _ = self.capture.read()
            if not ret:
                raise Exception(f"Cannot read from camera {self.camera_id}")
            
            self.running = True
            Thread(target=self._update_frame, daemon=True).start()
            return True
            
        except Exception as e:
            print(f"Failed to start camera {self.camera_id}: {e}")
            self.cleanup()
            return False
    
    def _update_frame(self):
        """Continuously capture frames in background thread"""
        while self.running:
            try:
                if self.capture:
                    ret, frame = self.capture.read()
                    if not ret:
                        print(f"Failed to read from camera {self.camera_id}")
                        time.sleep(0.1)
                        continue
                else:
                    break
                
                with self.lock:
                    self.frame = frame
                    
            except Exception as e:
                print(f"Error capturing frame from camera {self.camera_id}: {e}")
                time.sleep(0.1)
            
            time.sleep(0.03)  # ~30 FPS
    
    def get_frame(self):
        """Get the latest frame"""
        self.last_access = time.time()
        with self.lock:
            return self.frame.copy() if self.frame is not None else None
    
    def generate_mjpeg(self):
        """Generate MJPEG stream"""
        while True:
            frame = self.get_frame()
            if frame is None:
                time.sleep(0.1)
                continue
            
            # Encode as JPEG
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ret:
                continue
                
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    def cleanup(self):
        """Release camera resources"""
        self.running = False
        if self.capture:
            self.capture.release()
        if self.picam:
            try:
                self.picam.stop()
            except:
                pass


class CameraDiscovery:
    """Discovers all available cameras"""
    
    @staticmethod
    def discover_usb_cameras(max_cameras=10):
        """Discover USB/webcams by trying to open them"""
        cameras = []
        print("Scanning for USB cameras...")
        
        for i in range(max_cameras):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    cameras.append({
                        'id': i,
                        'type': 'usb',
                        'name': f'USB Camera {i}'
                    })
                    print(f"  ✓ Found: USB Camera {i}")
            cap.release()
        
        return cameras
    
    @staticmethod
    def discover_all():
        """Discover all available USB cameras only"""
        cameras = []
        
        # Check for USB cameras
        usb_cameras = CameraDiscovery.discover_usb_cameras()
        cameras.extend(usb_cameras)
        
        return cameras


# Flask app for streaming
app = Flask(__name__)
camera_streams = {}

# Load .env file
load_dotenv()

# Security configuration
# If token doesn't exist in env, generate and save a new one
if not os.getenv('CAMERA_STREAM_TOKEN'):
    ACCESS_TOKEN = generate_and_save_token('.env')
    # Reload to get the new token
    load_dotenv(override=True)
else:
    ACCESS_TOKEN = os.getenv('CAMERA_STREAM_TOKEN')

REQUIRE_AUTH = os.getenv('CAMERA_STREAM_AUTH', 'true').lower() == 'true'

# URL configuration
CUSTOM_URL = os.getenv('CAMERA_STREAM_URL', '').strip()
FORCE_CUSTOM_URL = os.getenv('CAMERA_STREAM_FORCE_URL', 'false').lower() == 'true'


def get_base_url():
    """Get the base URL for camera streams"""
    if FORCE_CUSTOM_URL and CUSTOM_URL:
        return CUSTOM_URL.rstrip('/')
    elif CUSTOM_URL and not FORCE_CUSTOM_URL:
        # Custom URL provided but not forced, use it as fallback
        return CUSTOM_URL.rstrip('/')
    else:
        # Auto-detect local IP
        local_ip = get_local_ip()
        return f"http://{local_ip}:1337"


def require_token(f):
    """Decorator to require authentication token"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not REQUIRE_AUTH:
            return f(*args, **kwargs)
        
        # Check for token in header
        token = request.headers.get('X-Camera-Token')
        
        # Check for token in query parameter
        if not token:
            token = request.args.get('token')
        
        if not token or token != ACCESS_TOKEN:
            abort(403, description="Invalid or missing authentication token")
        
        return f(*args, **kwargs)
    return decorated_function


@app.route('/')
@require_token
def index():
    """List available camera streams"""
    cameras = []
    base_url = get_base_url()
    
    for cam_id, stream in camera_streams.items():
        if REQUIRE_AUTH:
            cameras.append({
                'id': cam_id,
                'type': stream.camera_type,
                'stream_url': f"{base_url}/camera/{cam_id}/stream?token={ACCESS_TOKEN}",
                'snapshot_url': f"{base_url}/camera/{cam_id}/snapshot?token={ACCESS_TOKEN}"
            })
        else:
            cameras.append({
                'id': cam_id,
                'type': stream.camera_type,
                'stream_url': f"{base_url}/camera/{cam_id}/stream",
                'snapshot_url': f"{base_url}/camera/{cam_id}/snapshot"
            })
    
    return jsonify({
        'cameras': cameras,
        'count': len(cameras),
        'auth_required': REQUIRE_AUTH
    })


@app.route('/camera/<camera_id>/stream')
@require_token
def camera_stream(camera_id):
    """Stream camera as MJPEG"""
    if camera_id not in camera_streams:
        return "Camera not found", 404
    
    return Response(
        camera_streams[camera_id].generate_mjpeg(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/camera/<camera_id>/snapshot')
@require_token
def camera_snapshot(camera_id):
    """Get a single frame snapshot"""
    if camera_id not in camera_streams:
        return "Camera not found", 404
    
    frame = camera_streams[camera_id].get_frame()
    if frame is None:
        return "No frame available", 503
    
    ret, buffer = cv2.imencode('.jpg', frame)
    return Response(buffer.tobytes(), mimetype='image/jpeg')


def get_local_ip():
    """Get the local IP address"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"


def main():
    """Main entry point"""
    print("="*60)
    print("Universal Camera Streamer")
    print("="*60)
    
    # Discover all cameras
    discovered_cameras = CameraDiscovery.discover_all()
    
    if not discovered_cameras:
        print("\n❌ No cameras found!")
        print("Make sure your camera is connected and not in use by another application.")
        sys.exit(1)
    
    print(f"\n✓ Found {len(discovered_cameras)} camera(s)")
    
    # Initialize camera streams
    for cam_info in discovered_cameras:
        cam_id = f"{cam_info['type']}_{cam_info['id']}"
        stream = CameraStream(cam_info['id'], cam_info['type'])
        
        if stream.start():
            camera_streams[cam_id] = stream
            print(f"✓ Started stream: {cam_info['name']} -> {cam_id}")
    
    if not camera_streams:
        print("\n❌ Failed to start any camera streams!")
        sys.exit(1)
    
    # Display access information
    base_url = get_base_url()
    local_ip = get_local_ip()
    
    print("\n" + "="*60)
    print("Camera Streamer is running!")
    print("="*60)
    print(f"\nLocal IP: {local_ip}")
    print(f"Port: 8080")
    print(f"Authentication: {'ENABLED' if REQUIRE_AUTH else 'DISABLED'}")
    
    if FORCE_CUSTOM_URL and CUSTOM_URL:
        print(f"URL Mode: FORCED CUSTOM URL")
        print(f"Custom URL: {CUSTOM_URL}")
    elif CUSTOM_URL:
        print(f"URL Mode: Custom URL (fallback)")
        print(f"Custom URL: {CUSTOM_URL}")
    else:
        print(f"URL Mode: Auto-detect")
    
    if REQUIRE_AUTH:
        print(f"\n⚠️  SECURITY TOKEN (keep this secret!):")
        print(f"  {ACCESS_TOKEN}")
        print(f"\nSet this in your environment:")
        print(f"  PowerShell: $env:CAMERA_STREAM_TOKEN=\"{ACCESS_TOKEN}\"")
        print(f"  Linux/Mac:  export CAMERA_STREAM_TOKEN=\"{ACCESS_TOKEN}\"")
    
    print(f"\nAvailable Streams:")
    
    for cam_id in camera_streams:
        print(f"\n  Camera: {cam_id}")
        if REQUIRE_AUTH:
            print(f"    Stream:   {base_url}/camera/{cam_id}/stream?token={ACCESS_TOKEN}")
            print(f"    Snapshot: {base_url}/camera/{cam_id}/snapshot?token={ACCESS_TOKEN}")
        else:
            print(f"    Stream:   {base_url}/camera/{cam_id}/stream")
            print(f"    Snapshot: {base_url}/camera/{cam_id}/snapshot")
    
    print(f"\nAPI Endpoint:")
    if REQUIRE_AUTH:
        print(f"  {base_url}/?token={ACCESS_TOKEN} - List all cameras (JSON)")
    else:
        print(f"  {base_url}/ - List all cameras (JSON)")
    
    if REQUIRE_AUTH:
        print(f"\nFor Docker containers:")
        print(f"  Set CAMERA_STREAM_TOKEN environment variable")
        if FORCE_CUSTOM_URL and CUSTOM_URL:
            print(f"  Stream URL: {base_url}/camera/{list(camera_streams.keys())[0]}/stream?token={ACCESS_TOKEN}")
        else:
            print(f"  Stream URL: http://host.docker.internal:1337/camera/{list(camera_streams.keys())[0]}/stream?token={ACCESS_TOKEN}")
    else:
        if FORCE_CUSTOM_URL and CUSTOM_URL:
            print(f"\nFor Docker containers, use: {base_url}")
        else:
            print(f"\nFor Docker containers, use: http://host.docker.internal:1337")
    
    print(f"\n⚠️  To disable authentication (NOT RECOMMENDED):")
    print(f"  Set environment variable: CAMERA_STREAM_AUTH=false")
    print(f"\nPress Ctrl+C to stop")
    print("="*60 + "\n")
    
    # Start Flask server on port 1337
    try:
        app.run(host='0.0.0.0', port=1337, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\nStopping camera streamer...")
    finally:
        for stream in camera_streams.values():
            stream.cleanup()
        print("Cleaned up all cameras")


if __name__ == "__main__":
    main()
