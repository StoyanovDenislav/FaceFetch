"""
Status API Routes
Handles system status and health checks
"""
from flask import Blueprint, jsonify

status_bp = Blueprint('status', __name__)

# Shared state (will be initialized by main app)
face_recognition = None
camera = None

def init_status(face_recognition_instance, camera_instance):
    """Initialize with face recognition and camera instances"""
    global face_recognition, camera
    face_recognition = face_recognition_instance
    camera = camera_instance

@status_bp.route('/status')
def get_status():
    """API endpoint for system status"""
    camera_type = 'Network Stream' if camera.camera_source == 'network' else 'USB Camera' if camera.camera_source == 'usb' else 'Unknown'
    
    return jsonify({
        'status': 'running',
        'camera_type': camera_type,
        'camera_source': camera.camera_source,
        'camera_connected': camera.connected,
        'camera_fps': round(camera.fps, 1) if camera.connected else 0,
        'known_faces': len(face_recognition.known_face_names),
        'faces_loaded': face_recognition.known_face_names,
        'threads_active': camera.running
    })
