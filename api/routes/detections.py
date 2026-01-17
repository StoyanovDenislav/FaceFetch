"""
Detection API Routes
Handles real-time face detection results and history
"""
from flask import Blueprint, jsonify, request
import threading

detections_bp = Blueprint('detections', __name__)

# Shared state (will be initialized by main app)
face_recognition = None
detection_history = []
MAX_HISTORY = 50
history_lock = threading.Lock()

def init_detections(face_recognition_instance):
    """Initialize with face recognition instance"""
    global face_recognition
    face_recognition = face_recognition_instance

@detections_bp.route('/detections')
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

@detections_bp.route('/history')
def get_history():
    """API endpoint for detection history (thread-safe)"""
    with history_lock:
        return jsonify({
            'total_entries': len(detection_history),
            'history': list(reversed(detection_history))  # Most recent first
        })

@detections_bp.route('/history/clear', methods=['POST'])
def clear_history():
    """Clear detection history (thread-safe)"""
    global detection_history
    with history_lock:
        detection_history = []
    return jsonify({'status': 'success', 'message': 'History cleared'})
