"""
Alert API Routes
Handles security alerts and acknowledgments
"""
from flask import Blueprint, jsonify, request

alerts_bp = Blueprint('alerts', __name__)

# Shared state (will be initialized by main app)
face_recognition = None

def init_alerts(face_recognition_instance):
    """Initialize with face recognition instance"""
    global face_recognition
    face_recognition = face_recognition_instance

@alerts_bp.route('/alerts')
def get_alerts():
    """API endpoint for security alerts"""
    unacknowledged_only = request.args.get('unacknowledged', 'false').lower() == 'true'
    alerts = face_recognition.get_alerts(unacknowledged_only=unacknowledged_only)
    return jsonify({
        'total_alerts': len(alerts),
        'alerts': alerts
    })

@alerts_bp.route('/alerts/<int:alert_id>/acknowledge', methods=['POST'])
def acknowledge_alert(alert_id):
    """Acknowledge a specific alert"""
    face_recognition.acknowledge_alert(alert_id)
    return jsonify({'status': 'success', 'message': f'Alert {alert_id} acknowledged'})

@alerts_bp.route('/alerts/clear', methods=['POST'])
def clear_alerts():
    """Clear all alerts"""
    face_recognition.clear_alerts()
    return jsonify({'status': 'success', 'message': 'All alerts cleared'})
