"""
Registration API Routes
Handles user registration and face enrollment
"""
from flask import Blueprint, jsonify, request

registration_bp = Blueprint('registration', __name__)

# Shared state (will be initialized by main app)
face_recognition = None
db = None

def init_registration(face_recognition_instance, database):
    """Initialize with face recognition instance and database"""
    global face_recognition, db
    face_recognition = face_recognition_instance
    db = database

@registration_bp.route('/register', methods=['POST'])
def register_user():
    """Register a new user with face data"""
    # TODO: Implement user registration
    # - Accept user info (name, email, etc.)
    # - Accept face image(s)
    # - Generate face encodings
    # - Store in database
    # - Add to known faces
    return jsonify({'status': 'error', 'message': 'Not implemented yet'}), 501

@registration_bp.route('/users', methods=['GET'])
def list_users():
    """List all registered users"""
    # TODO: Implement user listing from database
    return jsonify({'status': 'error', 'message': 'Not implemented yet'}), 501

@registration_bp.route('/users/<int:user_id>', methods=['GET'])
def get_user(user_id):
    """Get user details"""
    # TODO: Implement user details retrieval
    return jsonify({'status': 'error', 'message': 'Not implemented yet'}), 501

@registration_bp.route('/users/<int:user_id>', methods=['DELETE'])
def delete_user(user_id):
    """Delete a user"""
    # TODO: Implement user deletion
    return jsonify({'status': 'error', 'message': 'Not implemented yet'}), 501
