"""
Registration API Routes
Handles user registration and face enrollment
"""
from flask import Blueprint, jsonify, request, current_app
import face_recognition
import numpy as np
import base64
import io
from PIL import Image

registration_bp = Blueprint('registration', __name__)

# Shared state (will be initialized by main app)
face_recognition_instance = None
db = None

def init_registration(face_recognition_inst, database):
    """Initialize with face recognition instance and database"""
    global face_recognition_instance, db
    face_recognition_instance = face_recognition_inst
    db = database

@registration_bp.route('/register', methods=['POST'])
def register_user():
    """Register a new user with face data"""
    try:
        from models import User, FaceProfile
        
        data = request.get_json()
        name = data.get('name')
        image_data = data.get('image')
        
        if not name or not image_data:
            return jsonify({'status': 'error', 'message': 'Name and image are required'}), 400
        
        # Decode base64 image
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        image_array = np.array(image)
        
        # Generate face encoding
        encodings = face_recognition.face_encodings(image_array, model="small")
        
        if not encodings:
            return jsonify({'status': 'error', 'message': 'No face detected in image'}), 400
        
        # Create user
        user = User(name=name, active=True)
        db.session.add(user)
        db.session.flush()
        
        # Create face profile
        encoding_binary = encodings[0].tobytes()
        face_profile = FaceProfile(
            user_id=user.id,
            label=name,
            image_path=None,
            face_encoding=encoding_binary
        )
        db.session.add(face_profile)
        db.session.commit()
        
        # Add to known faces in memory
        if face_recognition_instance:
            face_recognition_instance.known_face_encodings.append(encodings[0])
            face_recognition_instance.known_face_names.append(name)
        
        return jsonify({
            'status': 'success',
            'message': f'User {name} registered successfully',
            'user_id': user.id
        }), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'status': 'error', 'message': str(e)}), 500

@registration_bp.route('/users', methods=['GET'])
def list_users():
    """List all registered users"""
    try:
        from models import User, FaceProfile
        
        users = User.query.all()
        user_list = []
        
        for user in users:
            face_profiles = FaceProfile.query.filter_by(user_id=user.id).all()
            user_list.append({
                'id': user.id,
                'name': user.name,
                'active': user.active,
                'face_profiles': len(face_profiles)
            })
        
        return jsonify({'status': 'success', 'users': user_list}), 200
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@registration_bp.route('/users/<int:user_id>', methods=['GET'])
def get_user(user_id):
    """Get user details"""
    try:
        from models import User
        
        user = User.query.get(user_id)
        if not user:
            return jsonify({'status': 'error', 'message': 'User not found'}), 404
        
        return jsonify({
            'status': 'success',
            'user': {
                'id': user.id,
                'name': user.name,
                'active': user.active
            }
        }), 200
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@registration_bp.route('/users/<int:user_id>', methods=['DELETE'])
def delete_user(user_id):
    """Delete a user and their face profiles"""
    try:
        from models import User, FaceProfile
        
        user = User.query.get(user_id)
        if not user:
            return jsonify({'status': 'error', 'message': 'User not found'}), 404
        
        user_name = user.name
        
        # Delete face profiles (cascade should handle this, but being explicit)
        FaceProfile.query.filter_by(user_id=user_id).delete()
        
        # Delete user
        db.session.delete(user)
        db.session.commit()
        
        # Remove from known faces in memory
        if face_recognition_instance and user_name in face_recognition_instance.known_face_names:
            try:
                idx = face_recognition_instance.known_face_names.index(user_name)
                face_recognition_instance.known_face_names.pop(idx)
                face_recognition_instance.known_face_encodings.pop(idx)
            except ValueError:
                pass
        
        return jsonify({
            'status': 'success',
            'message': f'User {user_name} deleted successfully'
        }), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'status': 'error', 'message': str(e)}), 500
