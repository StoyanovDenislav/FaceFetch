import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from web_server import create_app
from models import db, User, KnownFace, FaceProfile
import face_recognition
import numpy as np

from werkzeug.security import generate_password_hash

app = create_app()
with app.app_context():
    
    # 1. Create Default Admin User (Operator)
    admin_email = "admin@example.com"
    admin_user = User.query.filter_by(email=admin_email).first()
    
    if not admin_user:
        print("Creating default Admin user...")
        admin = User(
            first_name="Admin",
            last_name="User",
            email=admin_email,
            password_hash=generate_password_hash("admin123"),
            role="operator",
            company="FaceFetch Security"
        )
        db.session.add(admin)
        db.session.commit()
        print("✅ Created Admin user (admin@example.com / admin123)")
    else:
        print("ℹ️  Admin user already exists")

    # 2. Populate Known Faces
    people_data = [
        {"name": "Denislav", "image_path": "D:\\FaceFetch\\FaceFetch\\faces\\denkata4.jpg"},
        {"name": "Kristian", "image_path": "D:\\FaceFetch\\FaceFetch\\faces\\stefcho.jpg"},
    ]
    
    # Encode images and store in db
    for person_data in people_data:
        full_name = person_data["name"]
        image_path = person_data["image_path"]
        
        # Check if known face already exists by name
        existing_face = KnownFace.query.filter_by(name=full_name).first()
        
        if existing_face:
            print(f"KnownFace {full_name} already exists. Checking profile...")
            known_face = existing_face
        else:
            print(f"Creating KnownFace for {full_name}...")
            known_face = KnownFace(
                name=full_name,
                active=True
            )
            db.session.add(known_face)
            db.session.flush() # Get ID
        
        # Process Image and Create Profile
        if os.path.exists(image_path):
            print(f"Processing image for {full_name}...")
            try:
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image, model="small")
                
                if encodings:
                    # Check if this profile already exists for this face
                    # (Simple check: exact image path match or just skipping duplicate checks for simplicity and relying on user intence)
                    # Let's check if this known face has any profiles yet
                    existing_profile = FaceProfile.query.filter_by(known_face_id=known_face.id, image_path=image_path).first()
                    
                    if not existing_profile:
                         # Convert encoding to binary format
                        encoding_binary = encodings[0].tobytes()
                        
                        face_profile = FaceProfile(
                            known_face_id=known_face.id,
                            label=full_name,
                            image_path=image_path,
                            face_encoding=encoding_binary
                        )
                        db.session.add(face_profile)
                        db.session.commit()
                        print(f"  ✓ Added profile for {full_name}")
                    else:
                        print(f"  ℹ️  Profile with this image already exists for {full_name}")

                else:
                    print(f"  ⚠️  No face found in {image_path}")
            except Exception as e:
                print(f"  ❌ Error processing image {image_path}: {e}")
        else:
            print(f"  ⚠️  Image not found: {image_path}")
    
    print("✅ Database population complete!")
