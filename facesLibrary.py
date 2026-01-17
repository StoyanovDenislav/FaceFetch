from web_server import create_app
from models import db, User, FaceProfile
import face_recognition
import numpy as np
import os

app = create_app()
with app.app_context():
    people_data = [
        {"name": "Denislav", "image_path": "D:\\FaceFetch\\FaceFetch\\faces\\denkata4.jpg"},
        {"name": "Kristian", "image_path": "D:\\FaceFetch\\FaceFetch\\faces\\stefcho.jpg"},
        
    ]
    
    # Encode images and store in db
    for person_data in people_data:
        name = person_data["name"]
        
        # Check if user already exists
        existing_user = User.query.filter_by(name=name).first()
        if existing_user:
            print(f"User {name} already exists, skipping...")
            continue
            
        # Create new user
        user = User(name=name, active=True)
        db.session.add(user)
        db.session.flush()  # Get user.id without committing
        
        image_path = person_data["image_path"]
        if os.path.exists(image_path):
            # Load image for face recognition
            print(f"Processing {name}...")
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image, model="small")
            
            if encodings:
                # Convert encoding to binary format
                encoding_binary = encodings[0].tobytes()
                
                face_profile = FaceProfile(
                    user_id=user.id,
                    label=name,
                    image_path=image_path,
                    face_encoding=encoding_binary
                )
                db.session.add(face_profile)
                db.session.commit()
                print(f"✓ Added {name}")
            else:
                print(f"⚠️  No face found in {image_path}")
        else:
            print(f"⚠️  Image not found: {image_path}")
    
    print("✅ Database population complete!")
