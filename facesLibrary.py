from __init__ import create_app, db
from models import Person, Face
import face_recognition
import os
import base64

app = create_app()
with app.app_context():
    people_data = [
        {"name": "Denislav", "image_path": "path/to/denislav.jpg"},
        {"name": "Person2", "image_path": "path/to/person2.jpg"},
        {"name": "Person3", "image_path": "path/to/person3.jpg"},
        {"name": "Person4", "image_path": "path/to/person4.jpg"},
        {"name": "Person5", "image_path": "path/to/person5.jpg"},
        {"name": "Person6", "image_path": "path/to/person6.jpg"},
    ]
    
    # Encode images in b64 and store in db
    for person_data in people_data:
        person = Person(name=person_data["name"])
        db.session.add(person)
        db.session.commit()
        
        image_path = person_data["image_path"]
        if os.path.exists(image_path):
            # Read and encode image to base64
            with open(image_path, 'rb') as image_file:
                image_b64 = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Load image for face recognition
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            
            if encodings:
                face = Face(
                    person_id=person.id,
                    image_path=image_path,
                    image_b64=image_b64,  # Store base64 encoded image
                    face_encoding=encodings[0]
                )
                db.session.add(face)
                db.session.commit()
                print(f"Added {person_data['name']}")
            else:
                print(f"No face found in {image_path}")
        else:
            print(f"Image not found: {image_path}")