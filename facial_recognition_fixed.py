import face_recognition
import os, sys
import cv2
import numpy as np
import math
from scipy.spatial import distance as dist
import time

# Check for GPU support
try:
    import dlib
    GPU_AVAILABLE = dlib.DLIB_USE_CUDA
    if GPU_AVAILABLE:
        GPU_COUNT = dlib.cuda.get_num_devices()
        print(f"🚀 GPU Acceleration ENABLED - {GPU_COUNT} CUDA device(s) available")
    else:
        print("⚠️  GPU Acceleration DISABLED - Running on CPU")
except Exception as e:
    GPU_AVAILABLE = False
    print(f"⚠️  GPU check failed: {e}")

def face_confidence(face_distance, face_match_threshold=0.6):
    range = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + '%'
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + '%'

def eye_aspect_ratio(eye):
    """Calculate the eye aspect ratio to detect blinks"""
    # Compute the euclidean distances between the vertical eye landmarks
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # Compute the euclidean distance between the horizontal eye landmarks
    C = dist.euclidean(eye[0], eye[3])
    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear

def detect_photo_spoof(frame, face_location):
    """Enhanced photo detection using multiple methods"""
    top, right, bottom, left = face_location
    # Scale back to original frame size
    top *= 4
    right *= 4
    bottom *= 4
    left *= 4
    
    # Add padding
    h, w = frame.shape[:2]
    top = max(0, top - 20)
    bottom = min(h, bottom + 20)
    left = max(0, left - 20)
    right = min(w, right + 20)
    
    face_roi = frame[top:bottom, left:right]
    if face_roi.size == 0:
        return True  # If we can't get the ROI, assume spoof
    
    # Convert to grayscale
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    
    # Method 1: Laplacian variance (blurriness/sharpness)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Method 2: Check for screen patterns (Moiré effect)
    # Calculate FFT to detect screen refresh patterns
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    high_freq_energy = np.sum(magnitude_spectrum[magnitude_spectrum > np.percentile(magnitude_spectrum, 90)])
    
    # Method 3: Color diversity - photos/screens have limited color range
    hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
    color_std = np.std(hsv)
    
    # Scoring system
    spoof_score = 0
    
    # Photos/screens are often too sharp (high var) or too blurry (low var)
    if laplacian_var < 80 or laplacian_var > 1500:
        spoof_score += 1
    
    # Screens have unusual high-frequency patterns
    if high_freq_energy > 50000:
        spoof_score += 1
    
    # Photos have less color variation
    if color_std < 30:
        spoof_score += 1
    
    # If 2 or more indicators suggest spoof
    return spoof_score >= 2
    
class FaceRecognition:
        face_locations = []
        face_encodings = []
        face_names = []
        known_face_encodings = []
        known_face_names = []
        process_current_frame = True
        
        # Liveness detection parameters
        EYE_AR_THRESH = 0.25
        EYE_AR_CONSEC_FRAMES = 3
        BLINK_COUNTER = 0
        TOTAL_BLINKS = 0
        liveness_check = {}  # Track liveness for each face
        last_positions = {}  # Track face positions for motion detection
        
        def __init__(self, known_faces_dir='faces'):
            self.known_face_encodings = []
            self.known_face_names = []
            self.encode_faces(known_faces_dir)
            
        def encode_faces(self, known_faces_dir='faces'):
            for image in os.listdir(known_faces_dir):
                face_image = face_recognition.load_image_file(f'{known_faces_dir}/{image}')
                # Encode with model parameter explicitly set
                encodings = face_recognition.face_encodings(face_image, model="large")
                if encodings:
                    face_encoding = encodings[0]
                    self.known_face_encodings.append(face_encoding)
                    self.known_face_names.append(image)
                
            print(self.known_face_names)
            
        def get_detection_results(self):
            """
            Get current detection results for API/backend integration
            Returns a list of dictionaries with detection information
            """
            return {
                'timestamp': time.time(),
                'total_faces': len(self.detection_results) if hasattr(self, 'detection_results') else 0,
                'faces': self.detection_results if hasattr(self, 'detection_results') else []
            }
        
        def process_frame(self, frame, draw_annotations=True):
            """
            Process a single frame for face detection and recognition
            
            Args:
                frame: OpenCV frame (BGR format)
                draw_annotations: Whether to draw rectangles and labels on the frame
            
            Returns:
                frame: Processed frame with annotations (if draw_annotations=True)
            """
            if self.process_current_frame:
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
             
                self.face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
                
                # Get encodings
                self.face_encodings = []
                for face_location in self.face_locations:
                    encodings = face_recognition.face_encodings(rgb_small_frame, [face_location], model="large")
                    if encodings:
                        self.face_encodings.append(encodings[0])
             
                # Store detection results for API/backend
                self.detection_results = []
                self.face_names = []
                
                for idx, (face_encoding, face_location) in enumerate(zip(self.face_encodings, self.face_locations)):
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    
                    # Initialize result object
                    result = {
                        'face_id': idx,
                        'location': {
                            'top': face_location[0] * 4,
                            'right': face_location[1] * 4,
                            'bottom': face_location[2] * 4,
                            'left': face_location[3] * 4
                        },
                        'state': None,  # 'known', 'unknown', 'spoof', 'pending_verification'
                        'name': None,
                        'confidence': None,
                        'is_live': False
                    }
                    
                    # Photo detection - this is ABSOLUTE, if detected as photo, reject immediately
                    is_photo = detect_photo_spoof(frame, face_location)
                    
                    # If detected as photo/screen, reject immediately regardless of anything else
                    if is_photo:
                        result['state'] = 'spoof'
                        result['name'] = "PHOTO/SCREEN DETECTED!"
                        self.face_names.append(result['name'])
                        self.detection_results.append(result)
                        continue
                    
                    # Only proceed with motion check if NOT a photo
                    has_significant_motion = False
                    face_key = f"face_{idx}"
                    
                    if face_key in self.last_positions:
                        last_pos = self.last_positions[face_key]
                        movement = abs(face_location[0] - last_pos[0]) + abs(face_location[3] - last_pos[3])
                        if movement > 2:  # Lower threshold for more sensitivity
                            has_significant_motion = True
                    
                    self.last_positions[face_key] = face_location
                    
                    # Initialize tracking
                    if face_key not in self.liveness_check:
                        self.liveness_check[face_key] = {'frames': 0, 'motion_frames': 0}
                    
                    self.liveness_check[face_key]['frames'] += 1
                    if has_significant_motion:
                        self.liveness_check[face_key]['motion_frames'] += 1
                    
                    # Give a grace period of 60 frames (~2 seconds)
                    if self.liveness_check[face_key]['frames'] < 60:
                        is_live = True
                    else:
                        # After grace period, need to have shown motion in at least 10% of frames
                        motion_ratio = self.liveness_check[face_key]['motion_frames'] / self.liveness_check[face_key]['frames']
                        is_live = motion_ratio > 0.1
                    
                    result['is_live'] = is_live
                    
                    # Determine what to display (only if passed photo detection)
                    if not is_live:
                        result['state'] = 'pending_verification'
                        result['name'] = "Move naturally to verify"
                    elif len(self.known_face_encodings) > 0:
                        face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                        best_match_index = np.argmin(face_distances)
                    
                        if matches[best_match_index]:
                            result['state'] = 'known'
                            result['name'] = self.known_face_names[best_match_index]
                            confidence = face_confidence(face_distances[best_match_index])
                            result['confidence'] = confidence
                            display_name = f"{result['name']} ({confidence})"
                            result['name'] = display_name
                        else:
                            result['state'] = 'unknown'
                            result['name'] = "Unknown (Verified)"
                    else:
                        result['state'] = 'unknown'
                        result['name'] = "Unknown (Verified)"
                
                    self.face_names.append(result['name'])
                    self.detection_results.append(result)
                
            self.process_current_frame = not self.process_current_frame
            
            # Draw annotations if requested
            if draw_annotations:
                # Draw face rectangles and labels
                for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4
                    
                    # Use different color for spoofing detection
                    if "PHOTO" in name or "SCREEN" in name or "Move" in name:
                        color = (0, 0, 255)  # Red for suspicious/unverified
                    else:
                        color = (0, 255, 0)  # Green for verified live faces
                    
                    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
            
            return frame
            
        def run_recognition(self):
            """Run face recognition using local video capture"""
            video_capture = cv2.VideoCapture(0)
            
            if not video_capture.isOpened():
                sys.exit("Error: Could not open video.")
            
            while True:
                ret, frame = video_capture.read()
                if not ret:
                    break
                
                # Process frame using the new method
                frame = self.process_frame(frame, draw_annotations=True)
                
                cv2.imshow('Video', frame)
                if cv2.waitKey(1) == ord('q'):
                    break
                
            video_capture.release()
            cv2.destroyAllWindows()
            
if __name__ == '__main__':
    fr = FaceRecognition()
    fr.run_recognition()
