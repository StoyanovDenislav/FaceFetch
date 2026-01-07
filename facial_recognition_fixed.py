import face_recognition
import os, sys
import cv2
import numpy as np
import math
from scipy.spatial import distance as dist
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

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
    """Enhanced photo detection using multiple methods - balanced accuracy"""
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
        return (False, 0.0)  # Return tuple: (is_spoof, confidence_score)
    
    # Convert to grayscale and HSV
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
    
    # Method 1: Laplacian variance (blurriness/sharpness)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Method 2: Texture analysis using Local Binary Patterns
    # Real skin has more texture variation than photos
    texture_var = np.var(cv2.Canny(gray, 50, 150))
    
    # Method 3: Check for screen patterns and refresh artifacts
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    # Look for periodic patterns typical of screens
    high_freq_energy = np.sum(magnitude_spectrum[magnitude_spectrum > np.percentile(magnitude_spectrum, 95)])
    
    # Method 4: Color diversity - screens compress color range
    hsv_std = np.std(hsv, axis=(0, 1))
    color_diversity = np.mean(hsv_std)
    
    # Method 5: Brightness distribution - real faces have natural gradients
    brightness_std = np.std(gray)
    brightness_hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    brightness_entropy = -np.sum((brightness_hist + 1e-10) * np.log2(brightness_hist + 1e-10))
    
    # Method 6: Screen edge detection - phone screens have sharp boundaries
    edges = cv2.Canny(face_roi, 100, 200)
    edge_density = np.sum(edges > 0) / edges.size
    
    # Scoring system with weighted importance
    spoof_score = 0
    confidence_factors = []
    
    # Very blurry or unnaturally sharp
    if laplacian_var < 60:
        spoof_score += 2.5
        confidence_factors.append("very_blurry")
    elif laplacian_var > 2000:
        spoof_score += 1.5
        confidence_factors.append("too_sharp")
    
    # Lack of natural texture (strong indicator for photos)
    if texture_var < 500:
        spoof_score += 2.0
        confidence_factors.append("low_texture")
    
    # Screen patterns detected (strong for phone/monitor)
    if high_freq_energy > 60000:
        spoof_score += 2.5
        confidence_factors.append("screen_pattern")
    
    # Limited color diversity (photos/screens)
    if color_diversity < 25:
        spoof_score += 1.5
        confidence_factors.append("flat_color")
    
    # Unnatural brightness distribution
    if brightness_std < 30:
        spoof_score += 1.5
        confidence_factors.append("flat_brightness")
    
    # Low brightness complexity (printed photos)
    if brightness_entropy < 5.0:
        spoof_score += 1.0
        confidence_factors.append("low_entropy")
    
    # High edge density (phone screen borders, photo edges)
    if edge_density > 0.15:
        spoof_score += 1.5
        confidence_factors.append("sharp_edges")
    
    # Multi-factor detection: require strong evidence
    # Phones/screens typically trigger: screen_pattern + flat_color + sharp_edges
    # Printed photos typically trigger: very_blurry + low_texture + flat_brightness
    # Real faces with glare: might trigger flat_color but NOT screen_pattern or low_texture
    
    # Threshold: Need score >= 5.0 for definite spoof detection
    # Return both the boolean result and the confidence score
    is_spoof = spoof_score >= 5.0
    
    return (is_spoof, spoof_score)
    
def detect_3d_depth(frame, face_location, prev_frame=None, prev_location=None):
    """
    Detect if face is 3D (real) or 2D (photo/screen) using depth cues
    Returns: (is_3d, confidence_score)
    """
    top, right, bottom, left = face_location
    # Scale to original size
    top *= 4
    right *= 4
    bottom *= 4
    left *= 4
    
    h, w = frame.shape[:2]
    top = max(0, top - 10)
    bottom = min(h, bottom + 10)
    left = max(0, left - 10)
    right = min(w, right + 10)
    
    face_roi = frame[top:bottom, left:right]
    if face_roi.size == 0:
        return (False, 0.0)
    
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    depth_score = 0
    
    # Method 1: Lighting gradient analysis (3D faces have depth-based shadows)
    # Calculate horizontal and vertical gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    gradient_complexity = np.std(gradient_magnitude)
    
    # Real 3D faces have complex gradient patterns (nose, cheeks, etc.)
    if gradient_complexity > 15:
        depth_score += 2.0
    
    # Method 2: Nose bridge detection (center should be brighter/more prominent)
    roi_height, roi_width = gray.shape
    center_region = gray[roi_height//4:3*roi_height//4, 2*roi_width//5:3*roi_width//5]
    edge_regions = [
        gray[roi_height//4:3*roi_height//4, 0:roi_width//5],  # left
        gray[roi_height//4:3*roi_height//4, 4*roi_width//5:roi_width]  # right
    ]
    
    if center_region.size > 0:
        center_intensity = np.mean(center_region)
        edge_intensity = np.mean([np.mean(r) for r in edge_regions if r.size > 0])
        
        # 3D faces: center (nose area) typically different from edges
        if abs(center_intensity - edge_intensity) > 10:
            depth_score += 1.5
    
    # Method 3: Optical flow (if previous frame available)
    if prev_frame is not None and prev_location is not None:
        try:
            prev_top, prev_right, prev_bottom, prev_left = prev_location
            prev_top *= 4
            prev_right *= 4
            prev_bottom *= 4
            prev_left *= 4
            prev_top = max(0, prev_top - 10)
            prev_bottom = min(h, prev_bottom + 10)
            prev_left = max(0, prev_left - 10)
            prev_right = min(w, prev_right + 10)
            
            prev_roi = prev_frame[prev_top:prev_bottom, prev_left:prev_right]
            prev_gray = cv2.cvtColor(prev_roi, cv2.COLOR_BGR2GRAY)
            
            # Resize to match if needed
            if prev_gray.shape == gray.shape:
                # Calculate optical flow
                flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                
                # 3D faces show differential motion (parallax effect)
                # Different parts move at different rates due to depth
                flow_magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
                flow_variance = np.var(flow_magnitude)
                
                # High variance = different depths moving differently
                if flow_variance > 0.5:
                    depth_score += 2.0
        except:
            pass
    
    # Method 4: Contour depth analysis
    # Use bilateral filter to preserve edges while smoothing
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    edges = cv2.Canny(bilateral, 30, 100)
    
    # Analyze edge distribution - 3D faces have structured edges (nose, jawline, etc.)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 5:  # Multiple distinct contours suggest 3D structure
        depth_score += 1.0
    
    # Threshold: score >= 4.0 indicates 3D face
    is_3d = depth_score >= 3.5
    
    return (is_3d, depth_score)

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
        prev_frame = None  # Store previous frame for depth detection
        prev_locations = {}  # Store previous locations for depth detection
        
        # Face persistence - keep faces displayed
        face_history = {}  # Store face data with timestamps
        face_retention_time = 15.0  # Keep faces for 5 seconds after last detection
        
        # Alert/Alarm system
        alerts = []  # Store alert events
        alert_cooldown = {}  # Prevent spam alerts
        alert_cooldown_time = 5.0  # Seconds between alerts for same type
        
        def __init__(self, known_faces_dir='faces', max_workers=4):
            self.known_face_encodings = []
            self.known_face_names = []
            # Thread pool for parallel processing
            self.max_workers = max_workers
            self.executor = ThreadPoolExecutor(max_workers=max_workers)
            self.lock = threading.Lock()
            print(f"🔧 Initialized with {max_workers} worker threads for CPU optimization")
            self.encode_faces(known_faces_dir)
            
        def encode_faces(self, known_faces_dir='faces'):
            """Load and encode known faces in parallel"""
            if not os.path.exists(known_faces_dir):
                print(f"⚠️  Known faces directory '{known_faces_dir}' not found")
                return
                
            image_files = [f for f in os.listdir(known_faces_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            if not image_files:
                print(f"⚠️  No face images found in '{known_faces_dir}'")
                return
            
            print(f"📸 Loading {len(image_files)} known faces in parallel...")
            
            def load_and_encode(image_file):
                """Load and encode a single face image"""
                try:
                    face_image = face_recognition.load_image_file(f'{known_faces_dir}/{image_file}')
                    encodings = face_recognition.face_encodings(face_image, model="large")
                    if encodings:
                        return (encodings[0], image_file)
                    return None
                except Exception as e:
                    print(f"⚠️  Error loading {image_file}: {e}")
                    return None
            
            # Use thread pool to load faces in parallel
            futures = [self.executor.submit(load_and_encode, img) for img in image_files]
            
            for future in as_completed(futures):
                result = future.result()
                if result:
                    encoding, name = result
                    self.known_face_encodings.append(encoding)
                    self.known_face_names.append(name)
                
            print(f"✅ Loaded faces: {self.known_face_names}")
            
        def trigger_alert(self, alert_type, message, face_data=None):
            """
            Trigger an alert/alarm for security events
            
            Args:
                alert_type: 'spoof' or 'unknown'
                message: Alert message
                face_data: Optional face detection data
            """
            current_time = time.time()
            
            # Check cooldown to prevent spam
            if alert_type in self.alert_cooldown:
                if current_time - self.alert_cooldown[alert_type] < self.alert_cooldown_time:
                    return  # Still in cooldown
            
            # Update cooldown
            self.alert_cooldown[alert_type] = current_time
            
            # Create alert
            alert = {
                'type': alert_type,
                'message': message,
                'timestamp': current_time,
                'face_data': face_data,
                'acknowledged': False
            }
            
            with self.lock:
                self.alerts.append(alert)
                # Keep only last 50 alerts
                if len(self.alerts) > 50:
                    self.alerts = self.alerts[-50:]
            
            # Console notification
            print(f"🚨 ALERT [{alert_type.upper()}]: {message}")
            
        def get_alerts(self, unacknowledged_only=False):
            """Get all alerts or only unacknowledged ones"""
            with self.lock:
                if unacknowledged_only:
                    return [a for a in self.alerts if not a['acknowledged']]
                return list(self.alerts)
        
        def acknowledge_alert(self, alert_index):
            """Mark an alert as acknowledged"""
            with self.lock:
                if 0 <= alert_index < len(self.alerts):
                    self.alerts[alert_index]['acknowledged'] = True
        
        def clear_alerts(self):
            """Clear all alerts"""
            with self.lock:
                self.alerts = []
            
        def get_detection_results(self):
            """
            Get current detection results for API/backend integration
            Returns a list of dictionaries with detection information
            """
            return {
                'timestamp': time.time(),
                'total_faces': len(self.detection_results) if hasattr(self, 'detection_results') else 0,
                'faces': self.detection_results if hasattr(self, 'detection_results') else [],
                'alerts': self.get_alerts(unacknowledged_only=True)
            }
        
        def process_single_face(self, frame, face_encoding, face_location, idx):
            """Process a single face detection (for parallel execution)"""
            result = {
                'face_id': idx,
                'location': {
                    'top': face_location[0] * 4,
                    'right': face_location[1] * 4,
                    'bottom': face_location[2] * 4,
                    'left': face_location[3] * 4
                },
                'state': None,
                'name': None,
                'confidence': None,
                'is_live': False
            }
            
            face_key = f"face_{idx}"
            
            # Get previous frame data for depth detection
            prev_frame = self.prev_frame
            prev_location = self.prev_locations.get(face_key)
            
            # 3D Depth detection (primary check)
            is_3d, depth_confidence = detect_3d_depth(frame, face_location, prev_frame, prev_location)
            
            # Photo detection with confidence score (secondary check)
            is_photo, spoof_confidence = detect_photo_spoof(frame, face_location)
            
            # Combined decision logic - balanced approach
            # Very high photo confidence always rejects
            if spoof_confidence >= 7.0:
                result['state'] = 'spoof'
                result['name'] = "PHOTO/SCREEN DETECTED!"
                self.trigger_alert('spoof', f"Spoofing attempt detected (confidence: {spoof_confidence:.1f})", result)
                return result, result['name']
            
            # Photo detection triggered (>=5.0) - check depth for confirmation
            if is_photo:
                if not is_3d or depth_confidence < 4.0:
                    # Photo detected AND (not 3D OR weak 3D confidence)
                    result['state'] = 'spoof'
                    result['name'] = "PHOTO/SCREEN DETECTED!"
                    self.trigger_alert('spoof', f"Photo/screen spoofing detected", result)
                    return result, result['name']
                # else: Photo detected BUT strong 3D evidence - likely false positive, continue
            
            # Medium-high spoof score (4.0-5.0) with weak 3D evidence
            if spoof_confidence >= 4.0 and depth_confidence < 3.0:
                result['state'] = 'spoof'
                result['name'] = "PHOTO/SCREEN DETECTED!"
                self.trigger_alert('spoof', f"Possible spoofing detected", result)
                return result, result['name']
            
            # Motion-based liveness - now even more lenient since we have depth detection
            has_motion = False
            
            with self.lock:
                if face_key in self.last_positions:
                    last_pos = self.last_positions[face_key]
                    movement = abs(face_location[0] - last_pos[0]) + abs(face_location[3] - last_pos[3])
                    if movement > 1:
                        has_motion = True
                
                self.last_positions[face_key] = face_location
                
                if face_key not in self.liveness_check:
                    self.liveness_check[face_key] = {'frames': 0, 'motion_frames': 0}
                
                self.liveness_check[face_key]['frames'] += 1
                if has_motion:
                    self.liveness_check[face_key]['motion_frames'] += 1
                
                frames_seen = self.liveness_check[face_key]['frames']
                motion_frames = self.liveness_check[face_key]['motion_frames']
                
                # More lenient motion requirements with depth detection
                # If 3D detected, require minimal motion
                # If depth uncertain, require more motion evidence
                
                if is_3d:
                    # Clearly 3D - very lenient
                    is_live = frames_seen < 20 or motion_frames > 0
                elif depth_confidence > 2.5:
                    # Moderately confident 3D - some motion preferred
                    if frames_seen < 40:
                        is_live = True
                    else:
                        motion_ratio = motion_frames / frames_seen
                        is_live = motion_ratio > 0.03  # Only 3% motion needed
                else:
                    # Low 3D confidence - require more motion
                    if frames_seen < 60:
                        is_live = True
                    else:
                        motion_ratio = motion_frames / frames_seen
                        is_live = motion_ratio > 0.08  # 8% motion needed
            
            result['is_live'] = is_live
            
            # Determine what to display
            if not is_live:
                result['state'] = 'pending_verification'
                result['name'] = "Slight movement needed"
            elif len(self.known_face_encodings) > 0:
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
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
                    self.trigger_alert('unknown', f"Unknown person detected", result)
            else:
                result['state'] = 'unknown'
                result['name'] = "Unknown (Verified)"
                self.trigger_alert('unknown', f"Unknown person detected (no known faces loaded)", result)
        
            return result, result['name']
        
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
             
                # Face detection
                self.face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
                
                # Store current frame for next iteration's depth detection
                if self.face_locations:
                    with self.lock:
                        self.prev_frame = frame.copy()
                
                # Get encodings in parallel if we have multiple faces
                if len(self.face_locations) > 1:
                    def get_encoding(face_loc):
                        encodings = face_recognition.face_encodings(rgb_small_frame, [face_loc], model="large")
                        return encodings[0] if encodings else None
                    
                    # Parallel encoding
                    encoding_futures = [self.executor.submit(get_encoding, loc) for loc in self.face_locations]
                    self.face_encodings = [f.result() for f in encoding_futures if f.result() is not None]
                else:
                    # Single face - no need for parallel processing overhead
                    self.face_encodings = []
                    for face_location in self.face_locations:
                        encodings = face_recognition.face_encodings(rgb_small_frame, [face_location], model="large")
                        if encodings:
                            self.face_encodings.append(encodings[0])
             
                # Store detection results for API/backend
                self.detection_results = []
                self.face_names = []
                
                # Update previous locations for next frame
                temp_prev_locations = {}
                for idx, face_location in enumerate(self.face_locations):
                    temp_prev_locations[f"face_{idx}"] = face_location
                
                # Process faces in parallel if we have multiple faces
                if len(self.face_encodings) > 1:
                    futures = []
                    for idx, (face_encoding, face_location) in enumerate(zip(self.face_encodings, self.face_locations)):
                        future = self.executor.submit(self.process_single_face, frame, face_encoding, face_location, idx)
                        futures.append(future)
                    
                    # Collect results
                    for future in as_completed(futures):
                        result, name = future.result()
                        self.detection_results.append(result)
                        self.face_names.append(name)
                else:
                    # Single face - process directly
                    for idx, (face_encoding, face_location) in enumerate(zip(self.face_encodings, self.face_locations)):
                        result, name = self.process_single_face(frame, face_encoding, face_location, idx)
                        self.detection_results.append(result)
                        self.face_names.append(name)
                
                # Update previous locations
                with self.lock:
                    self.prev_locations = temp_prev_locations
                    
                    # Update face history - keep faces for persistence
                    current_time = time.time()
                    
                    # Add/update current detections
                    for idx, (face_location, name) in enumerate(zip(self.face_locations, self.face_names)):
                        face_key = f"persistent_face_{idx}"
                        self.face_history[face_key] = {
                            'location': face_location,
                            'name': name,
                            'timestamp': current_time
                        }
                    
                    # Clean up old faces (not seen for retention_time seconds)
                    keys_to_remove = []
                    for key, data in self.face_history.items():
                        if current_time - data['timestamp'] > self.face_retention_time:
                            keys_to_remove.append(key)
                    
                    for key in keys_to_remove:
                        del self.face_history[key]
                
            self.process_current_frame = not self.process_current_frame
            
            # Draw annotations if requested
            if draw_annotations:
                # Draw all faces from history (persistent display)
                current_time = time.time()
                
                with self.lock:
                    for face_data in self.face_history.values():
                        top, right, bottom, left = face_data['location']
                        name = face_data['name']
                        age = current_time - face_data['timestamp']
                        
                        # Scale up coordinates
                        top *= 4
                        right *= 4
                        bottom *= 4
                        left *= 4
                        
                        # Fade effect based on age (optional - reduces opacity for old detections)
                        # For now, just use solid colors
                        
                        # Use different color for spoofing detection
                        if "PHOTO" in name or "SCREEN" in name or "Move" in name or "movement" in name:
                            color = (0, 0, 255)  # Red for suspicious/unverified
                        else:
                            color = (0, 255, 0)  # Green for verified live faces
                        
                        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                        font = cv2.FONT_HERSHEY_DUPLEX
                        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
            
            return frame
            
        def cleanup(self):
            """Clean up thread pool resources"""
            self.executor.shutdown(wait=True)
            
        def run_recognition(self):
            """Run face recognition using local video capture"""
            video_capture = cv2.VideoCapture(0)
            
            if not video_capture.isOpened():
                sys.exit("Error: Could not open video.")
            
            # Performance monitoring
            frame_count = 0
            start_time = time.time()
            fps_display = 0
            
            try:
                while True:
                    ret, frame = video_capture.read()
                    if not ret:
                        break
                    
                    # Process frame using the new method
                    frame = self.process_frame(frame, draw_annotations=True)
                    
                    # Calculate and display FPS
                    frame_count += 1
                    if frame_count % 30 == 0:
                        elapsed = time.time() - start_time
                        fps_display = frame_count / elapsed
                    
                    # Display FPS on frame
                    cv2.putText(frame, f"FPS: {fps_display:.1f}", (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    cv2.imshow('Video', frame)
                    if cv2.waitKey(1) == ord('q'):
                        break
            finally:
                video_capture.release()
                cv2.destroyAllWindows()
                self.cleanup()
                print(f"Average FPS: {frame_count / (time.time() - start_time):.2f}")
            
if __name__ == '__main__':
    fr = FaceRecognition()
    fr.run_recognition()
