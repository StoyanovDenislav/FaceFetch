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

# Individual detection methods for parallel execution
def check_moire_pattern(face_roi):
    """Detect moiré patterns from screen pixels"""
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    # Apply FFT to detect periodic patterns
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    
    # Look for regular patterns in frequency domain
    # Screens create periodic patterns, real faces don't
    h, w = magnitude.shape
    center_h, center_w = h // 2, w // 2
    
    # Exclude DC component (center)
    mask = np.ones((h, w), dtype=bool)
    mask[center_h-5:center_h+5, center_w-5:center_w+5] = False
    
    # Check for high-frequency peaks (screen pixel grid)
    freq_peaks = magnitude[mask]
    peak_ratio = np.max(freq_peaks) / (np.mean(freq_peaks) + 1e-10)
    
    score = 0
    if peak_ratio > 50:  # Strong periodic pattern
        score = 3.0
    elif peak_ratio > 30:
        score = 2.0
    elif peak_ratio > 20:
        score = 1.0
    
    return score

def check_screen_refresh(face_roi):
    """Detect screen refresh artifacts and flicker"""
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    
    # Look for horizontal banding (screen refresh lines)
    horizontal_profile = np.mean(gray, axis=1)
    horizontal_var = np.var(np.diff(horizontal_profile))
    
    # Screens often have subtle horizontal patterns
    score = 0
    if horizontal_var > 50:
        score = 2.5
    elif horizontal_var > 30:
        score = 1.5
    
    return score

def check_color_temperature(face_roi):
    """Analyze color temperature - screens have artificial lighting"""
    hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
    
    # Real faces under natural/room lighting have varied color temps
    # Screens produce uniform artificial light
    h_channel = hsv[:, :, 0]
    s_channel = hsv[:, :, 1]
    
    h_std = np.std(h_channel)
    s_std = np.std(s_channel)
    
    # Also check for blue light dominance (LED screens)
    b, g, r = cv2.split(face_roi)
    blue_dominance = np.mean(b) / (np.mean(r) + np.mean(g) + 1)
    
    score = 0
    if h_std < 15 and s_std < 20:  # Very uniform color
        score += 2.0
    if blue_dominance > 0.6:  # Screen blue light
        score += 1.5
    
    return score

def check_texture_quality(face_roi):
    """Analyze skin texture - photos/screens lack micro-texture"""
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    
    # Real skin has pores, fine wrinkles, etc.
    # Use multiple scales of texture analysis
    score = 0
    
    # Fine texture (pores)
    fine_edges = cv2.Canny(gray, 50, 150)
    fine_density = np.sum(fine_edges > 0) / fine_edges.size
    
    # Texture variance
    texture_var = np.var(cv2.Laplacian(gray, cv2.CV_64F))
    
    if fine_density < 0.05:  # Too smooth
        score += 2.0
    if texture_var < 100:  # Lack of texture
        score += 2.0
    
    return score

def check_edge_artifacts(face_roi):
    """Detect screen/photo borders and artificial edges"""
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    
    # Look for straight lines (screen edges, photo borders)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
    
    score = 0
    if lines is not None and len(lines) > 5:
        # Check for long straight lines (unnatural for faces)
        long_lines = [l for l in lines if np.hypot(l[0][2]-l[0][0], l[0][3]-l[0][1]) > 50]
        if len(long_lines) > 3:
            score = 3.0
        elif len(long_lines) > 1:
            score = 1.5
    
    # Also check overall edge density
    edge_density = np.sum(edges > 0) / edges.size
    if edge_density > 0.15:  # Too many edges (screen bezels, photo edges)
        score += 1.5
    
    return score

def check_lighting_consistency(face_roi):
    """Analyze lighting - screens have flat/uniform lighting"""
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    
    # Real faces have depth-based lighting gradients
    # Divide face into regions and check variance
    h, w = gray.shape
    regions = [
        gray[0:h//3, :],           # top
        gray[h//3:2*h//3, :],      # middle
        gray[2*h//3:h, :],         # bottom
        gray[:, 0:w//3],           # left
        gray[:, w//3:2*w//3],      # center
        gray[:, 2*w//3:w],         # right
    ]
    
    region_means = [np.mean(r) for r in regions if r.size > 0]
    lighting_variance = np.var(region_means)
    
    score = 0
    if lighting_variance < 100:  # Too uniform
        score = 2.0
    elif lighting_variance < 200:
        score = 1.0
    
    # Also check for artificial brightness
    overall_brightness = np.mean(gray)
    if overall_brightness > 200 or overall_brightness < 40:  # Unnatural
        score += 1.0
    
    return score

def detect_photo_spoof(frame, face_location, executor=None):
    """Parallel anti-spoofing detection - runs all checks simultaneously"""
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
        return (False, 0.0, {})
    
    # If executor provided, run checks in parallel
    if executor is not None:
        futures = {
            'moire': executor.submit(check_moire_pattern, face_roi),
            'refresh': executor.submit(check_screen_refresh, face_roi),
            'color_temp': executor.submit(check_color_temperature, face_roi),
            'texture': executor.submit(check_texture_quality, face_roi),
            'edges': executor.submit(check_edge_artifacts, face_roi),
            'lighting': executor.submit(check_lighting_consistency, face_roi)
        }
        
        # Collect results
        scores = {name: future.result() for name, future in futures.items()}
    else:
        # Run sequentially if no executor
        scores = {
            'moire': check_moire_pattern(face_roi),
            'refresh': check_screen_refresh(face_roi),
            'color_temp': check_color_temperature(face_roi),
            'texture': check_texture_quality(face_roi),
            'edges': check_edge_artifacts(face_roi),
            'lighting': check_lighting_consistency(face_roi)
        }
    
    # Calculate total score
    total_score = sum(scores.values())
    
    # Determine if spoof (threshold: 6.0 for high confidence)
    is_spoof = total_score >= 6.0
    
    return (is_spoof, total_score, scores)
    
def check_parallax_motion(frame, face_location, prev_frame=None, prev_location=None):
    """Check for parallax effect - 3D objects show differential motion"""
    if prev_frame is None or prev_location is None:
        return 0
    
    top, right, bottom, left = face_location
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
        return 0
    
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
        
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(prev_roi, cv2.COLOR_BGR2GRAY)
        
        if prev_gray.shape == gray.shape:
            # Calculate optical flow
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flow_magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            flow_variance = np.var(flow_magnitude)
            
            # 3D faces show high variance (parallax)
            # 2D images move uniformly
            if flow_variance > 1.0:
                return 3.0
            elif flow_variance > 0.5:
                return 2.0
            elif flow_variance > 0.2:
                return 1.0
    except:
        pass
    
    return 0

def check_depth_gradients(face_roi):
    """Check for 3D depth cues in lighting gradients"""
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    
    # 3D faces have complex lighting from depth
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    gradient_complexity = np.std(gradient_magnitude)
    
    score = 0
    if gradient_complexity > 20:
        score = 2.5
    elif gradient_complexity > 15:
        score = 1.5
    
    return score

def check_face_depth_structure(face_roi):
    """Check for 3D facial structure (nose, cheeks, etc.)"""
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    
    # Nose bridge should be prominent in center
    roi_height, roi_width = gray.shape
    center_region = gray[roi_height//4:3*roi_height//4, 2*roi_width//5:3*roi_width//5]
    edge_regions = [
        gray[roi_height//4:3*roi_height//4, 0:roi_width//5],
        gray[roi_height//4:3*roi_height//4, 4*roi_width//5:roi_width]
    ]
    
    score = 0
    if center_region.size > 0:
        center_intensity = np.mean(center_region)
        edge_intensity = np.mean([np.mean(r) for r in edge_regions if r.size > 0])
        
        if abs(center_intensity - edge_intensity) > 15:
            score = 2.0
        elif abs(center_intensity - edge_intensity) > 10:
            score = 1.0
    
    return score

def detect_3d_depth(frame, face_location, prev_frame=None, prev_location=None, executor=None):
    """
    Detect if face is 3D (real) or 2D (photo/screen) using depth cues
    Returns: (is_3d, confidence_score, scores_dict)
    """
    top, right, bottom, left = face_location
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
        return (False, 0.0, {})
    
    # Run checks in parallel if executor provided
    if executor is not None:
        futures = {
            'parallax': executor.submit(check_parallax_motion, frame, face_location, prev_frame, prev_location),
            'gradients': executor.submit(check_depth_gradients, face_roi),
            'structure': executor.submit(check_face_depth_structure, face_roi)
        }
        scores = {name: future.result() for name, future in futures.items()}
    else:
        scores = {
            'parallax': check_parallax_motion(frame, face_location, prev_frame, prev_location),
            'gradients': check_depth_gradients(face_roi),
            'structure': check_face_depth_structure(face_roi)
        }
    
    total_score = sum(scores.values())
    
    # Stricter threshold: need 4.0+ to be considered 3D
    is_3d = total_score >= 4.0
    
    return (is_3d, total_score, scores)

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
                'is_live': False,
                'debug': {}  # For debugging detection scores
            }
            
            face_key = f"face_{idx}"
            
            # Get previous frame data for depth detection
            prev_frame = self.prev_frame
            prev_location = self.prev_locations.get(face_key)
            
            # Run all detection checks in parallel using thread pool
            is_3d, depth_score, depth_details = detect_3d_depth(
                frame, face_location, prev_frame, prev_location, executor=self.executor
            )
            
            is_spoof, spoof_score, spoof_details = detect_photo_spoof(
                frame, face_location, executor=self.executor
            )
            
            # Store debug info
            result['debug'] = {
                'spoof_score': spoof_score,
                'depth_score': depth_score,
                'spoof_details': spoof_details,
                'depth_details': depth_details
            }
            
            # BALANCED DECISION LOGIC:
            # Spoof detected with very high confidence -> REJECT
            if spoof_score >= 7.0:
                result['state'] = 'spoof'
                result['name'] = f"SCREEN/PHOTO! (s:{spoof_score:.1f})"
                self.trigger_alert('spoof', f"Spoofing detected - score: {spoof_score:.1f}", result)
                return result, result['name']
            
            # High spoof score AND no strong 3D evidence -> REJECT
            if spoof_score >= 5.5 and depth_score < 4.0:
                result['state'] = 'spoof'
                result['name'] = f"LIKELY SPOOF (s:{spoof_score:.1f}/d:{depth_score:.1f})"
                self.trigger_alert('spoof', f"Likely spoofing - spoof:{spoof_score:.1f}, depth:{depth_score:.1f}", result)
                return result, result['name']
            
            # Medium spoof indicators AND weak 3D -> REJECT
            if spoof_score >= 4.0 and depth_score < 3.0:
                result['state'] = 'spoof'
                result['name'] = f"POSSIBLE SPOOF (s:{spoof_score:.1f}/d:{depth_score:.1f})"
                self.trigger_alert('spoof', f"Possible spoofing detected", result)
                return result, result['name']
            
            # If we get here, spoof checks passed or are inconclusive
            # Check motion-based liveness with stricter requirements
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
                
                # Stricter motion requirements for security
                # Strong 3D evidence -> lenient motion requirement
                # Weak/no 3D evidence -> strict motion requirement
                
                if is_3d and depth_score >= 5.0:
                    # Very strong 3D evidence - minimal motion needed
                    is_live = frames_seen < 30 or motion_frames > 0
                elif is_3d or depth_score >= 3.0:
                    # Some 3D evidence - moderate motion needed
                    if frames_seen < 45:
                        is_live = True
                    else:
                        motion_ratio = motion_frames / frames_seen
                        is_live = motion_ratio > 0.05  # 5% motion
                else:
                    # Weak 3D evidence - strict motion requirement
                    if frames_seen < 40:
                        is_live = True
                    else:
                        motion_ratio = motion_frames / frames_seen
                        is_live = motion_ratio > 0.10  # 10% motion needed
            
            result['is_live'] = is_live
            
            # Determine what to display
            if not is_live:
                result['state'] = 'pending_verification'
                result['name'] = f"Move slightly (d:{depth_score:.1f})"
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
