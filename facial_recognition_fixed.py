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

def detect_blinks(face_landmarks, blink_history, face_key):
    """Detect blinks using eye aspect ratio - real faces blink"""
    EYE_AR_THRESH = 0.25
    EYE_AR_CONSEC_FRAMES = 2
    
    if face_key not in blink_history:
        blink_history[face_key] = {'counter': 0, 'total': 0, 'frames': 0}
    
    if not face_landmarks:
        return 0, blink_history
    
    # Get eye coordinates (face_recognition uses 68 landmark model)
    left_eye = face_landmarks['left_eye']
    right_eye = face_landmarks['right_eye']
    
    # Calculate EAR for both eyes
    left_ear = eye_aspect_ratio(left_eye)
    right_ear = eye_aspect_ratio(right_eye)
    ear = (left_ear + right_ear) / 2.0
    
    # Detect blink
    if ear < EYE_AR_THRESH:
        blink_history[face_key]['counter'] += 1
    else:
        if blink_history[face_key]['counter'] >= EYE_AR_CONSEC_FRAMES:
            blink_history[face_key]['total'] += 1
        blink_history[face_key]['counter'] = 0
    
    blink_history[face_key]['frames'] += 1
    
    # Calculate liveness score based on blinks
    frames = blink_history[face_key]['frames']
    blinks = blink_history[face_key]['total']
    
    # Natural blink rate: 15-20 per minute (0.25-0.33 per second)
    # At 30 FPS, expect ~1 blink per 90-120 frames
    score = 0
    if frames > 120:  # After 4 seconds - give more time
        if blinks >= 1:
            score = 3.0  # Strong evidence of liveness
        elif frames > 300 and blinks == 0:  # 10 seconds with no blinks
            score = -1.5  # Less harsh penalty
    
    return score, blink_history

def detect_pulse(face_roi_history, face_key):
    """Detect subtle color changes from blood flow - real faces have pulse"""
    if face_key not in face_roi_history or len(face_roi_history[face_key]) < 30:
        return 0
    
    # Analyze green channel (best for pulse detection)
    green_means = []
    for roi in face_roi_history[face_key][-30:]:
        if roi is not None and roi.size > 0:
            # Use forehead area (top 1/3 of face)
            h = roi.shape[0]
            forehead = roi[0:h//3, :]
            green = forehead[:, :, 1]
            green_means.append(np.mean(green))
    
    if len(green_means) < 20:
        return 0
    
    # Check for rhythmic variation (pulse)
    green_std = np.std(green_means)
    
    # Real faces: 0.5-3.0 std in green channel
    # Static images: < 0.3 std
    score = 0
    if green_std > 0.8:
        score = 2.0  # Good pulse signal
    elif green_std > 0.5:
        score = 1.0
    elif green_std < 0.3:
        score = -1.0  # Too static = fake
    
    return score

def detect_micro_movements(landmarks_history, face_key):
    """Detect subtle natural movements - real people constantly make tiny adjustments"""
    if face_key not in landmarks_history or len(landmarks_history[face_key]) < 15:
        return 0
    
    # Track nose tip movement (most stable landmark)
    nose_positions = []
    for landmarks in landmarks_history[face_key][-15:]:
        if landmarks and 'nose_tip' in landmarks:
            nose_tip = landmarks['nose_tip'][2]  # Middle point
            nose_positions.append(nose_tip)
    
    if len(nose_positions) < 10:
        return 0
    
    # Calculate micro-movement variance
    x_positions = [p[0] for p in nose_positions]
    y_positions = [p[1] for p in nose_positions]
    
    x_std = np.std(x_positions)
    y_std = np.std(y_positions)
    total_movement = x_std + y_std
    
    score = 0
    # Real faces: 1-10 pixels of micro-movement
    # Photos/screens: < 0.5 pixels
    if total_movement > 3.0:
        score = 2.0  # Good natural movement
    elif total_movement > 1.0:
        score = 1.0
    elif total_movement < 0.5:
        score = -1.5  # Too static = fake
    
    return score

def check_temporal_consistency(frame_history, face_key):
    """Real faces have consistent lighting changes, screens have abrupt changes"""
    if face_key not in frame_history or len(frame_history[face_key]) < 10:
        return 0
    
    # Calculate brightness variance over time
    brightness_values = []
    for frame in frame_history[face_key][-10:]:
        if frame is not None and frame.size > 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness_values.append(np.mean(gray))
    
    if len(brightness_values) < 8:
        return 0
    
    # Check for smooth transitions
    diffs = np.diff(brightness_values)
    abrupt_changes = np.sum(np.abs(diffs) > 10)  # Count large jumps
    
    score = 0
    if abrupt_changes >= 3:  # Multiple abrupt changes = screen
        score = 1.5
    
    return score

def check_face_size_consistency(face_location, size_history, face_key):
    """Real faces change size slightly with movement, photos stay constant"""
    top, right, bottom, left = face_location
    current_size = (bottom - top) * (right - left)
    
    if face_key not in size_history:
        size_history[face_key] = []
    
    size_history[face_key].append(current_size)
    
    # Keep only last 30 frames
    if len(size_history[face_key]) > 30:
        size_history[face_key] = size_history[face_key][-30:]
    
    score = 0
    if len(size_history[face_key]) > 20:  # Need more samples
        sizes = size_history[face_key]
        size_variance = np.var(sizes)
        size_std = np.std(sizes)
        
        # Real faces: moderate variance from breathing, head movement
        # Photos: very consistent size
        if size_std > 50:
            score = 2.0  # Good variance
        elif size_std < 5:  # More lenient threshold
            score = -1.0  # Less harsh penalty
    
    return score, size_history

def estimate_face_distance(face_location, frame_shape):
    """
    Estimate relative distance between camera and face based on face size.
    Larger face in frame = closer to camera = potential phone being held up close.
    Returns: (distance_category, face_size_ratio, is_too_close)
    """
    top, right, bottom, left = face_location
    # Locations are in downscaled coordinates (0.25x), so multiply by 4
    face_height = (bottom - top) * 4
    face_width = (right - left) * 4
    
    # Calculate face area as percentage of frame
    frame_height, frame_width = frame_shape[:2]
    face_area = face_height * face_width
    frame_area = frame_height * frame_width
    face_size_ratio = face_area / frame_area
    
    # Categorize distance based on face size
    # Normal face at comfortable distance: 8-12% of frame
    # Too close (auto-lock): >=15% of frame
    
    if face_size_ratio >= 0.15:
        return ("too_close", face_size_ratio, True)
    elif face_size_ratio > 0.08:
        return ("normal", face_size_ratio, False)
    else:
        return ("far", face_size_ratio, False)

def check_reflection_patterns(face_roi):
    """Detect screen/photo reflections and glare"""
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    
    # Look for very bright spots (screen glare, photo reflections)
    bright_mask = gray > 240
    bright_ratio = np.sum(bright_mask) / gray.size
    
    # Check for specular highlights (screen reflections)
    # Real skin has diffuse reflection, screens have specular
    lab = cv2.cvtColor(face_roi, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    
    # High L values indicate bright spots
    high_l_mask = l_channel > 230
    highlight_ratio = np.sum(high_l_mask) / l_channel.size
    
    score = 0
    if bright_ratio > 0.05 or highlight_ratio > 0.05:
        score = 2.0  # Suspicious reflections
    elif bright_ratio > 0.03 or highlight_ratio > 0.03:
        score = 1.0
    
    return score

def check_color_histogram_uniformity(face_roi):
    """Photos/screens have artificial color distribution"""
    hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
    
    # Calculate histogram for each channel
    h_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
    s_hist = cv2.calcHist([hsv], [1], None, [256], [0, 256])
    v_hist = cv2.calcHist([hsv], [2], None, [256], [0, 256])
    
    # Normalize
    h_hist = h_hist.flatten() / np.sum(h_hist)
    s_hist = s_hist.flatten() / np.sum(s_hist)
    v_hist = v_hist.flatten() / np.sum(v_hist)
    
    # Calculate entropy - real faces have higher entropy
    def entropy(hist):
        hist = hist[hist > 0]
        return -np.sum(hist * np.log2(hist))
    
    h_entropy = entropy(h_hist)
    s_entropy = entropy(s_hist)
    
    score = 0
    # Low entropy = uniform color (artificial)
    if h_entropy < 3.0 or s_entropy < 4.0:
        score = 1.5
    
    return score

def check_head_pose_variation(face_landmarks_history, face_key):
    """Real people naturally move their head slightly"""
    if face_key not in face_landmarks_history or len(face_landmarks_history[face_key]) < 10:
        return 0
    
    # Get nose tip positions over time
    nose_positions = []
    for landmarks in face_landmarks_history[face_key]:
        if landmarks and 'nose_tip' in landmarks:
            nose_tip = landmarks['nose_tip'][0]  # Get first point
            nose_positions.append(nose_tip)
    
    if len(nose_positions) < 10:
        return 0
    
    # Calculate variance in position
    nose_x = [p[0] for p in nose_positions]
    nose_y = [p[1] for p in nose_positions]
    
    x_variance = np.var(nose_x)
    y_variance = np.var(nose_y)
    
    total_variance = x_variance + y_variance
    
    score = 0
    if total_variance > 20:
        score = 2.5  # Good head movement
    elif total_variance > 10:
        score = 1.5
    elif total_variance > 5:
        score = 0.5  # Some movement
    # Don't penalize for being static
    
    return score

def detect_screen_contrast(face_roi):
    """
    Analyze contrast patterns - screens have artificial contrast vs natural face lighting.
    Screens: High contrast edges, flat regions, artificial uniformity.
    Real faces: Smooth gradients, natural shadows, varied contrast.
    """
    try:
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # Calculate global contrast (std deviation of intensity)
        global_contrast = np.std(gray)
        
        # Calculate local contrast using Sobel gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        edge_strength = np.mean(gradient_magnitude)
        
        # Detect sharp transitions (screens have artificial edges)
        sharp_edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(sharp_edges > 0) / sharp_edges.size
        
        # Check contrast uniformity (screens are artificially uniform)
        # Split into regions and check variance
        h, w = gray.shape
        regions = [
            gray[0:h//2, 0:w//2],
            gray[0:h//2, w//2:w],
            gray[h//2:h, 0:w//2],
            gray[h//2:h, w//2:w]
        ]
        region_contrasts = [np.std(r) for r in regions if r.size > 0]
        contrast_variance = np.var(region_contrasts)
        
        score = 0
        
        # Screens have:
        # 1. Artificially uniform contrast (low variance across regions)
        # 2. Sharp artificial edges (high edge density)
        # 3. Flat regions with sudden transitions
        
        # Low contrast variance = uniform screen backlight
        if contrast_variance < 20 and global_contrast > 30:
            score = 3.0
            print(f"    🔴 Artificial contrast uniformity: var={contrast_variance:.1f}")
        elif contrast_variance < 40:
            score = 2.0
        elif contrast_variance < 60:
            score = 1.0
        
        # High edge density = screen pixels/artifacts
        if edge_density > 0.15:
            score += 1.5
            print(f"    🔴 High edge density (screen pixels): {edge_density:.2%}")
        elif edge_density > 0.10:
            score += 1.0
        
        return min(score, 5.0)
        
    except Exception as e:
        print(f"    ⚠️  Contrast detection error: {e}")
        return 0

def detect_screen_brightness(face_roi, background_roi=None):
    """
    Analyze brightness levels - screens emit light (bright), real faces reflect (dimmer).
    Screens have artificially high and uniform brightness.
    Also checks relative brightness vs background - screens are brighter than environment even at low brightness.
    """
    try:
        # Convert to LAB color space for accurate luminance analysis
        lab = cv2.cvtColor(face_roi, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]  # Luminance channel
        
        # Calculate brightness statistics
        mean_brightness = np.mean(l_channel)
        brightness_std = np.std(l_channel)
        max_brightness = np.max(l_channel)
        
        # Calculate percentage of very bright pixels
        very_bright_mask = l_channel > 200
        bright_ratio = np.sum(very_bright_mask) / l_channel.size
        
        # Also check RGB brightness
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        rgb_brightness = np.mean(gray)
        
        score = 0
        relative_score = 0
        
        # ABSOLUTE BRIGHTNESS CHECK
        # Screens characteristics:
        # 1. Very high mean brightness (>170 in LAB L channel)
        # 2. Low std (uniform backlight)
        # 3. Many pixels near maximum brightness
        
        # High brightness = screen emission
        if mean_brightness > 180:
            score = 3.0
            print(f"    🔴 Screen brightness detected: L={mean_brightness:.1f}")
        elif mean_brightness > 160:
            score = 2.5
        elif mean_brightness > 140:
            score = 1.5
        
        # Low std with high brightness = uniform screen backlight
        if mean_brightness > 140 and brightness_std < 25:
            score += 1.5
            print(f"    🔴 Uniform screen backlight: std={brightness_std:.1f}")
        
        # High ratio of very bright pixels
        if bright_ratio > 0.3:
            score += 1.0
            print(f"    🔴 High bright pixel ratio: {bright_ratio:.2%}")
        
        # RELATIVE BRIGHTNESS CHECK (vs background)
        if background_roi is not None and background_roi.size > 0:
            try:
                # Analyze background brightness
                bg_lab = cv2.cvtColor(background_roi, cv2.COLOR_BGR2LAB)
                bg_l_channel = bg_lab[:, :, 0]
                bg_mean_brightness = np.mean(bg_l_channel)
                bg_std = np.std(bg_l_channel)
                
                # Calculate brightness difference
                brightness_diff = mean_brightness - bg_mean_brightness
                brightness_ratio = mean_brightness / (bg_mean_brightness + 1e-10)
                
                # Screens are ALWAYS brighter than their environment
                # Even at low brightness, a phone screen emits light while environment reflects it
                # Real faces: similar brightness to environment (±20 points)
                # Screens: 30-80+ points brighter than environment
                
                if brightness_diff > 40:  # Significantly brighter than environment
                    relative_score += 3.0
                    print(f"    🔴 Much brighter than environment: diff={brightness_diff:.1f}")
                elif brightness_diff > 25:
                    relative_score += 2.0
                elif brightness_diff > 15:
                    relative_score += 1.0
                
                # Also check ratio
                if brightness_ratio > 1.3:  # 30% brighter
                    relative_score += 1.5
                elif brightness_ratio > 1.15:  # 15% brighter
                    relative_score += 0.5
                
                # Uniformity check - screens are more uniform than natural environments
                uniformity_diff = abs(brightness_std - bg_std)
                if brightness_std < 20 and bg_std > 30:  # Face uniform, background varied
                    relative_score += 1.0
                    
            except Exception as e:
                pass  # Background analysis failed, use only absolute score
        
        return min(score, 5.0), min(relative_score, 5.0)
        
    except Exception as e:
        print(f"    ⚠️  Brightness detection error: {e}")
        return 0, 0

# Individual detection methods for parallel execution
def check_moire_pattern(gray):
    """Detect moiré patterns from screen pixels (accepts grayscale)"""
    # Apply FFT to detect periodic patterns
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    
    # Look for regular patterns in frequency domain
    h, w = magnitude.shape
    center_h, center_w = h // 2, w // 2
    
    # Exclude DC component
    mask = np.ones((h, w), dtype=bool)
    mask[center_h-5:center_h+5, center_w-5:center_w+5] = False
    
    freq_peaks = magnitude[mask]
    peak_ratio = np.max(freq_peaks) / (np.mean(freq_peaks) + 1e-10)
    
    # Much higher thresholds - only very clear screen patterns
    score = 0
    if peak_ratio > 100:  # Very strong (was 50)
        score = 3.0
    elif peak_ratio > 70:  # Strong (was 30)
        score = 2.0
    elif peak_ratio > 50:  # Moderate (was 20)
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

def check_edge_artifacts(gray):
    """Detect screen/photo borders (accepts grayscale)"""
    # High thresholds - avoid natural face contours
    edges = cv2.Canny(gray, 200, 300)
    
    # Very strict - only perfect straight lines
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=60, maxLineGap=3)
    
    score = 0
    if lines is not None and len(lines) > 10:
        long_lines = [l for l in lines if np.hypot(l[0][2]-l[0][0], l[0][3]-l[0][1]) > 80]
        
        if len(long_lines) > 0:
            angles = [np.arctan2(l[0][3]-l[0][1], l[0][2]-l[0][0]) for l in long_lines]
            # Must be horizontal or vertical (screen bezels)
            rect_lines = sum(1 for a in angles if abs(a) < 0.05 or abs(a - np.pi/2) < 0.05 or abs(a + np.pi/2) < 0.05)
            
            if rect_lines > 6:
                score = 3.0
            elif rect_lines > 4:
                score = 1.5
    
    return score

def check_lighting_consistency(gray):
    """Analyze lighting - only flag extremely flat lighting (accepts grayscale)"""
    h, w = gray.shape
    regions = [
        gray[0:h//3, :], gray[h//3:2*h//3, :], gray[2*h//3:h, :],
        gray[:, 0:w//3], gray[:, w//3:2*w//3], gray[:, 2*w//3:w],
    ]
    
    region_means = [np.mean(r) for r in regions if r.size > 0]
    lighting_variance = np.var(region_means)
    
    # Only flag extremely flat lighting (screens)
    score = 0
    if lighting_variance < 30:  # Very flat (was 100)
        score = 2.0
    elif lighting_variance < 50:  # Somewhat flat (was 200)
        score = 1.0
    
    return score

def detect_phone_rectangle(face_roi, visualize=False):
    """Detect if the detected face appears to be displayed on a screen (any rectangle = phone/screen)"""
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # Edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    score = 0
    rectangles_found = 0
    detected_rectangles = []  # Store all detected rectangles for visualization
    glasses_rectangles = []  # Store filtered glasses rectangles
    
    # Since we already detected a face, ANY rectangle in the frame suggests it's on a screen
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Skip tiny contours (glasses lenses are typically small)
        if area < 500:
            continue
        
        # Get bounding rectangle
        x, y, w_rect, h_rect = cv2.boundingRect(contour)
        
        # Check if it's rectangular using contour approximation
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.03 * peri, True)
        
        # Look for 4-sided shapes (rectangles)
        if len(approx) == 4:
            aspect_ratio = float(w_rect) / h_rect if h_rect > 0 else 0
            
            # Calculate rectangularity
            rect_area = w_rect * h_rect
            rectangularity = area / rect_area if rect_area > 0 else 0
            
            # Calculate area as percentage of face ROI
            area_ratio = area / (w * h)
            
            # FILTER OUT GLASSES:
            # - Glasses are small (< 10% of face ROI)
            # - Glasses have specific aspect ratios (1.5-3.0, wider than tall)
            # - Glasses are in upper portion of face (y < h/2)
            is_glasses_like = (
                area_ratio < 0.10 and  # Small relative to face
                1.2 < aspect_ratio < 3.5 and  # Wider than tall (glasses shape)
                y < h * 0.6  # Upper portion of face
            )
            
            if is_glasses_like:
                glasses_rectangles.append({
                    'x': x, 'y': y, 'w': w_rect, 'h': h_rect
                })
                continue  # Skip glasses
            
            # Phone-like rectangle characteristics:
            # - Must be larger (at least 20% of face ROI)
            # - Good rectangularity (> 0.8)
            # - Phone-like aspect ratio
            if rectangularity > 0.80 and 0.5 < aspect_ratio < 2.0 and area_ratio > 0.20:
                detected_rectangles.append({
                    'x': x, 'y': y, 'w': w_rect, 'h': h_rect,
                    'area_ratio': area_ratio, 'aspect_ratio': aspect_ratio
                })
                score += 1.5
                rectangles_found += 1
    
    # Also check for parallel straight lines (screen bezels)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=35, minLineLength=int(min(w,h)*0.3), maxLineGap=10)
    
    h_lines = []
    v_lines = []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            
            if length < 30:
                continue
                
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            
            # Horizontal (±12 degrees for tolerance)
            if angle < 12 or angle > 168:
                h_lines.append((x1, y1, x2, y2))
            # Vertical (90° ±12 degrees)
            elif 78 < angle < 102:
                v_lines.append((x1, y1, x2, y2))
        
        # Phones have parallel edges
        if len(h_lines) >= 2 and len(v_lines) >= 2:
            score += 2.0
            print(f"    ⚠️  Phone edges detected: H={len(h_lines)}, V={len(v_lines)}")
        elif len(h_lines) >= 1 and len(v_lines) >= 1:
            score += 1.0
    
    # Return detection data for drawing on main frame
    detection_data = {
        'score': score,
        'rectangles': detected_rectangles,
        'glasses': glasses_rectangles,
        'h_lines': h_lines,
        'v_lines': v_lines,
        'rectangles_found': rectangles_found
    }
    
    return score, detection_data

def detect_photo_spoof(frame, face_location, executor=None, camera_profile=None, visualize=False):
    """SIMPLE: Just detect if there's a rectangular phone shape"""
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
        return (False, 0.0, {}, None)
    
    # SINGLE DETECTION: Look for phone rectangle
    rectangle_score, detection_data = detect_phone_rectangle(face_roi, visualize=visualize)
    
    scores = {
        'phone_rectangle': rectangle_score,
    }
    
    # If rectangle detected, max out the spoof score
    if rectangle_score >= 1.0:
        weighted_total = 10.0  # Maximum spoof score
        is_spoof = True
        print(f"\n🚨🚨🚨 PHONE RECTANGLE DETECTED! 🚨🚨🚨")
        print(f"  Rectangle Score: {rectangle_score:.2f}")
        print(f"  SPOOF SCORE: 10.0 (MAX)")
        print("=" * 40)
    else:
        weighted_total = rectangle_score
        is_spoof = False
        print(f"\n✅ REAL FACE - No rectangles detected")
        print(f"  Rectangle Score: {rectangle_score:.2f}")
        print("=" * 40)
    
    return (is_spoof, weighted_total, scores, detection_data)
    
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
        
        # Enhanced anti-spoofing tracking
        blink_history = {}  # Track blinks per face
        size_history = {}  # Track face size changes
        landmarks_history = {}  # Track facial landmarks over time
        
        # Biometric analysis tracking
        face_roi_history = {}  # Track face ROI for pulse detection
        frame_history = {}  # Track frames for temporal consistency
        biometric_scores = {}  # Store biometric scores with rolling average
        
        # Screen detection history (rolling averages)
        contrast_score_history = {}  # Track contrast scores over time
        brightness_score_history = {}  # Track brightness scores over time
        
        # Screen detection state - remember when screen was detected too close
        screen_detected_state = {}  # Track if face was flagged as screen (persists until cleared)
        screen_detection_counter = {}  # Count frames since screen detected
        
        # Face persistence - keep faces displayed
        face_history = {}  # Store face data with timestamps
        face_retention_time = 15.0  # Keep faces for 5 seconds after last detection
        
        # Alert/Alarm system
        alerts = []  # Store alert events
        alert_cooldown = {}  # Prevent spam alerts
        alert_cooldown_time = 5.0  # Seconds between alerts for same type
        
        # Spoof score history for temporal smoothing
        spoof_score_history = {}  # Track spoof scores per face to prevent flickering
        spoof_flags = {}  # Track faces that have been marked as spoofs
        spoof_clear_counter = {}  # Counter for clearing spoof flags
        
        # Persistent face tracking
        tracked_faces = {}  # Map encoding to persistent face_id
        next_face_id = 0  # Counter for assigning new face IDs
        
        def __init__(self, known_faces_dir='faces', max_workers=4):
            self.known_face_encodings = []
            self.known_face_names = []
            # Thread pool for parallel processing
            self.max_workers = max_workers
            self.executor = ThreadPoolExecutor(max_workers=max_workers)
            self.lock = threading.Lock()
            
            # Initialize persistent tracking
            self.tracked_faces = {}
            self.next_face_id = 0
            
            print(f"🔧 Initialized with {max_workers} worker threads for CPU optimization")
            
            # Camera calibration - detect sensor and tune parameters
            print("\n📷 Starting camera calibration...")
            self.camera_profile = self.calibrate_camera()
            
            self.encode_faces(known_faces_dir)
            
        def calibrate_camera(self):
            """Detect camera sensor format and focal length to tune detection parameters"""
            profile = {
                'sensor_width': None,
                'sensor_height': None,
                'focal_length': None,
                'resolution': None,
                'brightness_scale': 1.0,
                'distance_scale': 1.0,
                'calibrated': False
            }
            
            try:
                # Open camera temporarily for calibration
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    print("  ⚠️  Could not open camera for calibration")
                    return profile
                
                # Get resolution
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                profile['resolution'] = (width, height)
                print(f"  ✓ Resolution: {width}x{height}")
                
                # LOCK CAMERA SETTINGS - Prevent auto-adjustment during operation
                print("  🔒 Locking camera settings...")
                
                # Disable auto exposure
                try:
                    # Try to disable auto exposure (platform dependent)
                    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 0.25 = manual mode (some cameras)
                    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)     # 1 = manual mode (other cameras)
                    print("    ✓ Auto exposure disabled")
                except:
                    print("    ⚠️  Could not disable auto exposure")
                
                # Get current exposure and lock it
                current_exposure = cap.get(cv2.CAP_PROP_EXPOSURE)
                if current_exposure != -1:
                    cap.set(cv2.CAP_PROP_EXPOSURE, current_exposure)
                    profile['locked_exposure'] = current_exposure
                    print(f"    ✓ Locked exposure: {current_exposure}")
                
                # Disable auto white balance
                try:
                    cap.set(cv2.CAP_PROP_AUTO_WB, 0)  # 0 = disable
                    print("    ✓ Auto white balance disabled")
                except:
                    print("    ⚠️  Could not disable auto white balance")
                
                # Lock brightness
                current_brightness = cap.get(cv2.CAP_PROP_BRIGHTNESS)
                if current_brightness != -1:
                    cap.set(cv2.CAP_PROP_BRIGHTNESS, current_brightness)
                    profile['locked_brightness'] = current_brightness
                    print(f"    ✓ Locked brightness: {current_brightness}")
                
                # Lock gain
                current_gain = cap.get(cv2.CAP_PROP_GAIN)
                if current_gain != -1:
                    cap.set(cv2.CAP_PROP_GAIN, current_gain)
                    profile['locked_gain'] = current_gain
                    print(f"    ✓ Locked gain: {current_gain}")
                
                # Store locked settings
                profile['settings_locked'] = True
                
                # Try to get focal length (not all cameras support this)
                focal_length = cap.get(cv2.CAP_PROP_FOCUS)
                if focal_length > 0:
                    profile['focal_length'] = focal_length
                    print(f"  ✓ Focal Length: {focal_length:.1f}mm")
                else:
                    # Estimate based on typical webcam FOV (70-80 degrees)
                    # Assuming 75 degree horizontal FOV
                    fov_rad = 75 * (3.14159 / 180)
                    estimated_focal = (width / 2) / np.tan(fov_rad / 2)
                    profile['focal_length'] = estimated_focal
                    print(f"  ⚠️  Focal length unknown, estimated: {estimated_focal:.1f}px (75° FOV)")
                
                # Estimate sensor size based on resolution
                # Most webcams are 1/3" to 1/2.3" sensors
                aspect_ratio = width / height
                if width >= 1920:  # HD or higher
                    sensor_diagonal_mm = 7.0  # ~1/2.3" sensor
                elif width >= 1280:
                    sensor_diagonal_mm = 6.0  # ~1/3" sensor  
                else:
                    sensor_diagonal_mm = 4.5  # ~1/4" sensor
                
                # Calculate sensor dimensions from diagonal
                sensor_width_mm = sensor_diagonal_mm / np.sqrt(1 + (1/aspect_ratio)**2)
                sensor_height_mm = sensor_width_mm / aspect_ratio
                profile['sensor_width'] = sensor_width_mm
                profile['sensor_height'] = sensor_height_mm
                print(f"  ✓ Estimated Sensor: {sensor_width_mm:.1f}x{sensor_height_mm:.1f}mm")
                
                # Calculate scaling factors relative to full frame (36x24mm)
                full_frame_width = 36.0
                crop_factor = full_frame_width / sensor_width_mm
                profile['crop_factor'] = crop_factor
                print(f"  ✓ Crop Factor: {crop_factor:.2f}x")
                
                # Capture test frame for brightness calibration
                ret, frame = cap.read()
                if ret:
                    # Analyze ambient brightness
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    mean_brightness = np.mean(gray)
                    std_brightness = np.std(gray)
                    
                    # Convert to LAB for better brightness analysis
                    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                    l_channel = lab[:, :, 0]
                    lab_brightness = np.mean(l_channel)
                    
                    print(f"  ✓ Ambient: Gray={mean_brightness:.1f}, LAB L={lab_brightness:.1f}")
                    
                    # Adjust brightness threshold based on sensor characteristics
                    # Smaller sensors with higher crop factors tend to be noisier and have different brightness characteristics
                    if crop_factor > 5.0:  # Small sensor (1/3" or smaller)
                        profile['brightness_scale'] = 0.85  # More lenient
                    elif crop_factor > 3.0:  # Medium sensor (1/2.3")
                        profile['brightness_scale'] = 0.92
                    else:  # Larger sensor
                        profile['brightness_scale'] = 1.0
                    
                    # Adjust distance thresholds based on focal length and sensor
                    # Wider FOV cameras need adjusted distance calculations
                    fov_horizontal = 2 * np.arctan(sensor_width_mm / (2 * profile['focal_length'])) * (180 / 3.14159)
                    print(f"  ✓ Estimated Horizontal FOV: {fov_horizontal:.1f}°")
                    
                    if fov_horizontal > 80:  # Wide angle
                        profile['distance_scale'] = 1.15  # Face appears smaller, adjust threshold
                    elif fov_horizontal < 60:  # Narrow angle
                        profile['distance_scale'] = 0.9  # Face appears larger
                    else:
                        profile['distance_scale'] = 1.0
                    
                    profile['calibrated'] = True
                    print(f"  ✓ Brightness Scale: {profile['brightness_scale']:.2f}")
                    print(f"  ✓ Distance Scale: {profile['distance_scale']:.2f}")
                
                cap.release()
                print("  ✅ Camera calibration complete\n")
                
            except Exception as e:
                print(f"  ⚠️  Calibration error: {e}")
                print("  ℹ️  Using default parameters\n")
            
            return profile
            
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
        
        def process_single_face(self, frame, face_encoding, face_location, idx, rgb_small_frame):
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
            
            # PERSISTENT FACE TRACKING - Match face to existing tracked faces
            matched_id = None
            for tracked_key, tracked_encoding in list(self.tracked_faces.items()):
                # Compare with tracked encodings
                distance = face_recognition.face_distance([tracked_encoding], face_encoding)[0]
                if distance < 0.6:  # Same face
                    matched_id = tracked_key
                    # Update encoding (slight drift over time)
                    self.tracked_faces[tracked_key] = face_encoding
                    break
            
            # If no match, this is a new face
            if matched_id is None:
                matched_id = f"persistent_face_{self.next_face_id}"
                self.next_face_id += 1
                self.tracked_faces[matched_id] = face_encoding
                print(f"  🆕 New face tracked: {matched_id}")
            
            # Use persistent ID for history tracking
            face_key = matched_id
            
            # Get facial landmarks for blink detection and head pose
            face_landmarks_list = face_recognition.face_landmarks(rgb_small_frame, [face_location])
            face_landmarks = face_landmarks_list[0] if face_landmarks_list else None
            
            # Store landmarks history
            if face_key not in self.landmarks_history:
                self.landmarks_history[face_key] = []
            if face_landmarks:
                self.landmarks_history[face_key].append(face_landmarks)
                # Keep last 30 frames
                if len(self.landmarks_history[face_key]) > 30:
                    self.landmarks_history[face_key] = self.landmarks_history[face_key][-30:]
            
            # Store face ROI for biometric analysis (pulse detection)
            if matched_id not in self.face_roi_history:
                self.face_roi_history[matched_id] = []
            
            # Extract face ROI
            top, right, bottom, left = face_location
            top_scaled = max(0, top * 4 - 10)
            bottom_scaled = min(frame.shape[0], bottom * 4 + 10)
            left_scaled = max(0, left * 4 - 10)
            right_scaled = min(frame.shape[1], right * 4 + 10)
            face_roi = frame[top_scaled:bottom_scaled, left_scaled:right_scaled]
            
            # Extract background ROI (area around face but not face itself)
            # Get regions to the left, right, top, bottom of face
            h, w = frame.shape[:2]
            margin = 50  # pixels around face for background sampling
            
            # Sample from sides and top (avoid bottom which might be neck/clothing)
            bg_samples = []
            # Left side
            if left_scaled > margin:
                bg_samples.append(frame[top_scaled:bottom_scaled, max(0, left_scaled-margin):left_scaled])
            # Right side  
            if right_scaled + margin < w:
                bg_samples.append(frame[top_scaled:bottom_scaled, right_scaled:min(w, right_scaled+margin)])
            # Top
            if top_scaled > margin:
                bg_samples.append(frame[max(0, top_scaled-margin):top_scaled, left_scaled:right_scaled])
            
            # Combine background samples
            background_roi = None
            if bg_samples:
                try:
                    # Concatenate all background samples
                    background_roi = np.vstack([s.reshape(-1, s.shape[-1]) for s in bg_samples if s.size > 0])
                    background_roi = background_roi.reshape(-1, background_roi.shape[0] // len(bg_samples[0]), 3)
                except:
                    background_roi = bg_samples[0] if bg_samples else None
            
            self.face_roi_history[matched_id].append(face_roi.copy())
            if len(self.face_roi_history[matched_id]) > 30:
                self.face_roi_history[matched_id] = self.face_roi_history[matched_id][-30:]
            
            # Store frame history for temporal consistency
            if matched_id not in self.frame_history:
                self.frame_history[matched_id] = []
            self.frame_history[matched_id].append(face_roi.copy())
            if len(self.frame_history[matched_id]) > 10:
                self.frame_history[matched_id] = self.frame_history[matched_id][-10:]
            
            # SCREEN DETECTION ANALYSIS
            biometric_details = {}
            
            # DISTANCE ESTIMATION - Check if face is too close (phone being held up)
            distance_category, face_size_ratio, is_too_close = estimate_face_distance(face_location, frame.shape)
            biometric_details['distance_category'] = distance_category
            biometric_details['face_size_ratio'] = face_size_ratio
            biometric_details['is_too_close'] = is_too_close
            
            # SCREEN BRIGHTNESS - Detect screen emission vs reflected light
            brightness_score, relative_brightness_score = detect_screen_brightness(face_roi, background_roi)
            biometric_details['screen_brightness'] = brightness_score
            biometric_details['relative_brightness'] = relative_brightness_score
            
            # Track brightness with rolling average (5 frames)
            if matched_id not in self.brightness_score_history:
                self.brightness_score_history[matched_id] = []
            self.brightness_score_history[matched_id].append(brightness_score + relative_brightness_score)  # Combined score
            if len(self.brightness_score_history[matched_id]) > 5:
                self.brightness_score_history[matched_id] = self.brightness_score_history[matched_id][-5:]
            avg_brightness = np.mean(self.brightness_score_history[matched_id]) if self.brightness_score_history[matched_id] else 0
            biometric_details['avg_brightness'] = avg_brightness
            biometric_details['avg_absolute_brightness'] = brightness_score
            biometric_details['avg_relative_brightness'] = relative_brightness_score
            
            # If face is too close, log it
            if is_too_close:
                print(f"    ⚠️  Face too close ({face_size_ratio:.1%} of frame) - Distance: {distance_category}")
            
            # Get previous frame data for depth detection
            prev_frame = self.prev_frame
            prev_location = self.prev_locations.get(face_key)
            
            # Run depth detection check
            is_3d, depth_score, depth_details = detect_3d_depth(
                frame, face_location, prev_frame, prev_location, executor=self.executor
            )
            
            # Store debug info
            result['debug'] = {
                'depth_score': depth_score,
                'depth_details': depth_details,
                'biometric_details': biometric_details,
            }
            
            # Get screen indicators (using rolling averages)
            avg_brightness = biometric_details.get('avg_brightness', 0)
            is_too_close = biometric_details.get('is_too_close', False)
            distance_category = biometric_details.get('distance_category', 'normal')
            face_size_ratio = biometric_details.get('face_size_ratio', 0)
            
            # Apply camera calibration to brightness threshold
            brightness_threshold = 1.2
            if hasattr(self, 'camera_profile') and self.camera_profile.get('calibrated', False):
                brightness_threshold *= self.camera_profile.get('brightness_scale', 1.0)
            
            # CHECK PERSISTENT SCREEN STATE - If previously flagged as too close, stay blocked until cleared
            if matched_id in self.screen_detected_state and self.screen_detected_state[matched_id]:
                # Check if conditions met to CLEAR the state
                # Must be: Not too close + Low brightness
                can_clear = (
                    not is_too_close and
                    avg_brightness < (brightness_threshold * 0.7)  # More lenient for clearing
                )
                
                if can_clear:
                    # Clear the state - real person detected
                    self.screen_detected_state[matched_id] = False
                    print(f"  State CLEARED | Distance:{distance_category}")
                else:
                    # Still blocked - maintain locked state
                    result['state'] = 'spoof'
                    result['name'] = f"LOCKED - Too Close"
                    print(f"  LOCKED | Distance:{distance_category} Size:{face_size_ratio:.0%} Bright:{avg_brightness:.1f}")
                    return result, result['name']
            
            # AUTO-LOCK - If face is 25% or more of frame (too close), lock system
            if is_too_close:
                # Set persistent locked state
                self.screen_detected_state[matched_id] = True
                self.screen_detection_counter[matched_id] = 0
                
                result['state'] = 'spoof'
                result['name'] = f"TOO CLOSE ({face_size_ratio:.0%})"
                self.trigger_alert('spoof', f"Face too close - Distance:{distance_category} Size:{face_size_ratio:.0%}", result)
                print(f"  TOO CLOSE LOCK | {distance_category} ({face_size_ratio:.1%})")
                print(f"  State LOCKED - Awaiting real person at normal distance")
                return result, result['name']
            
            # REJECT if screen brightness detected (rolling average)
            if avg_brightness >= brightness_threshold:  # Calibrated threshold for screen detection
                result['state'] = 'spoof'
                result['name'] = f"Screen Brightness Detected"
                self.trigger_alert('spoof', f"Screen brightness - Avg:{avg_brightness:.1f} (threshold:{brightness_threshold:.2f})", result)
                print(f"  SCREEN BRIGHTNESS | Avg:{avg_brightness:.1f} Threshold:{brightness_threshold:.2f}")
                return result, result['name']
            
            # If spoof score is low, accept the face
            # Higher tolerance for appearance changes
            if len(self.known_face_encodings) > 0:
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.55)
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
                        future = self.executor.submit(self.process_single_face, frame, face_encoding, face_location, idx, rgb_small_frame)
                        futures.append(future)
                    
                    # Collect results
                    for future in as_completed(futures):
                        result, name = future.result()
                        self.detection_results.append(result)
                        self.face_names.append(name)
                else:
                    # Single face - process directly
                    for idx, (face_encoding, face_location) in enumerate(zip(self.face_encodings, self.face_locations)):
                        result, name = self.process_single_face(frame, face_encoding, face_location, idx, rgb_small_frame)
                        self.detection_results.append(result)
                        self.face_names.append(name)
                
                # Update previous locations
                with self.lock:
                    self.prev_locations = temp_prev_locations
                    
                    # Update face history - keep faces for persistence
                    current_time = time.time()
                    
                    # Add/update current detections with debug info
                    for idx, (face_location, name) in enumerate(zip(self.face_locations, self.face_names)):
                        face_key = f"persistent_face_{idx}"
                        # Store debug info with face history
                        debug_info = {}
                        if idx < len(self.detection_results):
                            debug_info = self.detection_results[idx].get('debug', {})
                        
                        self.face_history[face_key] = {
                            'location': face_location,
                            'name': name,
                            'timestamp': current_time,
                            'debug': debug_info
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
                        debug = face_data.get('debug', {})
                        
                        # Scale up coordinates
                        top *= 4
                        right *= 4
                        bottom *= 4
                        left *= 4
                        
                        # Enhanced visual feedback with better colors and status indicators
                        if "PHOTO" in name or "Screen" in name or "SPOOF" in name or "LOCKED" in name or "TOO CLOSE" in name or "Phone" in name:
                            color = (0, 0, 255)  # Red for spoof/blocked
                            status_icon = "BLOCKED"
                            box_thickness = 3
                        elif "Move" in name or "BLINK" in name or "Verifying" in name:
                            color = (0, 165, 255)  # Orange for verification
                            status_icon = "VERIFY"
                            box_thickness = 2
                        elif "Unknown" in name:
                            color = (255, 165, 0)  # Blue-orange for unknown
                            status_icon = "UNKNOWN"
                            box_thickness = 2
                        else:
                            color = (0, 255, 0)  # Green for verified
                            status_icon = "OK"
                            box_thickness = 3
                        
                        # Draw main rectangle with variable thickness
                        cv2.rectangle(frame, (left, top), (right, bottom), color, box_thickness)
                        
                        # Draw corner accents for modern look
                        corner_length = 20
                        # Top-left
                        cv2.line(frame, (left, top), (left + corner_length, top), color, box_thickness + 1)
                        cv2.line(frame, (left, top), (left, top + corner_length), color, box_thickness + 1)
                        # Top-right
                        cv2.line(frame, (right, top), (right - corner_length, top), color, box_thickness + 1)
                        cv2.line(frame, (right, top), (right, top + corner_length), color, box_thickness + 1)
                        # Bottom-left
                        cv2.line(frame, (left, bottom), (left + corner_length, bottom), color, box_thickness + 1)
                        cv2.line(frame, (left, bottom), (left, bottom - corner_length), color, box_thickness + 1)
                        # Bottom-right
                        cv2.line(frame, (right, bottom), (right - corner_length, bottom), color, box_thickness + 1)
                        cv2.line(frame, (right, bottom), (right, bottom - corner_length), color, box_thickness + 1)
                        
                        # Draw enhanced name label with icon
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.7
                        font_thickness = 2
                        label_text = f"{status_icon} {name}"
                        
                        # Calculate text size for proper background
                        text_size = cv2.getTextSize(label_text, font, font_scale, font_thickness)[0]
                        label_height = text_size[1] + 16
                        
                        # Draw semi-transparent background
                        overlay = frame.copy()
                        cv2.rectangle(overlay, (left, bottom - label_height), (right, bottom), color, cv2.FILLED)
                        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
                        
                        # Draw text with shadow for better readability
                        text_y = bottom - 8
                        # Shadow
                        cv2.putText(frame, label_text, (left + 9, text_y + 1), font, font_scale, (0, 0, 0), font_thickness)
                        # Main text
                        cv2.putText(frame, label_text, (left + 8, text_y), font, font_scale, (255, 255, 255), font_thickness)
                        
                        # Draw enhanced debug info above the face
                        if debug:
                            y_offset = top - 15
                            font_scale = 0.5
                            thickness = 1
                            
                            # Show key metrics with color coding
                            metrics = []
                            bio_details = debug.get('biometric_details', {})
                            
                            if 'avg_brightness' in bio_details:
                                brightness = bio_details['avg_brightness']
                                abs_bright = bio_details.get('avg_absolute_brightness', 0)
                                rel_bright = bio_details.get('avg_relative_brightness', 0)
                                bright_color = (0, 0, 255) if brightness > 2.0 else (255, 255, 0)
                                metrics.append((f"B:{brightness:.1f}(A:{abs_bright:.1f}+R:{rel_bright:.1f})", bright_color))
                            if 'distance_category' in bio_details:
                                dist = bio_details['distance_category']
                                dist_color = (0, 255, 0) if dist == 'normal' else (0, 165, 255)
                                metrics.append((f"Dist: {dist}", dist_color))
                            
                            # Draw each metric with styled background
                            for metric_text, metric_color in metrics:
                                text_size = cv2.getTextSize(metric_text, font, font_scale, thickness)[0]
                                # Semi-transparent dark background
                                overlay = frame.copy()
                                cv2.rectangle(overlay, (left, y_offset - text_size[1] - 6), 
                                            (left + text_size[0] + 8, y_offset + 2), (0, 0, 0), -1)
                                cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
                                # Colored text
                                cv2.putText(frame, metric_text, (left + 4, y_offset - 2), 
                                          font, font_scale, metric_color, thickness)
                                y_offset -= (text_size[1] + 10)
            
            return frame
            
        def cleanup(self):
            """Clean up thread pool resources"""
            self.executor.shutdown(wait=True)
            
        def run_recognition(self):
            """Run face recognition using local video capture"""
            video_capture = cv2.VideoCapture(0)
            
            if not video_capture.isOpened():
                sys.exit("Error: Could not open video.")
            
            # Apply locked camera settings from calibration
            if hasattr(self, 'camera_profile') and self.camera_profile.get('settings_locked', False):
                print("\n🔒 Applying locked camera settings...")
                
                # Disable auto exposure
                try:
                    video_capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
                    video_capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
                except:
                    pass
                
                # Apply locked exposure
                if 'locked_exposure' in self.camera_profile:
                    video_capture.set(cv2.CAP_PROP_EXPOSURE, self.camera_profile['locked_exposure'])
                    print(f"  ✓ Exposure locked: {self.camera_profile['locked_exposure']}")
                
                # Disable auto white balance
                try:
                    video_capture.set(cv2.CAP_PROP_AUTO_WB, 0)
                except:
                    pass
                
                # Apply locked brightness
                if 'locked_brightness' in self.camera_profile:
                    video_capture.set(cv2.CAP_PROP_BRIGHTNESS, self.camera_profile['locked_brightness'])
                    print(f"  ✓ Brightness locked: {self.camera_profile['locked_brightness']}")
                
                # Apply locked gain
                if 'locked_gain' in self.camera_profile:
                    video_capture.set(cv2.CAP_PROP_GAIN, self.camera_profile['locked_gain'])
                    print(f"  ✓ Gain locked: {self.camera_profile['locked_gain']}")
                
                print("  ✅ Camera settings locked - brightness will not auto-adjust\n")
            
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
                    
                    # Display enhanced FPS counter
                    fps_text = f"FPS: {fps_display:.1f}"
                    fps_color = (0, 255, 0) if fps_display > 20 else (0, 165, 255) if fps_display > 10 else (0, 0, 255)
                    
                    # Background for FPS
                    fps_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (5, 5), (fps_size[0] + 20, fps_size[1] + 20), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
                    
                    # FPS text
                    cv2.putText(frame, fps_text, (12, 28), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, fps_color, 2)
                    
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
    
    #test comment