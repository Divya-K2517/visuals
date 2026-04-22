"""
Space Hand Nebula - MediaPipe Tasks API (Python 3.13 + MediaPipe 0.10+)
Hand tracking with gestures controlling cosmic particles/stars.
Place 'hand_landmarker.task' in same folder.
"""

import cv2
import mediapipe as mp
import numpy as np
import math
import random
import time

# MediaPipe Tasks API
mp_image = mp.Image
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Initialize landmarker
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7
)

landmarker = HandLandmarker.create_from_options(options)

# Visual parameters
WIDTH, HEIGHT = 1280, 720
NUM_PARTICLES = 1000
NUM_STARS = 150
particles = []
stars = []

# Initialize stars
for _ in range(NUM_STARS):
    stars.append({
        'x': random.randint(0, WIDTH),
        'y': random.randint(0, HEIGHT),
        'vx': random.uniform(-0.5, 0.5),
        'vy': random.uniform(-0.5, 0.5),
        'size': random.uniform(0.5, 2),
        'brightness': random.random()
    })

# Initialize particles
def init_particles():
    global particles
    particles = []
    for _ in range(NUM_PARTICLES):
        particles.append({
            'x': random.uniform(-200, WIDTH + 200),
            'y': random.uniform(-200, HEIGHT + 200),
            'vx': random.uniform(-1, 1),
            'vy': random.uniform(-1, 1),
            'life': random.random(),
            'size': random.uniform(1, 4),
            'hue': random.uniform(200, 260)
        })
init_particles()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

def detect_gesture(landmarks):
    """Detect hand gesture from landmarks"""
    if not landmarks or len(landmarks) < 21:
        return 'none'
    
    fingers_up = 0
    # Thumb
    if landmarks[4].x < landmarks[3].x:
        fingers_up += 1
    # Other fingers
    tip_ids = [8, 12, 16, 20]
    pip_ids = [5, 9, 13, 17]
    for tip, pip in zip(tip_ids, pip_ids):
        if landmarks[tip].y < landmarks[pip].y:
            fingers_up += 1
    
    if fingers_up >= 4:
        return 'spread'
    elif fingers_up == 0:
        return 'fist'
    elif fingers_up == 1:
        return 'point'
    return 'open'

def get_hand_center(landmarks):
    """Get average landmark position"""
    if not landmarks or len(landmarks) < 21:
        return None
    coords = np.array([[lm.x * WIDTH, lm.y * HEIGHT] for lm in landmarks])
    return tuple(np.mean(coords, axis=0).astype(int))

def hsv_to_bgr(h, s, v):
    """Convert HSV to BGR for OpenCV"""
    hsv = np.uint8([[[h, s, v]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return tuple(int(c) for c in bgr)

time_global = 0
while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image_obj = mp_image.create_from_rgb(rgb_image)

    # Process frame
    timestamp_ms = int(time.time() * 1000)
    results = landmarker.detect_for_video(mp_image_obj, timestamp_ms)

    # Hand detection
    hand_landmarks = None
    hand_center = (WIDTH // 2, HEIGHT // 2)
    gesture = 'none'

    if results.hand_landmarks:
        hand_landmarks = results.hand_landmarks[0]
        hand_center = get_hand_center(hand_landmarks)
        gesture = detect_gesture(hand_landmarks)

        # Draw hand landmarks
        for lm in hand_landmarks:
            x, y = int(lm.x * WIDTH), int(lm.y * HEIGHT)
            cv2.circle(image, (x, y), 3, (0, 255, 0), -1)

    cx, cy = hand_center

    # Update stars
    for star in stars:
        dx, dy = star['x'] - cx, star['y'] - cy
        dist = math.sqrt(dx**2 + dy**2)
        if dist > 1:
            force = max(0, (100 - dist / 10) / 100)
            if gesture == 'fist':
                force *= -1.5
            warp_x = (dx / dist) * force * (2 if gesture == 'spread' else 1)
            warp_y = (dy / dist) * force * (2 if gesture == 'spread' else 1)
            star['vx'] += warp_x * 0.1
            star['vy'] += warp_y * 0.1

        star['x'] += star['vx']
        star['y'] += star['vy']
        star['vx'] *= 0.98
        star['vy'] *= 0.98
        
        if star['x'] < 0 or star['x'] > WIDTH: star['vx'] *= -1
        if star['y'] < 0 or star['y'] > HEIGHT: star['vy'] *= -1

        pulse = 1 + math.sin(time_global * 0.1 + star['brightness'] * 10) * 0.3
        dist_factor = max(0.1, 1 - dist / 300)
        alpha = star['brightness'] * pulse * dist_factor
        color = hsv_to_bgr(260 + dist * 0.1, 100, 70 + pulse * 20)
        cv2.circle(image, (int(star['x']), int(star['y'])), 
                  int(star['size'] * pulse), color, -1)

    # Update particles
    force_radius = 200
    for p in particles:
        dx, dy = p['x'] - cx, p['y'] - cy
        dist = math.sqrt(dx**2 + dy**2)
        if dist < force_radius:
            force_mag = (force_radius - dist) / force_radius * (3 if gesture == 'spread' else 1)
            dir_x = dx / dist if dist > 0 else 0
            dir_y = dy / dist if dist > 0 else 0
            pull = -5 if gesture == 'fist' else 3
            p['vx'] += dir_x * force_mag * pull
            p['vy'] += dir_y * force_mag * pull
            p['hue'] = (p['hue'] + force_mag * 30) % 360
            p['life'] -= force_mag * 0.02

        p['vx'] *= 0.96
        p['vy'] *= 0.96
        p['x'] += p['vx']
        p['y'] += p['vy']
        p['life'] = min(1, p['life'] + 0.005)
        p['size'] *= 0.995

        if p['life'] > 1 or p['size'] < 0.2 or abs(p['x']) > WIDTH * 2 or abs(p['y']) > HEIGHT * 2:
            p.update({
                'x': random.uniform(0, WIDTH), 'y': random.uniform(0, HEIGHT),
                'vx': random.uniform(-0.5, 0.5), 'vy': random.uniform(-0.5, 0.5),
                'life': 0, 'size': random.uniform(1, 4)
            })

        # Draw particle
        sat = 80 - p['life'] * 20
        val = 60 + (1 - p['life']) * 20
        color = hsv_to_bgr(p['hue'], sat, val)
        cv2.circle(image, (int(p['x']), int(p['y'])), int(p['size'] * 2), color, -1)
        cv2.circle(image, (int(p['x']), int(p['y'])), int(p['size']), (255, 255, 255), -1)

    # Nebula background
    overlay = image.copy()
    cv2.circle(overlay, (WIDTH//2, HEIGHT//2), 400, (100, 50, 200, 0.3*255), -1)
    image[:] = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)

    # Warp ring around hand
    cv2.circle(image, hand_center, 120 + int(math.sin(time_global * 0.05) * 30), (100, 200, 255), 2)

    # HUD
    cv2.putText(image, f"Gesture: {gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(image, "'q' to quit | Hand warps space!", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow('Space Hand Nebula (MediaPipe Tasks)', image)
    time_global += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()