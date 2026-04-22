import cv2
import numpy as np
import math
import random
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

# Camera fix for ThinkPad
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '1'

# CORRECT MediaPipe Tasks imports (works with latest)
from mediapipe.tasks import python as tasks
BaseOptions = tasks.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode
Image = mp.Image

# Initialize
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
landmarker = HandLandmarker.create_from_options(options)

# Visuals setup
WIDTH, HEIGHT = 1280, 720
NUM_PARTICLES, NUM_STARS = 800, 120
particles, stars = [], []

for _ in range(NUM_STARS):
    stars.append({'x': random.randint(0,WIDTH), 'y': random.randint(0,HEIGHT),
                  'vx': random.uniform(-0.3,0.3), 'vy': random.uniform(-0.3,0.3),
                  'size': random.uniform(0.5,1.5), 'brightness': random.random()})

def init_particles():
    global particles
    particles = [{'x': random.uniform(0,WIDTH), 'y': random.uniform(0,HEIGHT),
                  'vx': random.uniform(-0.5,0.5), 'vy': random.uniform(-0.5,0.5),
                  'life': 0, 'size': random.uniform(1,3), 'hue': random.uniform(200,280)}
                 for _ in range(NUM_PARTICLES)]
init_particles()

# Camera (multi-strategy)
cap = None
for backend in [cv2.CAP_MSMF, cv2.CAP_ANY]:
    for i in range(3):
        test_cap = cv2.VideoCapture(i, backend)
        if test_cap.isOpened():
            print(f"Camera OK: index={i}, backend={backend}")
            cap = test_cap
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
            break
    if cap: break

if not cap or not cap.isOpened():
    print("NO CAMERA DETECTED. Use USB webcam.")
    exit(1)

def detect_gesture(landmarks):
    if len(landmarks) < 21: return 'none'
    fingers = 0
    if landmarks[4].x < landmarks[3].x: fingers += 1  # Thumb
    for t, p in zip([8,12,16,20], [5,9,13,17]):
        if landmarks[t].y < landmarks[p].y: fingers += 1
    return {5:'spread', 0:'fist', 1:'point'}.get(fingers, 'open')

def get_hand_center(landmarks):
    coords = np.array([[lm.x*WIDTH, lm.y*HEIGHT] for lm in landmarks])
    return tuple(np.mean(coords, 0).astype(int))

def hsv2bgr(h, s, v):
    # Clamp ALL inputs 100% safe
    h = min(359, max(0, int(float(h)) % 360))
    s = min(255, max(0, int(float(s))))
    v = min(255, max(0, int(float(v))))
    
    # Pure Python HSV→RGB (no numpy!)
    h, s, v = h / 360.0, s / 255.0, v / 255.0
    c = v * s
    x = c * (1 - abs((h * 6) % 2 - 1))
    m = v - c
    
    if 0 <= h < 1/6: r, g, b = c, x, 0
    elif 1/6 <= h < 2/6: r, g, b = x, c, 0
    elif 2/6 <= h < 3/6: r, g, b = 0, c, x
    elif 3/6 <= h < 4/6: r, g, b = 0, x, c
    elif 4/6 <= h < 5/6: r, g, b = x, 0, c
    else: r, g, b = c, 0, x
    
    return (int((b + m) * 255), int((g + m) * 255), int((r + m) * 255))

t = 0
while True:
    ret, frame = cap.read()
    if not ret: continue
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # FIXED Image creation
    image_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    
    # Process
    timestamp = int(1000 * time.time())
    results = landmarker.detect_for_video(image_mp, timestamp)
    
    cx, cy, gesture = WIDTH//2, HEIGHT//2, 'none'
    if results.hand_landmarks:
        hlm = results.hand_landmarks[0]
        cx, cy = get_hand_center(hlm)
        gesture = detect_gesture(hlm)
        for lm in hlm:
            cv2.circle(frame, (int(lm.x*WIDTH), int(lm.y*HEIGHT)), 3, (0,255,0), -1)
    
    # Stars
    for star in stars:
        dx, dy = star['x']-cx, star['y']-cy
        d = math.hypot(dx, dy)
        if d > 1:
            f = max(0, (80-d/8)/80) * (2 if gesture=='spread' else 1)
            if gesture=='fist': f *= -2
            star['vx'] += (dx/d)*f*0.08
            star['vy'] += (dy/d)*f*0.08
        star['x'] += star['vx']; star['y'] += star['vy']
        star['vx'] *= 0.97; star['vy'] *= 0.97
        star['x'] = max(0, min(WIDTH, star['x']))
        star['y'] = max(0, min(HEIGHT, star['y']))
        pulse = 1 + math.sin(t*0.08 + star['brightness']*6)*0.4

        h_val = int((240 + d * 0.05) % 360)
        v_raw = 60 + pulse * 25
        v_val = int(min(255, max(0, 60 + pulse * 25))) 
        print(f"DEBUG v_raw={v_raw}, v_val={v_val}")
        color = hsv2bgr(h_val, 90, v_val)
        print("color: ", color)

        cv2.circle(frame, (int(star['x']), int(star['y'])), int(star['size']*pulse*2), color, -1)
    
    # Particles
    r = 180
    for p in particles:
        dx, dy = p['x']-cx, p['y']-cy
        d = math.hypot(dx, dy)
        if d < r:
            f = (r-d)/r * (2.5 if gesture=='spread' else 1)
            pull = -4 if gesture=='fist' else 2.5
            p['vx'] += (dx/d)*f*pull if d else 0
            p['vy'] += (dy/d)*f*pull if d else 0
            p['hue'] += f*25
        p['vx'] *= 0.96; p['vy'] *= 0.96
        p['x'] += p['vx']; p['y'] += p['vy']
        p['life'] += 0.008; p['size'] *= 0.99
        if p['life'] > 1:
            p.update({'x': random.uniform(0,WIDTH), 'y': random.uniform(0,HEIGHT),
                     'vx': random.uniform(-0.3,0.3), 'vy': random.uniform(-0.3,0.3), 'life': 0, 'size': random.uniform(1,3)})
        s_val = max(0, 85 - p['life'] * 30)
        v_val = int(min(255, max(0, 55 + (1 - p['life']) * 30)))
        color = hsv2bgr(int(p['hue'] % 360), s_val, v_val)

        cv2.circle(frame, (int(p['x']), int(p['y'])), int(p['size']*1.5), color, -1)
    
    # Effects + UI
    cv2.circle(frame, (cx, cy), int(100 + math.sin(t*0.06)*25), (120,220,255), 2)
    cv2.circle(frame, (WIDTH//2, HEIGHT//2), 350, (80,40,160), -1)
    cv2.putText(frame, f"{gesture.upper()}", (10,35), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)
    
    cv2.imshow('SPACE HAND NEBULA', frame)
    t += 1
    if cv2.waitKey(1) == ord('q'): break

cap.release()
cv2.destroyAllWindows()