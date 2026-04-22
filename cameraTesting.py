import cv2
print("Testing ALL backends...")

backends = [
    ('CAP_ANY', cv2.CAP_ANY),
    ('CAP_MSMF', cv2.CAP_MSMF),    # Windows Media Foundation (best for laptops)
    ('CAP_DSHOW', cv2.CAP_DSHOW),  # DirectShow
    ('CAP_VFW', cv2.CAP_VFW)
]

for name, backend in backends:
    print(f"\n=== Backend: {name} ===")
    for i in range(5):
        cap = cv2.VideoCapture(i, backend)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"  ✓ Camera {i}: {w}x{h}")
            cap.release()
            break
        else:
            print(f"  ✗ Camera {i}: FAILED")
        cap.release()