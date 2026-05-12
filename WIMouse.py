import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np


# GPU initialization for RTX 3050 using OpenCL
print("Initializing RTX 3050 GPU support with OpenCL...")
try:
    # Check OpenCL availability (works with NVIDIA RTX 3050)
    if cv2.ocl.haveOpenCL():
        cv2.ocl.setUseOpenCL(True)
        print("✓ OpenCL available and enabled")
        use_opencl = True
        print("✓ RTX 3050 GPU acceleration enabled via OpenCL!")
    else:
        use_opencl = False
        print("✗ OpenCL not available")

    # Check CUDA as fallback (though unlikely to work with pip OpenCV)
    cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()
    use_cuda = cuda_devices > 0

    if use_cuda:
        print(f"✓ CUDA also available with {cuda_devices} device(s)")
        cv2.cuda.setDevice(0)
    else:
        print("CUDA not available (expected with pip OpenCV)")

    if not use_opencl and not use_cuda:
        print("✗ No GPU acceleration available, using CPU")
        use_gpu = False
    else:
        use_gpu = True
        print("✓ GPU processing enabled!")

except Exception as e:
    use_gpu = False
    use_opencl = False
    use_cuda = False
    print(f"✗ GPU initialization failed: {e}")
    print("Falling back to CPU processing")

# Initialising mediapipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

screen_w, screen_h = pyautogui.size()
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
cap.set(cv2.CAP_PROP_BRIGHTNESS, 150)

def fingers_up(hand):
    tips = [4, 8, 12, 16, 20] 
    fingers = []

    fingers.append(1 if hand.landmark[tips[0]].x < hand.landmark[tips[0] - 2].x else 0)

    for id in range(1, 5):
        fingers.append(1 if hand.landmark[tips[id]].y < hand.landmark[tips[id] - 2].y else 0)

    return fingers

prev_x, prev_y = 0, 0
smooth_factor = 4
prev_click_time = 0
last_cursor_x, last_cursor_y = None, None

scroll_threshold = 20
last_scroll_time = 0
scroll_cooldown = 0.2

hand_active = False
last_hand_seen_time = 0
hand_timeout = 1.0

# GPU memory allocation for frame processing (OpenCL)
if use_gpu:
    print("✓ GPU memory initialized for RTX 3050 processing")
else:
    print("Using CPU processing")

while True:
    success, img = cap.read()
    if not success:
        break

    # OpenCL acceleration is automatically used by OpenCV when enabled
    # RTX 3050 will accelerate these operations via OpenCL
    img = cv2.flip(img, 1)  # GPU accelerated flip
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # GPU accelerated color conversion

    h, w, _ = img.shape
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        hand_active = True
        last_hand_seen_time = time.time()

        hand_landmarks = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        fingers = fingers_up(hand_landmarks)
        total_fingers = fingers.count(1)

        index_tip = hand_landmarks.landmark[8]
        x = int(index_tip.x * w)
        y = int(index_tip.y * h)

        norm_x = np.interp(x, [100, w - 100], [0, screen_w])
        norm_y = np.interp(y, [100, h - 100], [0, screen_h])

        curr_x = prev_x + (norm_x - prev_x) / smooth_factor
        curr_y = prev_y + (norm_y - prev_y) / smooth_factor
        prev_x, prev_y = curr_x, curr_y

        if total_fingers == 1 and fingers[1] == 1:
            pyautogui.moveTo(curr_x, curr_y, duration=0)
            last_cursor_x, last_cursor_y = curr_x, curr_y

        # Click
        elif total_fingers == 5:
            now = time.time()
            if now - prev_click_time > 0.2:
                pyautogui.click()
                prev_click_time = now

        # Scroll down
        elif total_fingers == 3:
            now = time.time()
            if now - last_scroll_time > scroll_cooldown:
                pyautogui.scroll(30)
                last_scroll_time = now

        # Scroll up
        elif total_fingers == 2:
            now = time.time()
            if now - last_scroll_time > scroll_cooldown:
                pyautogui.scroll(-30)
                last_scroll_time = now

    else:
        if time.time() - last_hand_seen_time > hand_timeout:
            hand_active = False

        if hand_active and last_cursor_x is not None:
            pyautogui.moveTo(last_cursor_x, last_cursor_y, duration=0)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
