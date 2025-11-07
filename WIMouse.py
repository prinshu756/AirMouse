import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# for using and changing the screen size
screen_w, screen_h = pyautogui.size()

# for the camera setup
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
cap.set(cv2.CAP_PROP_BRIGHTNESS, 150)

#function for checking which fingers are up and which are down
def fingers_up(hand):
    tips = [4, 8, 12, 16, 20]
    fingers = []

    # for detecting thumb
    if hand.landmark[tips[0]].x < hand.landmark[tips[0] - 2].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # for detecting other fingers
    for id in range(1, 5):
        if hand.landmark[tips[id]].y < hand.landmark[tips[id] - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers

# for smoothing the cursor and avoiding the jitter
smooth_x, smooth_y = 0, 0
alpha = 0.25  
#lower the value of alpha better will be the smoothing but it will also add lag sometimes

prev_click_time = 0

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    h, w, c = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            #for index finger tip
            index_tip = hand_landmarks.landmark[8]
            x, y = int(index_tip.x * w), int(index_tip.y * h)
            cv2.circle(img, (x, y), 10, (255, 0, 75), cv2.FILLED)

            # for getting other fingers status
            fingers = fingers_up(hand_landmarks)
            total_fingers = fingers.count(1)

            # Map camera coordinates to screen coordinates
            screen_x = np.interp(index_tip.x, [0.1, 0.9], [0, screen_w])
            screen_y = np.interp(index_tip.y, [0.1, 0.9], [0, screen_h])

            # smoothnening the movement
            smooth_x = smooth_x + alpha * (screen_x - smooth_x)
            smooth_y = smooth_y + alpha * (screen_y - smooth_y)

            pyautogui.FAILSAFE = False

            # Cursor move (index finger only)
            if total_fingers == 1 and fingers[1] == 1:
                pyautogui.moveTo(smooth_x, smooth_y, duration=0)

            # Click (all fingers open)
            elif total_fingers == 5:
                now = time.time()
                if now - prev_click_time > 0.4:  # avoid double click
                    pyautogui.click()
                    prev_click_time = now

            # Scroll down (2 fingers)
            elif total_fingers == 2 and fingers[1] == 1 and fingers[2] == 1:
                pyautogui.scroll(-40)

            # Scroll up (3 fingers)
            elif total_fingers == 3 and fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1:
                pyautogui.scroll(40)

    cv2.imshow("Virtual Mouse", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
