import cv2
import pyautogui
import time
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands = 1 ,
                       min_detection_confidence = 0.7,
                       min_tracking_confidence = 0.7)

screen_w , screen_h = pyautogui.size()

#for camera resolution and brightness
cap = cv2.VideoCapture(0)
cap.set(3 , 640)
cap.set(4 , 460)
cap.set(cv2.CAP_PROP_BRIGHTNESS , 150)

def fingers_up(hands_landmarks):
    tips = [4 , 8 , 12 , 16 , 20]
    fingers = []

    # thumb
    if hands_landmarks.landmark[tips[0]].x < hands_landmarks.landmark[tips[0] - 2].x :
        fingers.append(1)
    else:
        fingers.append(0)

    #for other fingers
    for id in range(1,5):
        if hands_landmarks.landmark[tips[id]].y < hands_landmarks.landmark[tips[id] - 2 ].y :
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers

while True:
    success , img = cap.read()
    if not success:
        continue

    img = cv2.flip(img , 1)   #miroring for right movement

    h , w , c  =  img.shape
    img_rgb = cv2.cvtColor(img ,cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks :
        for hand_landmarks in result.multi_hand_landmarks :
            mp_draw.draw_landmarks(img , hand_landmarks , mp_hands.HAND_CONNECTIONS)

            #getting index finger
            index_tip = hand_landmarks.landmark[8]
            x , y = int(index_tip.x*w) , int(index_tip.y*h)

            #drawing point for index 
            cv2.circle(img , (x, y ) , 10 , (255 , 0 , 75 ) , cv2.FILLED)

            # getting fingers 
            fingers = fingers_up(hand_landmarks)
            total_fingers = fingers.count(1)

            #motion of cursor logic

            if total_fingers == 1 and fingers[1] == 1 :
                screen_x = int(index_tip.x*screen_w) 
                screen_y = int(index_tip.y*screen_h)
                pyautogui.moveTo(screen_x , screen_y , duration = 0.05)
                pyautogui.FAILSAFE = False
                # cv2.putText(img , "It is moving!" , (10,70) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (0 , 255 , 25), 2)
                # print("Your index is moving")

            elif total_fingers == 5 :
                pyautogui.click()
                time.sleep(0.3) #preventing double click
                # print("I have clicked it!")
                
            elif total_fingers == 2 and fingers[1] == 1 and fingers[2] == 1 :
                pyautogui.scroll(-15)
            elif total_fingers == 3 and fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1 :
                pyautogui.scroll(15)
                # print("Scrolling in downward direction!")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
