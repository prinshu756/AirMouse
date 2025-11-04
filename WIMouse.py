# import cv2 
# import mediapipe as mp
# # import time
# import pyautogui

# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils

# green = (50, 255, 0)
# landmark_spec = mp_drawing.DrawingSpec(color=green, thickness=3, circle_radius=1)
# connection_spec = mp_drawing.DrawingSpec(color=green, thickness=2, circle_radius=1)
# cap = cv2.VideoCapture(0)
# cap.set(3, 640)
# cap.set(4, 480)
# cap.set(cv2.CAP_PROP_BRIGHTNESS , 160)

# prev_time = 0


# # while cap.isOpened():
# #     success , image = cap.read()
# #     if not success :
# #         continue

# #     image = cv2.cvtColor(cv2.flip(image , 1), cv2.COLOR_BGR2RGB)
# #     image.flags.writeable = False
# #     result = hands.process(image)

# #     h, w, c = image.shape

# #     if result.multi_hand_landmarks and result.multi_handedness:
# #         for hand_landmarks  in result.multi_hand_landmarks:
# #             mp_draw.draw_landmarks(image , hand_landmarks , mp_hands.HAND_CONNECTIONS , mp_draw.DrawingSpec(color = (0 , 255 , 0), thickness = 3 , circle_radius = 2),
# #                                    mp_draw.DrawingSpec(color=(255 , 0 ,0 ), thickness = 1 , circle_radius = 1))
# #             index_finger_tip = hand_landmarks.landmark[8]
# #             thumb_tip = hand_landmarks.landmark[4]

# #             x , y = int(index_finger_tip.x*w) , int(index_finger_tip.y*h)
# #             thumbOfX , thumbOfY = int(thumb_tip.x*w) , int(thumb_tip.y*h)

# #             cv2.circle(image, (x, y), 10 , (255 , 0 , 255), cv2.FILLED)

# #             screen_x = int(thumb_tip.x*w)
# #             screen_y = int(thumb_tip.y*h)

# #             pyautogui.moveTo(screen_x*2 ,screen_y*2 ,duration=0.05)

# #             thumb_x = int(thumb_tip.x*w)
# #             thumb_y = int(thumb_tip.y*h)

# #             cv2.circle(image, (thumb_x , thumb_y), 10 , (0 , 255 , 255) , cv2.FILLED)

# #             distance = ((thumb_x - x)**2 + (thumb_y - y)**2)**0.5
# #             if distance<40:
# #                 pyautogui.click()
# #                 cv2.putText(image , 'CLICKED!!', (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1 , (0 , 255 , 0))
# #                 print("Clcked at somewhere")

# with mp_hands.Hands(
#     static_image_mode = False,
#     max_num_hands = 2,
#     min_detection_confidence = 0.7,
#     min_tracking_confidence = 0.7
# ) as hands:
#     while cap.isOpened():
#         success , image = cap.read()
#         if not success:
#             continue 
#         image = cv2.cvtColor(cv2.flip(image , 1) , cv2.COLOR_RGB2BGR)
#         image.flags.writeable = False
#         results = hands.process(image)

#         image.flags.writeable = True
#         image = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)

#         h , w , c = image.shape

#         if results.multi_hand_landmarks and results.multi_handedness :
#             for hand_landmarks ,  handedness  in zip(results.multi_hand_landmarks , results.multi_handedness):
#                 mp_drawing.draw_landmarks(
#                     image,
#                     hand_landmarks,
#                     mp_hands.HAND_CONNECTIONS,
#                     landmark_drawing_spec = landmark_spec,
#                     connection_drawing_spec = connection_spec
#                 )
#             index_finger_tip = hand_landmarks.landmark[8]
#             thumb_tip = hand_landmarks.landmark[4]

#             x , y = int(index_finger_tip.x*w) , int(index_finger_tip.y*h)
#             thumbOfX , thumbOfY = int(thumb_tip.x*w) , int(thumb_tip.y*h)

#             cv2.circle(image, (x, y), 10 , (255 , 0 , 255), cv2.FILLED)

#             screen_x = int(index_finger_tip.x*w)
#             screen_y = int(index_finger_tip.y*h)

           

#             # pyautogui.moveTo(screen_x*2 ,screen_y*2 ,duration=0.05)

#             thumb_x = int(thumb_tip.x*w)
#             thumb_y = int(thumb_tip.y*h)

#             cv2.circle(image, (thumb_x , thumb_y), 10 , (0 , 255 , 255) , cv2.FILLED)

#             distance = ((thumb_x - x)**2 + (thumb_y - y)**2)**0.5
#             pyautogui.FAILSAFE = False
#             pyautogui.moveTo(screen_x*2 ,screen_y*2 ,duration=0.005)
#             if distance<80:
#                 pyautogui.click()
#                 cv2.putText(image , 'CLICKED!!', (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1 , (0 , 255 , 0))
#                 print("Clcked at somewhere")
#         cv2.imshow("AIRmouse", image)
#         if cv2.waitKey(5) & 0xFF == ord('q'):
#             break

# cap.release()
# cv2.destroyAllWindows()

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
cap.set(cv2.CAP_PROP_BRIGHTNESS , 200)

pev_time = 0 

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
            cv2.circle(img , (x, y ) , 10 , (255 , 0 , 25 ) , cv2.FILLED)

            # getting fingers 
            fingers = fingers_up(hand_landmarks)
            total_fingers = fingers.count(1)

            #motion of cursor logic

            if total_fingers == 1 and fingers[1] == 1 :
                screen_x = int(index_tip.x*screen_w) 
                screen_y = int(index_tip.y*screen_h)
                pyautogui.moveTo(screen_x , screen_y , duration = 0.01)
                pyautogui.FAILSAFE = False
                # cv2.putText(img , "It is moving!" , (10,70) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (0 , 255 , 25), 2)
                # print("Your index is moving")

            elif total_fingers == 5 :
                pyautogui.click()
                time.sleep(0.8) #preventing double click
                # print("I have clicked it!")
                
            elif total_fingers == 2 and fingers[1] == 1 and fingers[2] == 1 :
                pyautogui.scroll(-15)
                print("Scrolling in upward direction")
            elif total_fingers == 3 and fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1 :
                pyautogui.scroll(15)
                # print("Scrolling in downward direction!")
        cv2.imshow("Hand Gesture Mouse" , img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()