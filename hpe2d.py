import mediapipe as mp
import cv2
import numpy as np
import os
import time

import json

from visualization import render_animation

def display_fps(pre_time, image):
    curr_time = time.time()
    t = curr_time - pre_time
    fps = 1 / t
    
    cv2.putText(image, f'fps: {fps}', (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
    return curr_time

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

pre_time = 0

hands_keypoints = []
translation_flag = True
data_on_frame = []

cnt = 0

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image.flags.writeable = False

        results = hands.process(image)

        image.flags.writeable = True

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # pre_time = display_fps(pre_time, image)   # fps 표기
        
        if results.multi_hand_landmarks:
            if translation_flag:
                for num_hand, classification in enumerate(results.multi_handedness):
                    if num_hand == 0:
                        data_on_frame.append([classification, results.multi_hand_landmarks[0:21]])
                    else:
                        data_on_frame[-1].extend([classification, results.multi_hand_landmarks[21:]])
                hands_keypoints.append(data_on_frame)

            # print(results.multi_handedness)
            for num, hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS)
                if cnt == 0:
                    cnt+=1
                    print(hand)
                    print('ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ')         
                    print(hands_keypoints)


        

        cv2.imshow('Hand Tracking', image)

        if cv2.waitKey(10) & 0xFF == ord('s'):
            print('Start recording')
            translation_flag = True
        
        if cv2.waitKey(10) & 0xFF == ord('e'):
            translation_flag = False
            # sign_language_translator(hands_keypoints)
            print('End recording')

            with open('keypoints.json', 'w') as f:
                json.dump({'keypoint_sequence' : hands_keypoints}, f, default=str, indent='\t')
            hands_keypoints = []

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break