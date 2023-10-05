import mediapipe as mp
import cv2
import numpy as np
import os
import time

from visualization import render_animation

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

pre_time = 0

hand_keypoints = []

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:

    frame = cv2.imread('data/image44669.jpg')
        
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    image.flags.writeable = False

    results = hands.process(image)
    hand_keypoints.append(results)

    image.flags.writeable = True

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    if results.multi_hand_landmarks:
        print(results.multi_hand_landmarks[0])
        for num, hand in enumerate(results.multi_hand_landmarks):
            mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS)              
            
    curr_time = time.time()
    t = curr_time - pre_time
    fps = 1 / t

    pre_time = curr_time
    
    cv2.putText(image, f'fps: {fps}', (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))

    cv2.imshow('Hand Tracking', image)