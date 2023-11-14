import mediapipe as mp
import cv2
from utils import *
import json
import os

if __name__ == "__main__":
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    cap = cv2.VideoCapture(4)
    hands_keypoints = []
    translation_flag = False
    display_flag = False
    with mp_hands.Hands(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = cv2.flip(image, 1)
            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                if translation_flag and len(results.multi_handedness) == 2:
                    keypoints_on_frame = []
                    left_hand, right_hand = False, False
                    for num_hand, hand_type in enumerate(results.multi_handedness):
                        hand_score = hand_type.classification[0].score
                        # hand type 0: Left / 1: Right
                        hand_type = hand_type.classification[0].index
                        if hand_score < 0.8:
                            continue
                        if int(hand_type) == 0:
                            left_hand = True
                        if int(hand_type) == 1:
                            right_hand = True
                        keypoints_on_frame.extend(
                            landmarkxy2list(results.multi_hand_landmarks[num_hand])
                        )
                    if left_hand and right_hand:
                        hands_keypoints.append(keypoints_on_frame)

                for num, hand in enumerate(results.multi_hand_landmarks):
                    mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS)

            cv2.imshow("Hand Tracking", image)
            key_queue = cv2.waitKey(1)
            if (key_queue & 0xFF == ord("s")) and translation_flag == False:
                print("Start recording")
                translation_flag = True
                key_queue = 0

            if (key_queue & 0xFF == ord("e")) and translation_flag == True:
                print("Save record...")
                translation_flag = False
                print("End recording")
                key_queue = 0
                num = "10"
                dir_path = "/home/jaehyeong/Sign-Language-Project/our_data/"
                fn = f"kp_{num}_" + str(len(os.listdir(dir_path + num)) + 1) + ".json"
                print(fn)
                with open(f"our_data/{num}/" + fn, "w") as f:
                    json.dump({"data": hands_keypoints}, f)
                hands_keypoints = []

            if key_queue & 0xFF == ord("q"):
                break
