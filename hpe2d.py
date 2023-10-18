import mediapipe as mp
import cv2
import numpy as np
from model_dim import InferModel
import torch
import yaml
from utils import *

from PIL import ImageFont, ImageDraw, Image

if __name__ == "__main__":
    with open("configs/default.yaml") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    num_classes = cfg["num_classes"]
    ckpt_name = cfg["ckpt_name"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    cap = cv2.VideoCapture(0)

    w = 640
    h = 480

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

    ui_img = make_ui_img(w)

    hands_keypoints = []
    translation_flag = False
    korean = [
        "행복",
        "안녕",
        "슬픔",
        "눈",
        "당신",
        "식사하셨어요?",
        "이름이 뭐 예요?"
        #"수어",
        #"사랑",
        #"만나서 반가워요!",
    ]
    model = InferModel().load_from_checkpoint(checkpoint_path=ckpt_name)
    model.eval().to("cuda")
    label_play_time = 0
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

            image = drawUI(image, ui_img, np.zeros((100,w,3),np.uint8))

            if display_flag:
                image = display_label(labels, image, w, h)
                label_play_time += 1

            if label_play_time > 150:
                label_play_time = 0
                display_flag = False

            cv2.imshow("Sign-Language Translator", image)

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
                hands_keypoints = torch.tensor(hands_keypoints)
                frames_len = hands_keypoints.shape[0]
                ids = np.round(np.linspace(0, frames_len - 1, 60))
                keypoint_sequence = []
                for i in range(60):
                    keypoint_sequence.append(
                        hands_keypoints[int(ids[i]), ...].unsqueeze(0)
                    )
                keypoint_sequence = torch.cat(keypoint_sequence, dim=0)
                input_data = keypoint_sequence.unsqueeze(0).to("cuda")
                print(keypoint_sequence.shape)
                output = model(input_data)
                labels = torch.max(output, dim=1)[1][0]
                labels = korean[labels]
                display_flag = True
                hands_keypoints = []

            if key_queue & 0xFF == ord("q"):
                break
