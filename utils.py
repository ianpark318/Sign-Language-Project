import time
import cv2
from mediapipe.framework.formats import landmark_pb2
from PIL import ImageFont, ImageDraw, Image
import numpy as np


fontpath = "NanumBarunpenB.ttf"
font = ImageFont.truetype(fontpath, 50)


def landmarkxy2list(
    landmarks_list: landmark_pb2.NormalizedLandmarkList,
):
    result_list = []
    for hand_num, landmark in enumerate(landmarks_list.landmark):
        landmark_list = [landmark.x, landmark.y]
        result_list.append(landmark_list)
    return result_list


def make_ui_img(w):
    ui_img = np.zeros((100, w, 3), np.uint8)

    b, g, r, a = 255, 255, 255, 0

    ui_img_pil = Image.fromarray(ui_img)
    draw_top = ImageDraw.Draw(ui_img_pil)

    guide_message = "카메라를 향해 수어를 해보세요!"

    draw_top.text((w / 2, 50), guide_message, anchor="mm", font=font, fill=(b, g, r, a))
    ui_img = np.array(ui_img_pil)

    return ui_img


def drawUI(img, ui_img, subtitle_bg):
    stacked_img = cv2.vconcat([ui_img, img])
    stacked_img = cv2.vconcat([stacked_img, subtitle_bg])

    return stacked_img


def display_label(label, image, w, h):
    b, g, r, a = 255, 255, 255, 0

    img_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(img_pil)
    draw.text((w / 2, h + 100 + 50), label, anchor="mm", font=font, fill=(b, g, r, a))
    img = np.array(img_pil)

    return img
