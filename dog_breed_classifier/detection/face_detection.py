from pathlib import Path

import cv2
import cv2.cv2 as cv

face_detector = cv.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def detect_human_face(img_path: Path) -> bool:
    """
    Checks whether the passed image contains a human face or not.
    :param img_path: path to the image.
    :return: whether the passed image contains a human face or not.
    """
    img = cv.imread(str(img_path))
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray)
    return len(faces) > 0
