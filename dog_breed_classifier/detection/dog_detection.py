from pathlib import Path

import numpy as np
from tensorflow.keras.applications.nasnet import NASNetMobile, preprocess_input
from tensorflow.keras.preprocessing import image

classifier = NASNetMobile(include_top=True, weights='imagenet')


def detect_dog(img_path: Path) -> bool:
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(str(img_path)))
    prediction = np.argmax(classifier.predict(img))
    return (prediction <= 268) & (prediction >= 151)


def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)
