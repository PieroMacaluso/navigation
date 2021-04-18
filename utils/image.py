import cv2


def process_state(x, image_size=32):
    return cv2.resize(x.squeeze(), dsize=(image_size, image_size))
