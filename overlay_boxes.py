# overlay_boxes.py

import cv2

def draw_boxes(image_path, boxes, color=(0, 0, 255), thickness=2):
    img = cv2.imread(image_path)
    for box in boxes:
        cv2.rectangle(img,
                      (box['xmin'], box['ymin']),
                      (box['xmax'], box['ymax']),
                      color,
                      thickness)
    return img

