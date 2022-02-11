import os.path
from tinydb import TinyDB, Query

import numpy as np
import cv2


def dump_annotated_images(image, tlwhs, obj_ids, save_folder, current_time, args, frame_id=0, fps=0.):
    frame_name = str(frame_id)+".jpg"
    annotation_name = str(frame_id)+".json"
    json_dump_file = TinyDB(os.path.join(save_folder, annotation_name))
    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        json_dump_file.insert({'frame_id': frame_id, 'track_id': i, 'x1': x1, 'y1': y1, 'w': w, 'h': h})
    cv2.imwrite(os.path.join(save_folder, frame_name), image)
    cv2.waitKey(0)


def dump_non_annotated_images(image, save_folder, current_time, args, frame_id):
    frame_name = str(frame_id) + ".jpg"
    print(os.path.join(save_folder, frame_name))
    cv2.imwrite(os.path.join(save_folder, frame_name), image)
    cv2.waitKey(0)
    print("Hello - dump_non_annotated_images")
    print(save_folder)
    print(current_time)

def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color

