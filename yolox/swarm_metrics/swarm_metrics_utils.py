import base64
import json
import os.path
from tinydb import TinyDB
import cv2
import numpy as np


'''
This function can generate output images, output tracking datas or both.
The generated .json files (one per output image) follows the format used by "Labelme" annotation/visualization tool.
'''
def dump_annotated_images(image, tlwhs, obj_ids, save_folder, current_time, args, frame_id=0, fps=0.):
    if args.dump_tracking:
        im = np.ascontiguousarray(np.copy(image))
        retval, buffer = cv2.imencode('.jpg',im)  # https://stackoverflow.com/questions/40928205/python-opencv-image-to-byte-string-for-json-transfer
        jpg_in_ascii = base64.b64encode(buffer)
        im_h, im_w = im.shape[:2]
        frame_name = str(frame_id) + ".jpg"
        annotation_name = str(frame_id) + ".json"
        shape_list = []
        for i, tlwh in enumerate(tlwhs):
            x1, y1, w, h = tlwh
            bounding_box_dict = {'label': str(i), 'points': [[x1, y1], [x1 + w, y1 + h]], 'group_id': None,
                                 'shape_type': 'rectangle', 'flags': {}}
            shape_list.append(bounding_box_dict)
        json_dump_data = {'version': '4.6.0', 'flags': {}, 'shapes': shape_list, 'imagePath': frame_name,
                          'imageData': jpg_in_ascii.decode(), 'imageHeight': im_h, 'imageWidth': im_w}
        with open(os.path.join(save_folder, annotation_name), 'w') as f:
            json.dump(json_dump_data, f)
    if args.dump_images:
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


def dump_swarm_metrics(visualization_folder, frame_id, metric_1, metric_2, metric_5, metric_6):
    if metric_1 is not None:
        metric_1_0 = metric_1[0]
        metric_1_1 = metric_1[1]
    else:
        metric_1_0 = 'empty'
        metric_1_1 = 'empty'

    if metric_2 is not None:
        metric_2_0 = metric_2[0]
        metric_2_1 = metric_2[1]
    else:
        metric_2_0 = 'empty'
        metric_2_1 = 'empty'

    if metric_5 is not None:
        metric_5_0 = metric_5[0]
        metric_5_1 = metric_5[1]
    else:
        metric_5_0 = 'empty'
        metric_5_1 = 'empty'

    if metric_6 is not None:
        metric_6_0 = metric_6[0]
        metric_6_1 = metric_6[1]
    else:
        metric_6_0 = 'empty'
        metric_6_1 = 'empty'

    file_name = "metrics_dump.json"
    json_dump_file = TinyDB(os.path.join(visualization_folder, file_name))
    json_dump_file.insert({'frame_id': frame_id, 'metric_1_0': metric_1_0, 'metric_1_1': metric_1_1,
                           'metric_2_0': metric_2_0, 'metric_2_1': metric_2_1,
                           'metric_5_0': metric_5_0, 'metric_5_1': metric_5_1,
                           'metric_6_0': metric_6_0, 'metric_6_1': metric_6_1})


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color
