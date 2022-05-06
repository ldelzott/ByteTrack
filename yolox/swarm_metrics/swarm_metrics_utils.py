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
        # https://stackoverflow.com/questions/40928205/python-opencv-image-to-byte-string-for-json-transfer
        im = np.ascontiguousarray(np.copy(image))
        retval, buffer = cv2.imencode('.jpg',im)
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


def dump_frame_swarm_metrics_in_database(swarm_metric):
    file_name = "metrics_database.json"
    json_dump_file = TinyDB(os.path.join(swarm_metric.visualization_folder, file_name))

    global_immediate_metrics_table = json_dump_file.table('global_immediate_metrics')
    global_mean_metrics_table = json_dump_file.table('global_mean_metrics')
    individual_metrics_table = json_dump_file.table('individual_metrics')

    dump_list_individual_metric = []

    dump_dict_immediate_metric = {'frameID': swarm_metric.frame_id,
                                  'global_center_x': swarm_metric.metric_main_stack[0][-1][0],
                                  'global_center_y': swarm_metric.metric_main_stack[0][-1][1],
                                  'global_velocity_delta_x': swarm_metric.metric_main_stack[4][-1][0],
                                  'global_velocity_delta_y': swarm_metric.metric_main_stack[4][-1][1],
                                  'fastest_entity_ID': swarm_metric.metric_main_stack[3][-1][0],
                                  'fastest_entity_velocity_norm': swarm_metric.metric_main_stack[3][-1][1],
                                  'fastest_entity_pos_x': swarm_metric.metric_main_stack[3][-1][2][0],
                                  'fastest_entity_pos_y': swarm_metric.metric_main_stack[3][-1][2][1]}
    dump_dict_mean_metric = {'frameID': swarm_metric.frame_id,
                             'mean_global_center_x': swarm_metric.metric_main_stack[1][-1][0],
                             'mean_global_center_y': swarm_metric.metric_main_stack[1][-1][1],
                             'mean_global_velocity_delta_x': swarm_metric.metric_main_stack[4][-1][0],
                             'mean_global_velocity_delta_y': swarm_metric.metric_main_stack[4][-1][1],
                             'mean_fastest_entity_ID': swarm_metric.metric_main_stack[6][-1][0],
                             'mean_fastest_entity_velocity_norm': swarm_metric.metric_main_stack[6][-1][1],
                             'mean_fastest_entity_pos_x': swarm_metric.metric_main_stack[6][-1][2][0],
                             'mean_fastest_entity_pos_y': swarm_metric.metric_main_stack[6][-1][2][1],
                             'mean_global_center_window_size': swarm_metric.moving_average_global_center,
                             'mean_global_velocity_window_size': swarm_metric.moving_average_global_velocity,
                             'mean_fastest_entity_window_size': swarm_metric.moving_average_fastest_entity}

    for entity_data in swarm_metric.metric_main_stack[2][-1][1]:
        dump_dict_individual_metric = {'frameID': swarm_metric.frame_id, 'entity_id': entity_data[3],
                                       'entity_pos_x': entity_data[0][0], 'entity_pos_y': entity_data[0][1],
                                       'entity_velocity_delta_x': entity_data[1][0],
                                       'entity_velocity_delta_y': entity_data[1][1],
                                       'entity_velocity_norm': entity_data[2]}
        dump_list_individual_metric.append(dump_dict_individual_metric)

    global_immediate_metrics_table.insert(dump_dict_immediate_metric)
    global_mean_metrics_table.insert(dump_dict_mean_metric)
    individual_metrics_table.insert_multiple(dump_list_individual_metric)


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color
