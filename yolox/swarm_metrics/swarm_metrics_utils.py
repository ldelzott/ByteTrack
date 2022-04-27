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

def dump_frame_swarm_metrics_in_database(swarm_metric):
    file_name = "metrics_database.json"
    json_dump_file = TinyDB(os.path.join(swarm_metric.visualization_folder, file_name))

    global_immediate_metrics_table = json_dump_file.table('global_immediate_metrics')
    global_mean_metrics_table = json_dump_file.table('global_mean_metrics')
    individual_metrics_table = json_dump_file.table('individual_metrics')

    dump_dict_immediate_metric = {}
    dump_dict_mean_metric = {}
    dump_list_individual_metric = []

    dump_dict_immediate_metric['frameID'] = swarm_metric.frame_id
    dump_dict_immediate_metric['global_center_x'] = swarm_metric.metric_main_stack[0][-1][0]
    dump_dict_immediate_metric['global_center_y'] = swarm_metric.metric_main_stack[0][-1][1]
    dump_dict_immediate_metric['global_velocity_delta_x'] = swarm_metric.metric_main_stack[4][-1][0]
    dump_dict_immediate_metric['global_velocity_delta_y'] = swarm_metric.metric_main_stack[4][-1][1]
    dump_dict_immediate_metric['fastest_entity_ID'] = swarm_metric.metric_main_stack[3][-1][0]
    dump_dict_immediate_metric['fastest_entity_velocity_norm'] = swarm_metric.metric_main_stack[3][-1][1]
    dump_dict_immediate_metric['fastest_entity_pos_x'] = swarm_metric.metric_main_stack[3][-1][2][0]
    dump_dict_immediate_metric['fastest_entity_pos_y'] = swarm_metric.metric_main_stack[3][-1][2][1]

    dump_dict_mean_metric['frameID'] = swarm_metric.frame_id
    dump_dict_mean_metric['mean_global_center_x'] = swarm_metric.metric_main_stack[1][-1][0]
    dump_dict_mean_metric['mean_global_center_y'] = swarm_metric.metric_main_stack[1][-1][1]
    dump_dict_mean_metric['mean_global_velocity_delta_x'] = swarm_metric.metric_main_stack[4][-1][0]
    dump_dict_mean_metric['mean_global_velocity_delta_y'] = swarm_metric.metric_main_stack[4][-1][1]
    dump_dict_mean_metric['mean_fastest_entity_ID'] = swarm_metric.metric_main_stack[6][-1][0]
    dump_dict_mean_metric['mean_fastest_entity_velocity_norm'] = swarm_metric.metric_main_stack[6][-1][1]
    dump_dict_mean_metric['mean_fastest_entity_pos_x'] = swarm_metric.metric_main_stack[6][-1][2][0]
    dump_dict_mean_metric['mean_fastest_entity_pos_y'] = swarm_metric.metric_main_stack[6][-1][2][1]
    dump_dict_mean_metric['mean_global_center_window_size'] = swarm_metric.moving_average_global_center
    dump_dict_mean_metric['mean_global_velocity_window_size'] = swarm_metric.moving_average_global_velocity
    dump_dict_mean_metric['mean_fastest_entity_window_size'] = swarm_metric.moving_average_fastest_entity

    for entity_data in swarm_metric.metric_main_stack[2][-1][1]:
        dump_dict_individual_metric = {}
        dump_dict_individual_metric['frameID'] = swarm_metric.frame_id
        dump_dict_individual_metric['entity_id'] = entity_data[3]
        dump_dict_individual_metric['entity_pos_x'] = entity_data[0][0]
        dump_dict_individual_metric['entity_pos_y'] = entity_data[0][1]
        dump_dict_individual_metric['entity_velocity_delta_x'] = entity_data[1][0]
        dump_dict_individual_metric['entity_velocity_delta_y'] = entity_data[1][1]
        dump_dict_individual_metric['entity_velocity_norm'] = entity_data[2]
        dump_list_individual_metric.append(dump_dict_individual_metric)
    """
    dump_dict['frame_id'] = frame_id
    for i in range(len(metric_dump)):
        if metric_dump[i] is not None:
            dump_dict['metric_' + str(i) + '_0'] = metric_dump[i][0]
            dump_dict['metric_' + str(i) + '_1'] = metric_dump[i][1]
        else:
            dump_dict['metric_' + str(i) + '_0'] = 'empty'
            dump_dict['metric_' + str(i) + '_1'] = 'empty'
    """
    global_immediate_metrics_table.insert(dump_dict_immediate_metric)
    global_mean_metrics_table.insert(dump_dict_mean_metric)
    individual_metrics_table.insert_multiple(dump_list_individual_metric)

def dump_swarm_metrics(visualization_folder, frame_id, metric_dump, swarm_metric):
    #TODO: Adding proper name to the metric when saving them into the JSON file.
    file_name = "metrics_dump.json"
    json_dump_file = TinyDB(os.path.join(visualization_folder, file_name))
    dump_dict = {}
    dump_dict['frame_id'] = frame_id
    for i in range(len(metric_dump)):
        if metric_dump[i] is not None:
            dump_dict['metric_' + str(i) + '_0'] = metric_dump[i][0]
            dump_dict['metric_' + str(i) + '_1'] = metric_dump[i][1]
        else:
            dump_dict['metric_' + str(i) + '_0'] = 'empty'
            dump_dict['metric_' + str(i) + '_1'] = 'empty'
    json_dump_file.insert(dump_dict)



def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color
