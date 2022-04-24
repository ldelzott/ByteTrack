import PySimpleGUI as sg
from yolox.swarm_metrics.swarm_metrics_utils import get_color
import PIL
from PIL import Image, ImageTk
import cv2
import numpy as np


class SWARMMetricsGUI(object):
    def __init__(self, width, height):
        self.swarm_metric = None
        self.sg_window = None
        self.width = width
        self.height = height
        self.heatmap_blend_coefficient = 0.5
        self.comm_map_blend_coefficient = 0.5
        self.hide_sliders_values = False
        self.max_total_trail_points = 8000
        self.current_frame_entity_list = []
        self.selected_entity = 1
        self.new_entity_toggle = False

    def SWARMMetricGUI_init(self):
        if self.swarm_metric is None:
            print("A SWARMMetric object should be known prior to GUI init")
            exit(0)
        self.init_gui()

    def init_gui(self):
        sg.theme('DefaultNoMoreNagging')
        image_col = [[sg.Image(size=(self.width, self.height), key='-IMAGE-')]]
        control_tab1 = [[sg.Text('', pad=(15, 0))],
                        [sg.Checkbox('Trails', default=True, key="-checkbox2-")],
                        [sg.Checkbox('Global center', default=True, key="-checkbox3-")],
                        [sg.Checkbox('Moving average of global center', default=True, key="-checkbox4-")],
                        [sg.Checkbox('Individuals velocity vectors', default=True, key="-checkbox5-")],
                        [sg.Checkbox('Highlight fastest entity', default=True, key="-checkbox6-")],
                        [sg.Checkbox('Global velocity', default=True, key="-checkbox7-")],
                        [sg.Checkbox('Moving average of global velocity', default=True, key="-checkbox8-")],
                        [sg.Checkbox('Moving average of fastest entity', default=True, key="-checkbox9-")],
                        [sg.Checkbox('Display HUD infos', default=True, key="-checkbox10-")]]
        control_tab2 = [[sg.Text('', pad=(15, 0))],
                        [sg.Checkbox('Show global heatmap    ', default=False, key="-tab2_checkbox1-")],
                        [sg.Checkbox('Show individual heatmap', default=False, key="-tab2_checkbox2-")],
                        [sg.Text('   Blending factor', pad=(15, 0)),
                         sg.Slider(range=(0, 1), default_value=0.5, disable_number_display=self.hide_sliders_values, resolution=0.05,
                                   orientation="horizontal", visible=True, key="-tab2_slider1-", size=(25, 10),
                                   border_width=1, pad=(0, 15))],
                        [sg.Text('   Mask size    ', pad=(15, 0)),
                         sg.Slider(range=(0, 250),
                                   default_value=self.swarm_metric.individual_object_disk_size_in_heatmap,
                                   disable_number_display=self.hide_sliders_values, resolution=5,
                                   orientation="horizontal", visible=True, key="-tab2_slider2-", size=(25, 10),
                                   border_width=1, pad=(15, 15))],
                        [sg.Text('   Choose one entity    ', pad=(15, 0)),
                         sg.Combo([], size=(10, 1), key='-ENTITYSELECT-')],
                        [sg.Text('   Current selection: ' + str(self.selected_entity) + '', pad=(15, 0),
                                 key='-ENTITYSELECTION-')],
                        [sg.Text('', pad=(120, 0)), sg.Button('Select')]]
        control_tab3 = [[sg.Text('', pad=(15, 0))],
                        [sg.Checkbox('Show network areas', default=False, key="-tab3_checkbox1-")],
                        [sg.Text('   Blending factor', pad=(15, 0)),
                         sg.Slider(range=(0, 1), default_value=0.5, disable_number_display=self.hide_sliders_values, resolution=0.05,
                                   orientation="horizontal", visible=True, key="-tab3_slider2-", size=(25, 10),
                                   border_width=1, pad=(0, 15))],
                        [sg.Text('   Radius       ', pad=(15, 0)),
                         sg.Slider(range=(0, 250), default_value=self.swarm_metric.object_disk_size_in_heatmap,
                                   disable_number_display=self.hide_sliders_values, resolution=5,
                                   orientation="horizontal", visible=True, key="-tab3_slider1-", size=(25, 10),
                                   border_width=1, pad=(15, 15))]]
        control_tab4 = [[sg.Text('', pad=(15, 0))],
                        [sg.Slider(range=(0, 200), default_value=self.swarm_metric.max_queue_size,
                                   disable_number_display=self.hide_sliders_values, resolution=1,
                                   orientation="horizontal", visible=True, key="-tab4_slider1-", size=(15, 10),
                                   border_width=1, pad=(15, 15)), sg.Text('max_queue_size', pad=(5, 0))],
                        [sg.Slider(range=(0, 200), default_value=self.swarm_metric.moving_average_fastest_entity,
                                   disable_number_display=self.hide_sliders_values, resolution=1,
                                   orientation="horizontal", visible=True, key="-tab4_slider2-", size=(15, 10),
                                   border_width=1, pad=(15, 15)), sg.Text('moving_average_fastest_entity', pad=(5, 0))],
                        [sg.Slider(range=(0, 200), default_value=self.swarm_metric.moving_average_global_center,
                                   disable_number_display=self.hide_sliders_values, resolution=1,
                                   orientation="horizontal", visible=True, key="-tab4_slider3-", size=(15, 10),
                                   border_width=1, pad=(15, 15)),sg.Text('moving_average_global_center', pad=(5, 0))],
                        [sg.Slider(range=(0, 60), default_value=self.swarm_metric.velocity_vector_scale,
                                   disable_number_display=self.hide_sliders_values, resolution=1,
                                   orientation="horizontal", visible=True, key="-tab4_slider4-", size=(15, 10),
                                   border_width=1, pad=(15, 15)), sg.Text('velocity_vector_scale', pad=(5, 0))],
                        [sg.Slider(range=(0, self.swarm_metric.max_trail_size), default_value=self.max_total_trail_points,
                                   disable_number_display=self.hide_sliders_values, resolution=20,
                                   orientation="horizontal", visible=True, key="-tab4_slider5-", size=(15, 10),
                                   border_width=1, pad=(15, 15)),sg.Text('max_total_trail_points', pad=(5, 0))]
                        ]
        tab_col = [[sg.TabGroup(
            [[sg.Tab('Metric Settings', control_tab1, border_width=0,
                     tooltip='Metrics', element_justification='topleft'),
              sg.Tab('Heatmaps', control_tab2, element_justification='topleft'),
              sg.Tab('Networks', control_tab3, element_justification='topleft'),
              sg.Tab('Core Settings', control_tab4, element_justification='topleft')]],
            tab_location='topleft')],[sg.Button('Resume', disabled=False),sg.Button('Pause', disabled=True),sg.Text('', pad=(35,0)),sg.Button('Abort & Close')]]
        layout = [[sg.Column(image_col), sg.VSeperator(), sg.Column(tab_col)]]

        self.sg_window = sg.Window('Swarm metrics on ByteTrack', layout, no_titlebar=True, alpha_channel=1, grab_anywhere=True,
                                   margins=(0, 0))

    def set_swarm_metric(self, swarm_metric):
        self.swarm_metric = swarm_metric

    """
    This function will add new detected entity to the list of known entities. This is used, for example, when 
    selecting an entity in the combo menu of the tab 'heatmaps'
    """

    def generate_current_frame_entity_list(self):
        for detected_entity in self.swarm_metric.metric_persistent_stack[0]:
            if len(detected_entity) > 0:
                if detected_entity[0] not in self.current_frame_entity_list:
                    self.current_frame_entity_list.append(detected_entity[0])
                    self.new_entity_toggle = True

    def check_gui_event(self, timer, frame_id, objects_count):
        self.generate_current_frame_entity_list()
        event, values = self.sg_window.read(timeout=1)

        self.manage_tab4(event, values)
        self.manage_tab3(event, values)
        self.manage_tab2(event, values)
        self.manage_tab1(event, values, timer, frame_id, objects_count)
        self.manage_global(event, values)

        if event == sg.WIN_CLOSED or event == 'Abort & Close':
            self.sg_window.close()
            exit(0)

        image = ImageTk.PhotoImage(
            image=PIL.Image.fromarray(cv2.cvtColor(self.swarm_metric.tracking_datas[-1][0], cv2.COLOR_BGR2RGB)))
        self.sg_window['-IMAGE-'].update(data=image)

    def manage_global(self, event, values):
        if event == 'Resume':
            self.swarm_metric.paused = False
            self.sg_window['Resume'].update(disabled=True)
            self.sg_window['Pause'].update(disabled=False)
        if event == 'Pause':
            self.swarm_metric.paused = True
            self.sg_window['Resume'].update(disabled=False)
            self.sg_window['Pause'].update(disabled=True)

    def manage_tab4(self, event, values):
        if (values["-tab4_slider1-"] < values["-tab4_slider2-"]) or (values["-tab4_slider1-"] < values["-tab4_slider3-"]):
            self.sg_window['-tab4_slider2-'].update(value=int(values["-tab4_slider1-"]))
            self.sg_window['-tab4_slider3-'].update(value=int(values["-tab4_slider1-"]))
        self.swarm_metric.max_queue_size = int(values["-tab4_slider1-"])
        self.swarm_metric.moving_average_fastest_entity = int(values["-tab4_slider2-"])
        self.swarm_metric.moving_average_global_center = int(values["-tab4_slider3-"])
        self.swarm_metric.velocity_vector_scale = int(values["-tab4_slider4-"])
        self.max_total_trail_points = int(values["-tab4_slider5-"])

    def manage_tab3(self, event, values):
        if values["-tab3_checkbox1-"] is True:
            self.draw_metric_8_networks()
            self.swarm_metric.object_disk_size_in_heatmap = int(values["-tab3_slider1-"])

        self.comm_map_blend_coefficient = values["-tab3_slider2-"]

    def manage_tab2(self, event, values):
        if values["-tab2_checkbox1-"] is True:
            self.draw_metric_10_global_heatmap()

        if values["-tab2_checkbox2-"] is True:
            self.draw_metric_9_individual_heatmap()
            self.swarm_metric.individual_object_disk_size_in_heatmap = int(values["-tab2_slider2-"])

        if event == 'Select':
            self.selected_entity = values['-ENTITYSELECT-']
            self.sg_window['-ENTITYSELECTION-'].update(value='   Current selection: ' + str(self.selected_entity) + '')

        if self.new_entity_toggle:
            self.sg_window['-ENTITYSELECT-'].update(value='', values=self.current_frame_entity_list)
            self.new_entity_toggle = False

        self.heatmap_blend_coefficient = values["-tab2_slider1-"]

    def manage_tab1(self, event, values, timer, frame_id, objects_count):

        if values["-checkbox2-"] is True:
            self.draw_metric_x_trails()
        if values["-checkbox3-"] is True:
            self.draw_metric_1_global_center()
        if values["-checkbox4-"] is True:
            self.draw_metric_2_mean_global_center()
        if values["-checkbox5-"] is True:
            self.draw_metric_3_individual_velocity_vectors()
        if values["-checkbox6-"] is True:
            self.draw_metric_4_fastest_entity()
        if values["-checkbox7-"] is True:
            self.draw_metric_5_global_velocity()
        if values["-checkbox8-"] is True:
            self.draw_metric_6_mean_global_velocity()
        if values["-checkbox9-"] is True:
            self.draw_metric_7_mean_fastest_entity()
        if values["-checkbox10-"] is True:
            self.add_hud_infos(timer, frame_id, objects_count)

    def refresh_gui(self, timer, frame_id, objects_count):
        self.check_gui_event(timer, frame_id, objects_count)

    def draw_metric_10_global_heatmap(self):
        heatmap_non_normalized = np.zeros_like(self.swarm_metric.tracking_datas[-1][0][:, :, 0]).astype("float")
        if 1 >= self.heatmap_blend_coefficient >= 0:
            for individual_heatmap in self.swarm_metric.metric_persistent_stack[0]:
                heatmap_non_normalized += individual_heatmap[1].copy()
            cv2.normalize(heatmap_non_normalized, heatmap_non_normalized, 0.0, 255.0, cv2.NORM_MINMAX)
            heatmap_mask = np.uint8(heatmap_non_normalized)
            heatmap_output = cv2.applyColorMap(heatmap_mask, cv2.COLORMAP_JET)
            self.swarm_metric.tracking_datas[-1][0] = cv2.addWeighted(heatmap_output,
                                                                      self.heatmap_blend_coefficient,
                                                                      self.swarm_metric.tracking_datas[-1][0],
                                                                      1 - self.heatmap_blend_coefficient,
                                                                      0)

    def persistent_stack_0_get_index_of_ID(self):
        index = 0
        for object_id in self.swarm_metric.metric_persistent_stack[0]:
            if object_id[0] == self.selected_entity:
                return index
            index += 1

    def draw_metric_9_individual_heatmap(self):
        if 1 >= self.heatmap_blend_coefficient >= 0:
            heatmap_non_normalized = \
            self.swarm_metric.metric_persistent_stack[0][self.persistent_stack_0_get_index_of_ID()][1].copy()
            cv2.normalize(heatmap_non_normalized, heatmap_non_normalized, 0.0, 255.0, cv2.NORM_MINMAX)
            heatmap_mask = np.uint8(heatmap_non_normalized)
            heatmap_output = cv2.applyColorMap(heatmap_mask, cv2.COLORMAP_JET)
            self.swarm_metric.tracking_datas[-1][0] = cv2.addWeighted(heatmap_output,
                                                                      self.heatmap_blend_coefficient,
                                                                      self.swarm_metric.tracking_datas[-1][0],
                                                                      1 - self.heatmap_blend_coefficient,
                                                                      0)

    def draw_metric_8_networks(self):
        # Draw metric 8 - map of networks
        # https://docs.opencv.org/3.4/d5/dc4/tutorial_adding_images.html
        # https://stackoverflow.com/questions/46441893/connected-component-labeling-in-python
        if 1 >= self.comm_map_blend_coefficient >= 0:
            network_binary_mask = self.swarm_metric.metric_main_stack[7][-1].copy()
            num_labels, labels_im = cv2.connectedComponents(network_binary_mask)
            label_hue = np.uint8(179 * labels_im / np.max(labels_im))
            blank_ch = 255 * np.ones_like(label_hue)
            labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
            labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
            labeled_img[label_hue == 0] = 0
            self.swarm_metric.tracking_datas[-1][0] = cv2.addWeighted(labeled_img,
                                                                      self.comm_map_blend_coefficient,
                                                                      self.swarm_metric.tracking_datas[-1][0],
                                                                      1 - self.comm_map_blend_coefficient,
                                                                      0)

    def draw_metric_x_trails(self):
        # Drawing trails
        trail_length = len(self.swarm_metric.objects_trails)
        if trail_length > 0:
            i = 0
            for colored_point in reversed(self.swarm_metric.objects_trails):
                if i > self.max_total_trail_points:
                    break;
                center = tuple(map(int, colored_point[0]))
                colour = get_color(abs( colored_point[1]))
                i += 1
                cv2.circle(self.swarm_metric.tracking_datas[-1][0], center, radius=1, color=colour, thickness=-1)

    def draw_metric_1_global_center(self):
        # Draw metric 1 - global center
        immediate_global_center = tuple(map(int, self.swarm_metric.metric_main_stack[0][-1]))
        cv2.circle(self.swarm_metric.tracking_datas[-1][0], immediate_global_center, radius=5, color=(0, 0, 255),
                   thickness=7)

    def draw_metric_2_mean_global_center(self):
        # Draw metric 2 - mean global center
        mean_global_center = tuple(map(int, self.swarm_metric.metric_main_stack[1][-1]))
        cv2.circle(self.swarm_metric.tracking_datas[-1][0], mean_global_center, radius=5, color=(0, 255, 0),
                   thickness=7)

    def draw_metric_3_individual_velocity_vectors(self):
        # Draw metric 3 - velocity vectors of each moving entity
        for entity in self.swarm_metric.metric_main_stack[2][-1][1]:
            entity_color = get_color(abs(entity[3]))
            entity_center = tuple(map(int, entity[0]))
            entity_velocity_vector_tip = [entity[0][0] + entity[1][0] * self.swarm_metric.velocity_vector_scale,
                                          entity[0][1] + entity[1][1] * self.swarm_metric.velocity_vector_scale]
            entity_velocity_vector_tip = tuple(map(int, entity_velocity_vector_tip))
            cv2.line(self.swarm_metric.tracking_datas[-1][0], entity_center, entity_velocity_vector_tip, entity_color,
                     2)
            cv2.circle(self.swarm_metric.tracking_datas[-1][0], entity_center, radius=1, color=(0, 255, 0), thickness=2)

    def draw_metric_4_fastest_entity(self):
        # Draw metric 4 - fastest entity
        fastest_entity = self.swarm_metric.metric_main_stack[3][-1]
        if fastest_entity[0] is not None:
            cv2.circle(self.swarm_metric.tracking_datas[-1][0], fastest_entity[2], radius=15, color=(0, 0, 255),
                       thickness=2)

    def draw_metric_5_global_velocity(self):
        # Draw metric 5 - global velocity
        if self.swarm_metric.metric_main_stack[4][-1] is not None:
            global_velocity_vector_tip = tuple(
                map(int, (
                    self.swarm_metric.metric_main_stack[0][-1][
                        0] + self.swarm_metric.velocity_vector_scale * self.swarm_metric.metric_main_stack[4][-1][0],
                    self.swarm_metric.metric_main_stack[0][-1][
                        1] + self.swarm_metric.velocity_vector_scale * self.swarm_metric.metric_main_stack[4][-1][1])))
            cv2.line(self.swarm_metric.tracking_datas[-1][0],
                     tuple(map(int, self.swarm_metric.metric_main_stack[0][-1])), global_velocity_vector_tip,
                     (0, 0, 255), 3)

    def draw_metric_6_mean_global_velocity(self):
        # Draw metric 6 - mean global velocity
        mean_global_velocity_deltas = self.swarm_metric.metric_main_stack[5][-1]
        if mean_global_velocity_deltas is not None:
            vector_tail = tuple(map(int, self.swarm_metric.metric_main_stack[1][-1]))
            vector_tip = tuple(
                map(int, (
                    vector_tail[0] + self.swarm_metric.velocity_vector_scale *
                    self.swarm_metric.metric_main_stack[5][-1][
                        0],
                    vector_tail[1] + self.swarm_metric.velocity_vector_scale *
                    self.swarm_metric.metric_main_stack[5][-1][
                        1])))
            cv2.line(self.swarm_metric.tracking_datas[-1][0], vector_tail, vector_tip, (0, 255, 0), 3)

    def draw_metric_7_mean_fastest_entity(self):
        # Draw metric 7 - fastest entity
        fastest_mean_entity = self.swarm_metric.metric_main_stack[6][-1]
        text_scale = 2
        if fastest_mean_entity[0] is not None:
            cv2.circle(self.swarm_metric.tracking_datas[-1][0], tuple(map(int, fastest_mean_entity[2])), radius=10,
                       color=(0, 255, 0), thickness=2)
            cv2.putText(self.swarm_metric.tracking_datas[-1][0], 'mean_fastest: %d' % (fastest_mean_entity[0]),
                        (int(fastest_mean_entity[2][0]) + 20, int(fastest_mean_entity[2][1]) - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), thickness=1)

    def add_hud_infos(self, timer, frame_id, objects_count):
        timer.toc()
        text_scale = 1
        fps = 1. / timer.average_time
        cv2.putText(self.swarm_metric.tracking_datas[-1][0],
                    'frame: %d fps: %.2f num: %d' % (frame_id, fps, objects_count), (0, int(30 * text_scale)),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)
        cv2.putText(self.swarm_metric.metric_main_stack[7][-1],
                    'frame: %d fps: %.2f num: %d' % (frame_id, fps, objects_count), (0, int(30 * text_scale)),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)
