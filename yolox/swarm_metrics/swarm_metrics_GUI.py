import PySimpleGUI as sg
from yolox.swarm_metrics.swarm_metrics_utils import get_color
import PIL
from PIL import Image, ImageTk
import cv2


class SWARMMetricsGUI(object):
    def __init__(self, width, height):
        self.swarm_metric = None
        self.sg_window = None
        self.width = width
        self.height = height
        self.heatmap_blend_coefficient = 0.5

    def SWARMMetricGUI_init(self):
        if self.swarm_metric is None:
            print("A SWARMMetric object should be known prior to GUI init")
            exit(0)
        self.init_gui()

    def init_gui(self):
        sg.theme('DefaultNoMoreNagging')
        image_col = [[sg.Image(size=(self.width, self.height), key='-IMAGE-')]]
        control_col = [[sg.Text('Metric Settings')],
                       [sg.Checkbox('Position heatmap', default=True, key="-checkbox1-")],
                       [sg.Text('   Mask size    ', pad=(15,0)), sg.Slider(range=(0, 500), default_value=60, disable_number_display=True, resolution=50, orientation="horizontal", visible=True, key="-slider1-", size=(25, 10), border_width=1, pad=(15,15))],
                       [sg.Text('   Blending factor', pad=(15, 0)),
                        sg.Slider(range=(0, 1), default_value=0.5, disable_number_display=True, resolution=0.05,
                                  orientation="horizontal", visible=True, key="-slider2-", size=(25, 10),
                                  border_width=1, pad=(0, 15))],
                       [sg.Checkbox('Trails', default=True, key="-checkbox2-")],
                       [sg.Checkbox('Global center', default=True, key="-checkbox3-")],
                       [sg.Checkbox('Moving average of global center', default=True, key="-checkbox4-")],
                       [sg.Checkbox('Individuals velocity vectors', default=True, key="-checkbox5-")],
                       [sg.Checkbox('Highlight fastest entity', default=True, key="-checkbox6-")],
                       [sg.Checkbox('Global velocity', default=True, key="-checkbox7-")],
                       [sg.Checkbox('Moving average of global velocity', default=True, key="-checkbox8-")],
                       [sg.Checkbox('Moving average of fastest entity', default=True, key="-checkbox9-")],
                       [sg.Checkbox('Display HUD infos', default=True, key="-checkbox10-")],
                       [sg.Button('Button'), sg.Button('Exit')]]
        layout = [[sg.Column(image_col), sg.VSeperator(), sg.Column(control_col)]]
        self.sg_window = sg.Window('My new window', layout, no_titlebar=True, alpha_channel=1, grab_anywhere=True, margins=(0,0))

    def set_swarm_metric(self, swarm_metric):
        self.swarm_metric = swarm_metric

    def check_gui_event(self, timer, frame_id, objects_count):
        # print("check gui even")
        event, values = self.sg_window.read(timeout=1)  # Read the event that happened and the values dictionary
        # print(event, values)
        if event == sg.WIN_CLOSED or event == 'Exit':  # If user closed window with X or if user clicked "Exit" button then exit
            self.sg_window.close()
            exit(0)
        if event == 'Button':
            print('You pressed the button')
        if values["-checkbox1-"] is True:
            self.draw_metric_8_heatmap()
            self.sg_window['-slider1-'].update(disabled=False)
            self.sg_window['-slider2-'].update(disabled=False)
            self.swarm_metric.object_disk_size_in_heatmap = int(values["-slider1-"])
            self.heatmap_blend_coefficient = values["-slider2-"]
        else:
            self.sg_window['-slider1-'].update(disabled=True)
            self.sg_window['-slider2-'].update(disabled=True)
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

        image = ImageTk.PhotoImage(
            image=PIL.Image.fromarray(cv2.cvtColor(self.swarm_metric.tracking_datas[-1][0], cv2.COLOR_BGR2RGB)))
        self.sg_window['-IMAGE-'].update(data=image)

    def refresh_gui(self, timer, frame_id, objects_count):
        # self.refresh_window(timer, frame_id, objects_count)
        self.check_gui_event(timer, frame_id, objects_count)

    # def refresh_window(self, timer, frame_id, objects_count):
    # self.draw_metric_8_heatmap()

    def draw_metric_8_heatmap(self):
        # Draw metric 8 - heatmap
        # https://docs.opencv.org/3.4/d5/dc4/tutorial_adding_images.html
        if 1 >= self.heatmap_blend_coefficient >= 0:
            self.swarm_metric.tracking_datas[-1][0] = cv2.addWeighted(self.swarm_metric.metric_main_stack[7][-1],
                                                                      self.heatmap_blend_coefficient,
                                                                      self.swarm_metric.tracking_datas[-1][0],
                                                                      1 - self.heatmap_blend_coefficient,
                                                                      0)

    def draw_metric_x_trails(self):
        # Drawing trails
        trail_length = len(self.swarm_metric.objects_trails)
        if trail_length > 0:
            for i in range(trail_length):
                center = tuple(map(int, self.swarm_metric.objects_trails[i][0]))
                colour = get_color(abs(self.swarm_metric.objects_trails[i][1]))
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
