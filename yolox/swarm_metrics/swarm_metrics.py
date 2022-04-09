import numpy as np
from yolox.swarm_metrics.swarm_metrics_utils import get_color, dump_swarm_metrics
import cv2
from collections import deque
import numpy as np
import time
import matplotlib.pyplot as plt


class SWARMMetrics(object):
    def __init__(self, args, vis_folder):
        self.args = args
        self.number_of_metric_queues = 6
        self.number_of_graph_queues = 4
        self.max_trail_size = 8000
        self.current_entity_number = 0
        self.moving_average_items_number = 5 # Should be below or equal to "max_queue_size"
        self.max_queue_size = 7
        self.max_graph_size = 40
        self.frame_count = 0
        self.frame_id = 0
        self.velocity_vector_scale = 2
        self.visualization_folder = vis_folder
        self.tracking_datas = deque(maxlen=self.max_queue_size)
        self.objects_trails = deque(maxlen=self.max_trail_size)
        """
            metric_main_stack[0] = metric_1 : float_global_centers
            metric_main_stack[1] = metric_2 : float_mean_global_centers
            metric_main_stack[2] = metric_3 : individual centers, velocity vectors and norms
            metric_main_stack[3] = metric_3 : fastest_entity
            metric_main_stack[4] = metric_3 : float_global_velocity_deltas
            metric_main_stack[5] = metric_6 : float_mean_global_velocity_deltas
        """
        self.metric_main_stack = []
        for i in range(self.number_of_metric_queues):
            self.metric_main_stack.append(deque(maxlen=self.max_queue_size))
        self.metric_graph_stack = []
        for i in range(self.number_of_graph_queues):
            self.metric_graph_stack.append(deque(maxlen=self.max_graph_size))

        self.figures = [[], []]
        self.launch_graphs()

    def online_metrics_inputs(self, im, im_h, im_w, tlwhs, obj_ids, frame_id=0, timer=None):
        self.tracking_datas.append([im, [im_w, im_h], tlwhs, obj_ids])
        self.frame_id = frame_id
        self.select_online_metrics(timer, frame_id, len(tlwhs))
        self.dump_metrics()
        self.stack_trim()
        # Return an annotated image
        return self.tracking_datas[-1][0]

    def add_hud_infos(self, timer, frame_id, objects_count):
        timer.toc()
        text_scale = 1
        fps = 1. / timer.average_time
        cv2.putText(self.tracking_datas[-1][0], 'frame: %d fps: %.2f num: %d' % (frame_id, fps, objects_count),
                    (0, int(30 * text_scale)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)

    def dump_metrics(self):
        if not self.args.dump_metrics:
            return
        metric_dump = []
        for i in range(self.number_of_graph_queues):
            if len(self.metric_graph_stack[i]) == 0:
                metric_dump.append(None)
            else:
                metric_dump.append(self.metric_graph_stack[i][-1])

        dump_swarm_metrics(self.visualization_folder, self.frame_id, metric_dump)

    def draw_graphics(self):

        # Drawing trails
        trail_length = len(self.objects_trails)
        if trail_length > 0:
            for i in range(trail_length):
                center = tuple(map(int, self.objects_trails[i][0]))
                colour = get_color(abs(self.objects_trails[i][1]))
                cv2.circle(self.tracking_datas[-1][0], center, radius=1, color=colour, thickness=-1)

    """
        The various queues used in SWARMMetrics are limited in size: 'stack_trim' is used to remove the oldest item 
        of a queues when his maximum size is reached.
    """

    def stack_trim(self):
        if len(self.tracking_datas) > self.max_queue_size:
            self.tracking_datas.popleft()
        for i in range(self.number_of_metric_queues):
            if len(self.metric_main_stack[i]) > self.max_queue_size:
                self.metric_main_stack[i].popleft()
        for i in range(self.number_of_graph_queues):
            if len(self.metric_graph_stack[i]) > self.max_graph_size:
                self.metric_graph_stack[i].popleft()
        if len(self.objects_trails) > self.max_trail_size * self.current_entity_number:  # The trail size is defined per object
            self.objects_trails.popleft()

    def select_online_metrics(self, timer, frame_id, objects_count):
        # Add here additional metrics
        # Note that metric_2 require metric_1: see compute_metric_1() function
        # Same comment for metric_4 : see compute_metric_3() function
        # Same comment for metric_5 : see compute_metric_3() function
        # In the current configuration (metric 1 to 6), the computation time for ~100 detected objects
        # is around 2 ms. The drawing task takes around 70ms per frame to complete.

        # t0 = time.time()
        if self.args.swarm_metric_1:
            self.compute_metric_1_immediate_global_center()
        if self.args.swarm_metric_3:
            self.compute_metric_3_velocity_vectors()
        if self.args.swarm_metric_6:
            self.compute_metric_6_global_velocity_vector_moving_average()
        # t1 = time.time()-t0
        # print("Metric computation elapsed time: ", t1)

        self.update_graphs(timer, frame_id, objects_count)

    """
        'Immediate' center of mass location
    """

    def compute_metric_1_immediate_global_center(self, ):
        sum_mass = 0.000001
        sum_x = 0
        sum_y = 0
        # from https://stackoverflow.com/questions/12801400/find-the-center-of-mass-of-points
        for i, tlwh in enumerate(self.tracking_datas[-1][2]):
            x1, y1, w, h = tlwh
            sum_mass += 1
            sum_x += x1 + w / 2
            sum_y += y1 + h / 2
        float_immediate_global_center = [sum_x / sum_mass, sum_y / sum_mass]
        self.metric_main_stack[0].append(float_immediate_global_center)
        self.metric_graph_stack[0].append(float_immediate_global_center)
        center = tuple(map(int, float_immediate_global_center))
        cv2.circle(self.tracking_datas[-1][0], center, radius=5, color=(0, 0, 255), thickness=7)
        if self.args.swarm_metric_2:
            self.compute_metric_2_immediate_global_center_moving_average()

    """
        Moving average of the center masses locations over the last x=max_queue_size frames
    """

    def compute_metric_2_immediate_global_center_moving_average(self, ):
        sum_x = 0
        sum_y = 0
        i = 0
        for center in reversed(self.metric_main_stack[0]):
            if i < self.moving_average_items_number:
                x, y = center
                sum_x += x
                sum_y += y
                i += 1
            else:
                break
        float_mean_global_center = (sum_x / self.moving_average_items_number, sum_y / self.moving_average_items_number)
        mean_center = tuple(map(int, float_mean_global_center))
        self.metric_main_stack[1].append(float_mean_global_center)
        self.metric_graph_stack[1].append(float_mean_global_center)
        cv2.circle(self.tracking_datas[-1][0], mean_center, radius=5, color=(0, 255, 0), thickness=7)

    def find_object_center_and_velocity(self, tlwh_current, tlwh_previous, obj_id_previous):
        x1, y1, w1, h1 = tlwh_current
        x2, y2, w2, h2 = tlwh_previous
        center_current_float = [x1 + w1 / 2, y1 + h1 / 2]
        center_previous_float = [x2 + w2 / 2, y2 + h2 / 2]
        self.objects_trails.append([center_current_float, obj_id_previous])
        center_current = tuple(map(int, center_current_float))
        x_delta_float = center_current_float[0] - center_previous_float[0]
        y_delta_float = center_current_float[1] - center_previous_float[1]
        norm = x_delta_float * x_delta_float + y_delta_float * y_delta_float
        x_delta = int(x_delta_float)
        y_delta = int(y_delta_float)

        vector_tip_float = [center_current_float[0] + x_delta_float * self.velocity_vector_scale,
                            center_current_float[1] + y_delta_float * self.velocity_vector_scale]
        vector_tip = tuple(map(int, vector_tip_float))

        # Draw the trails
        self.draw_graphics()

        # Draw the object center and the object velocity vector
        color = get_color(abs(obj_id_previous))
        cv2.line(self.tracking_datas[-1][0], center_current, vector_tip, color, 2)
        cv2.circle(self.tracking_datas[-1][0], center_current, radius=1, color=(0, 255, 0), thickness=2)
        self.metric_main_stack[2][-1].append([center_current_float, norm, [x_delta_float, y_delta_float]])
        return center_current, norm, x_delta, y_delta

    def find_global_velocity_vector(self, sum_velocity_vector_x, sum_velocity_vector_y, current_entity_number):
        if current_entity_number > 0:
            global_velocity_vector_tip = tuple(
                map(int, (
                    self.metric_main_stack[0][-1][
                        0] + self.velocity_vector_scale * sum_velocity_vector_x / current_entity_number,
                    self.metric_main_stack[0][-1][
                        1] + self.velocity_vector_scale * sum_velocity_vector_y / current_entity_number)))

            if self.args.swarm_metric_5 and self.args.swarm_metric_1:
                instantaneous_center_of_masses = tuple(
                    map(int, (self.metric_main_stack[0][-1][0], self.metric_main_stack[0][-1][1])))

                # Draw the global velocity vector on top of the instantaneous center of mass.
                cv2.line(self.tracking_datas[-1][0], instantaneous_center_of_masses, global_velocity_vector_tip,
                         (0, 0, 255), 3)

                # Store the global velocity vector of the current frame into the corresponding graph stack & main stack.
                float_global_velocity_deltas = [sum_velocity_vector_x / current_entity_number,
                                                sum_velocity_vector_y / current_entity_number]
                self.metric_main_stack[4].append(float_global_velocity_deltas)
                self.metric_graph_stack[2].append(float_global_velocity_deltas)
        else:
            self.metric_main_stack[4].append(None)
            self.metric_graph_stack[2].append((0, 0))


    """
        Raw velocity vectors 
        metric_3 display the velocity vector of each moving entity
        metric_4 highlight the fastest moving entity on a given frame (require metric_3)
        metric_5 compute a global velocity vector; the tail of that vector is given by the inst. center of masse.
    """

    def compute_metric_3_velocity_vectors(self, ):
        sum_velocity_vector_x = 0
        sum_velocity_vector_y = 0
        fastest_entity = [None, 0, [0, 0]]
        tracks_records_number = len(self.tracking_datas)
        self.current_entity_number = len(self.tracking_datas[-1][2])
        self.metric_main_stack[2].append([])

        if tracks_records_number >= 2:
            for i, tlwh_current in enumerate(self.tracking_datas[-1][2]):
                obj_id_current = int(self.tracking_datas[-1][3][i])
                for j, tlwh_previous in enumerate(self.tracking_datas[-2][2]):
                    obj_id_previous = int(self.tracking_datas[-2][3][j])
                    if obj_id_previous == obj_id_current:
                        center_current, norm, x_delta, y_delta = self.find_object_center_and_velocity(tlwh_current,
                                                                                                      tlwh_previous,
                                                                                                      obj_id_previous)
                        sum_velocity_vector_x += x_delta
                        sum_velocity_vector_y += y_delta

                        if self.args.swarm_metric_4 and norm >= fastest_entity[1]:
                            fastest_entity = obj_id_current, norm, center_current

            if fastest_entity[0] is not None:
                cv2.circle(self.tracking_datas[-1][0], fastest_entity[2], radius=10, color=(0, 0, 255), thickness=2)

            self.find_global_velocity_vector(sum_velocity_vector_x, sum_velocity_vector_y, self.current_entity_number)

        self.metric_main_stack[3].append(fastest_entity)

    """
    Moving average of the global velocity vector ; displayed with "moving average of center of masses" as the vector's tail
    """

    def compute_metric_6_global_velocity_vector_moving_average(self, ):
        if len(self.metric_main_stack[4]) > 1:
            sum_dx = 0
            sum_dy = 0
            vector_scale = 2
            metric_5_stack_size = len(self.metric_main_stack[4])
            for _, deltas in enumerate(self.metric_main_stack[4]):
                if deltas is not None:
                    dx, dy = deltas
                    sum_dx += dx
                    sum_dy += dy
            float_mean_global_velocity_deltas = [sum_dx / metric_5_stack_size, sum_dy / metric_5_stack_size]
            self.metric_main_stack[5].append(float_mean_global_velocity_deltas)
            self.metric_graph_stack[3].append(float_mean_global_velocity_deltas)
            vector_tail = tuple(map(int, (self.metric_main_stack[1][-1][0], self.metric_main_stack[1][-1][1])))
            vector_tip = tuple(
                map(int, (vector_tail[0] + vector_scale * self.metric_main_stack[5][-1][0], vector_tail[1]
                          + vector_scale * self.metric_main_stack[5][-1][1])))
            cv2.line(self.tracking_datas[-1][0], vector_tail, vector_tip, (0, 255, 0), 3)

    def update_graph_data(self, metric_graph, plot_id):
        x_values = len(metric_graph)
        x_time = np.linspace(0, x_values, x_values)
        self.figures[1][plot_id][0].set_xdata(x_time)
        self.figures[1][plot_id][0].set_ydata([x_location[0] for x_location in metric_graph])
        self.figures[1][plot_id][1].set_xdata(x_time)
        self.figures[1][plot_id][1].set_ydata([y_location[1] for y_location in metric_graph])

    def update_graphs(self, timer, frame_id, objects_count):
        self.frame_count += 1
        if self.frame_count >= 1:
            for i in range(self.number_of_graph_queues):
                self.update_graph_data(self.metric_graph_stack[i], i)
            self.figures[0][0].canvas.flush_events()
            self.add_hud_infos(timer, frame_id, objects_count)

            # https://stackoverflow.com/questions/53324068/a-faster-refresh-rate-with-plt-imshow
            cv2.namedWindow('img', cv2.WINDOW_AUTOSIZE)
            cv2.imshow("img", self.tracking_datas[-1][0])
            cv2.waitKey(1)

    def create_plot(self, gridx, gridy, title, xlabel, ylabel, curve_1_color, curve_2_color, axs, y_range):
        axs[gridx, gridy].set_title(title)
        axs[gridx, gridy].set(xlabel=xlabel, ylabel=ylabel)
        x = np.linspace(0, self.max_graph_size, self.max_graph_size)
        y = np.linspace(y_range[0], y_range[1], self.max_graph_size)
        line1, = axs[gridx, gridy].plot(x, y, curve_1_color)
        line2, = axs[gridx, gridy].plot(x, y, curve_2_color)
        self.figures[1].append([line1, line2])

    def launch_graphs(self):
        # From https://www.delftstack.com/howto/matplotlib/how-to-plot-in-real-time-using-matplotlib/
        # From https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html
        plt.ion()

        figure1, axs = plt.subplots(2, 2, figsize=(12, 8),
                                    gridspec_kw={'width_ratios': [1, 1], 'height_ratios': [1, 1]})
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.35, hspace=0.40)

        figure1.suptitle('Online metrics', size=22)
        figure1.canvas.set_window_title('Online swarm metrics')
        self.create_plot(0, 0, 'Centers of mass (x,y)', 'Frames', 'x(Orange),y(green)', 'tab:orange', 'tab:green', axs,
                         [0, 1280])
        self.create_plot(0, 1, 'Moving Average over 7 frames', 'Frames', 'x(Orange),y(green)', 'tab:orange',
                         'tab:green', axs, [0, 1280])
        self.create_plot(1, 0, 'Global velocity vector (Dx,Dy)', 'Frames', 'Dx(Blue),Dy(Red)', 'tab:blue', 'tab:red',
                         axs, [-20, 20])
        self.create_plot(1, 1, 'Moving Average over 7 frames', 'Frames', 'Dx(Blue),Dy(Red)', 'tab:blue', 'tab:red', axs,
                         [-20, 20])

        self.figures[0].append(figure1)
