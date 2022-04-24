import numpy as np
# Heatmap of the positions ?
# Stitching tracks with each other
# Metric database
# Mean average of the fastest entity (keep the fastest among the last 7 ones)

# Individual heatmap for each ID computed for a batch of frames, offline processing
# Watershed to see the networks of connected entities per frame

from yolox.swarm_metrics.swarm_metrics_utils import get_color, dump_swarm_metrics
from yolox.swarm_metrics.swarm_metrics_GUI import SWARMMetricsGUI
import cv2
from collections import deque
import numpy as np
import matplotlib.pyplot as plt


class SWARMMetrics(object):
    def __init__(self, args, vis_folder, width, height):
        self.args = args
        self.number_of_metric_queues = 9
        self.number_of_persistent_queues = 1
        self.number_of_graph_queues = 4
        self.max_trail_size = 8000
        self.current_entity_number = 0
        self.moving_average_global_center = 5  # Should be below or equal to "max_queue_size"
        self.moving_average_fastest_entity = 5  # Should be below or equal to "max_queue_size"
        self.object_disk_size_in_heatmap = 90
        self.individual_object_disk_size_in_heatmap = 30
        self.max_queue_size = 15
        self.persistent_queue_size = 99999
        self.paused = True
        self.max_graph_size = 40
        self.frame_count = 0
        self.frame_id = 0
        self.velocity_vector_scale = 2
        self.visualization_folder = vis_folder
        self.tracking_datas = deque(maxlen=self.max_queue_size)
        self.objects_trails = deque(maxlen=self.max_trail_size)
        self.pySimpleGui = SWARMMetricsGUI(width, height)
        """
            metric_main_stack[0] = metric_1 : float_global_centers
            metric_main_stack[1] = metric_2 : float_mean_global_centers
            metric_main_stack[2] = metric_3 : individual centers, velocity vectors and norms ?
            metric_main_stack[3] = metric_4 : fastest_entity
            metric_main_stack[4] = metric_5 : float_global_velocity_deltas
            metric_main_stack[5] = metric_6 : float_mean_global_velocity_deltas
            metric_main_stack[6] = metric_7 : mean_fastest_entity
            metric_main_stack[7] = metric_8 : heat map of positions
        """
        self.metric_main_stack = []
        for i in range(self.number_of_metric_queues):
            self.metric_main_stack.append(deque(maxlen=self.max_queue_size))
        self.metric_persistent_stack = []
        for i in range(self.number_of_persistent_queues):
            self.metric_persistent_stack.append(deque(maxlen=self.persistent_queue_size))
        self.metric_graph_stack = []
        for i in range(self.number_of_graph_queues):
            self.metric_graph_stack.append(deque(maxlen=self.max_graph_size))

        self.figures = [[], []]
        self.launch_graphs()
        self.init_control_gui()

    def online_metrics_inputs(self, im, im_h, im_w, tlwhs, obj_ids, frame_id=0, timer=None):
        self.tracking_datas.append([im, [im_w, im_h], tlwhs, obj_ids])
        self.frame_id = frame_id
        self.select_online_metrics(timer, frame_id, len(tlwhs))
        self.dump_metrics()
        self.stack_trim()
        # Return an annotated image
        return self.tracking_datas[-1][0]


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
        for i in range(self.number_of_persistent_queues):
            if len(self.metric_persistent_stack[i]) > self.persistent_queue_size:
                self.metric_persistent_stack[i].popleft()
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
        self.verify_attributes_consistency()
        self.compute_metric_1_immediate_global_center()
        self.compute_metric_2_immediate_global_center_moving_average()
        self.compute_metric_3_and_4_and_5_velocity_vectors()
        self.compute_metric_6_global_velocity_vector_moving_average()
        self.compute_metric_7_mean_fastest_entity()
        self.compute_metric_8_networks()
        self.compute_metric_9_individual_heatmaps()
        # t1 = time.time()-t0
        # print("Metric computation elapsed time: ", t1)
        #self.draw_graphics()
        self.update_graphs()
        while 1:
            self.pySimpleGui.refresh_gui(timer, frame_id, objects_count)
            if not self.paused:
                break

    def verify_attributes_consistency(self):
        if self.moving_average_global_center > self.max_queue_size:
            self.moving_average_global_center = self.max_queue_size
        if self.moving_average_fastest_entity > self.max_queue_size:
            self.moving_average_fastest_entity = self.max_queue_size
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

    """
        Moving average of the center masses locations over the last x=max_queue_size frames
    """

    def compute_metric_2_immediate_global_center_moving_average(self, ):
        sum_x = 0
        sum_y = 0
        i = 0
        for center in reversed(self.metric_main_stack[0]):
            if i < self.moving_average_global_center:
                x, y = center
                sum_x += x
                sum_y += y
                i += 1
            else:
                break
        float_mean_global_center = (sum_x / self.moving_average_global_center, sum_y / self.moving_average_global_center)
        self.metric_main_stack[1].append(float_mean_global_center)
        self.metric_graph_stack[1].append(float_mean_global_center)

    def find_object_center_and_velocity(self, tlwh_current, tlwh_previous, obj_id_previous):
        x1, y1, w1, h1 = tlwh_current
        x2, y2, w2, h2 = tlwh_previous
        center_current_float = [x1 + w1 / 2, y1 + h1 / 2]
        center_previous_float = [x2 + w2 / 2, y2 + h2 / 2]
        center_current = tuple(map(int, center_current_float))
        x_delta_float = center_current_float[0] - center_previous_float[0]
        y_delta_float = center_current_float[1] - center_previous_float[1]
        norm = x_delta_float * x_delta_float + y_delta_float * y_delta_float
        x_delta = int(x_delta_float)
        y_delta = int(y_delta_float)

        self.objects_trails.append([center_current_float, obj_id_previous])
        self.metric_main_stack[2][-1][1].append(
            [center_current_float, [x_delta_float, y_delta_float], norm, obj_id_previous])
        return center_current, norm, x_delta, y_delta

    def find_global_velocity_vector(self, sum_velocity_vector_x, sum_velocity_vector_y, current_entity_number):
        if current_entity_number > 0:
            # Store the global velocity vector of the current frame into the corresponding graph stack & main stack.
            float_global_velocity_deltas = [sum_velocity_vector_x / current_entity_number,
                                            sum_velocity_vector_y / current_entity_number]
            self.metric_main_stack[4].append(float_global_velocity_deltas)
            self.metric_graph_stack[2].append(float_global_velocity_deltas)
        else:
            self.metric_main_stack[4].append(None)
            self.metric_graph_stack[2].append(
                (0, 0))  # TODO null velocities should be distinct from non available datas

    """
        Raw velocity vectors 
        metric_3 display the velocity vector of each moving entity
        metric_4 highlight the fastest moving entity on a given frame (require metric_3)
        metric_5 compute a global velocity vector; the tail of that vector is given by the inst. center of masse.
    """

    def compute_metric_3_and_4_and_5_velocity_vectors(self, ):
        sum_velocity_vector_x = 0
        sum_velocity_vector_y = 0
        fastest_entity = [None, 0, [0, 0]]
        tracks_records_number = len(self.tracking_datas)
        self.current_entity_number = len(self.tracking_datas[-1][2])
        self.metric_main_stack[2].append([self.frame_id, []])

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
                        if norm >= fastest_entity[1]:
                            fastest_entity = obj_id_current, norm, center_current

        self.find_global_velocity_vector(sum_velocity_vector_x, sum_velocity_vector_y, self.current_entity_number)
        self.metric_main_stack[3].append(fastest_entity)

    """
    Moving average of the global velocity vector ; displayed with "moving average of center of masses" as the vector's tail
    """

    def compute_metric_6_global_velocity_vector_moving_average(self, ):
        if len(self.metric_main_stack[4]) > 1:
            sum_dx = 0
            sum_dy = 0
            metric_5_stack_size = len(self.metric_main_stack[4])
            for _, deltas in enumerate(self.metric_main_stack[4]):
                if deltas is not None:
                    dx, dy = deltas
                    sum_dx += dx
                    sum_dy += dy
            float_mean_global_velocity_deltas = [sum_dx / metric_5_stack_size, sum_dy / metric_5_stack_size]
            self.metric_main_stack[5].append(float_mean_global_velocity_deltas)
            self.metric_graph_stack[3].append(float_mean_global_velocity_deltas)
        else:
            self.metric_main_stack[5].append(None)
            self.metric_graph_stack[3].append(
                [0, 0])  # TODO null velocities should be distinct from non available datas

    def average_last_known_norms_for_entity(self,
                                            entity_data):  # [center_current_float, [x_delta_float, y_delta_float], norm, obj_id_previous]
        known_norms = [None for i in range(self.moving_average_fastest_entity)]
        final_average = None
        i = 0
        for entities_description_per_frame in reversed(self.metric_main_stack[2]):
            if i >= self.moving_average_fastest_entity:
                break
            for entity_center_and_velocity in entities_description_per_frame[1]:
                if entity_center_and_velocity[3] == entity_data[3]:
                    known_norms[i] = entity_center_and_velocity[2]
            i += 1
        final_average = self.compute_average_on_sparse_list(known_norms)
        return final_average

    def compute_average_on_sparse_list(self, input_list):
        sum = 0
        none_count = 0
        list_size = len(input_list)
        for item in input_list:
            if item is not None:
                sum += item
            else:
                none_count += 1
        return sum / (list_size - none_count)

    def find_fastest_entity_and_his_position(self, average_list):
        actual_best = 0
        candidate_id = None
        winner = [None, 0, [0, 0]]
        if len(average_list) > 0:
            for candidate in average_list:
                if candidate[0] > actual_best:
                    actual_best = candidate[0]
                    candidate_id = candidate[1]

        if len(self.metric_main_stack[2]) > 0:
            for entity_center_and_velocity in self.metric_main_stack[2][-1][1]:
                if entity_center_and_velocity[3] == candidate_id:
                    winner = [candidate_id, actual_best, entity_center_and_velocity[0]]
        return winner

    def compute_metric_7_mean_fastest_entity(self, ):
        norms_stack = []
        if len(self.metric_main_stack[2]) > 1:
            for entity_center_and_velocity in self.metric_main_stack[2][-1][1]:
                norms_stack.append([self.average_last_known_norms_for_entity(entity_center_and_velocity),
                                    entity_center_and_velocity[3]])
        mean_fastest_entity = self.find_fastest_entity_and_his_position(norms_stack)
        self.metric_main_stack[6].append(mean_fastest_entity)

    def compute_metric_8_networks(
            self, ):  # [center_current_float, [x_delta_float, y_delta_float], norm, obj_id_previous]
        heatmap_mask = np.zeros_like(self.tracking_datas[-1][0][:, :, 0]).astype("uint8")
        if len(self.metric_main_stack[2]) > 0:
            for entity_center_and_velocity in self.metric_main_stack[2][-1][1]:
                object_mask = np.zeros_like(self.tracking_datas[-1][0][:, :, 0]).astype("uint8")
                cv2.circle(object_mask, tuple(map(int, entity_center_and_velocity[0])),
                           self.object_disk_size_in_heatmap, 1, -1)
                heatmap_mask[object_mask == 1] = 1
        self.metric_main_stack[7].append(heatmap_mask)

    def compute_metric_9_individual_heatmaps(self):
        current_entity_masks = [self.frame_id, []]
        if len(self.metric_main_stack[2]) > 0:
            for entity_center_and_velocity in self.metric_main_stack[2][-1][1]:
                current_entity_masks[1].append([entity_center_and_velocity[3], np.zeros_like(self.tracking_datas[-1][0][:, :, 0]).astype("float")])
                object_mask = np.zeros_like(self.tracking_datas[-1][0][:, :, 0]).astype("float")
                cv2.circle(object_mask, tuple(map(int, entity_center_and_velocity[0])),
                           self.individual_object_disk_size_in_heatmap, 1, -1)
                current_entity_masks[1][-1][1][object_mask == 1] = 255
        self.accumulate_individual_heatmaps(current_entity_masks)

    def accumulate_individual_heatmaps(self, masks_list):
        for current_obj_id_and_mask in masks_list[1]:
            is_in_stack_0, index = self.is_ID_already_in_stack_0(current_obj_id_and_mask[0])
            if is_in_stack_0:
                self.metric_persistent_stack[0][index][1][current_obj_id_and_mask[1] == 255] += 0.00001
            else:
                self.metric_persistent_stack[0].append([current_obj_id_and_mask[0], np.zeros_like(self.tracking_datas[-1][0][:, :, 0]).astype("float")])


    def is_ID_already_in_stack_0(self, obj_id):
        i = 0
        if len(self.metric_persistent_stack[0])>0:
            for id_and_heatmap in self.metric_persistent_stack[0]:
                if id_and_heatmap[0] == obj_id:
                    return True, i
                i+=1
        return False, i


    def update_graph_data(self, metric_graph, plot_id):
        x_values = len(metric_graph)
        x_time = np.linspace(0, x_values, x_values)
        self.figures[1][plot_id][0].set_xdata(x_time)
        self.figures[1][plot_id][0].set_ydata([x_location[0] for x_location in metric_graph])
        self.figures[1][plot_id][1].set_xdata(x_time)
        self.figures[1][plot_id][1].set_ydata([y_location[1] for y_location in metric_graph])

    def update_graphs(self):
        self.frame_count += 1
        if self.frame_count >= 1:
            for i in range(self.number_of_graph_queues):
                self.update_graph_data(self.metric_graph_stack[i], i)
            self.figures[0][0].canvas.flush_events()

            # https://stackoverflow.com/questions/53324068/a-faster-refresh-rate-with-plt-imshow
            """
            cv2.namedWindow('img', cv2.WINDOW_AUTOSIZE)
            cv2.imshow("img", self.tracking_datas[-1][0])
            cv2.waitKey(1)
            """

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
        self.create_plot(0, 1, 'Moving Average', 'Frames', 'x(Orange),y(green)', 'tab:orange',
                         'tab:green', axs, [0, 1280])
        self.create_plot(1, 0, 'Global velocity vector (Dx,Dy)', 'Frames', 'Dx(Blue),Dy(Red)', 'tab:blue', 'tab:red',
                         axs, [-20, 20])
        self.create_plot(1, 1, 'Moving Average', 'Frames', 'Dx(Blue),Dy(Red)', 'tab:blue', 'tab:red', axs,
                         [-20, 20])

        self.figures[0].append(figure1)

    def init_control_gui(self):
        self.pySimpleGui.set_swarm_metric(self)
        self.pySimpleGui.init_gui()


