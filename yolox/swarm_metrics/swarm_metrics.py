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
        self.max_queue_size = 7
        self.max_graph_size = 40
        self.frame_count = 0
        self.frame_id = 0
        self.visualization_folder = vis_folder
        self.tracking_datas = deque(maxlen=self.max_queue_size)

        # Contains center of masses
        self.metric_1_stack = deque(maxlen=self.max_queue_size)
        # Contains moving average of center of masses
        self.metric_2_stack = deque(maxlen=self.max_queue_size)
        # Contains delta_x and delta_y of inst. global velocity vector
        self.metric_5_stack = deque(maxlen=self.max_queue_size)
        # Contains moving average of delta_x and delta_y of inst. global velocity vector
        self.metric_6_stack = deque(maxlen=self.max_queue_size)

        # The 'metric_i_graph' are similar to 'metric_i_stack' but the size of the queues. TODO: remove redundancy ?
        self.metric_1_graph = deque(maxlen=self.max_graph_size)
        self.metric_2_graph = deque(maxlen=self.max_graph_size)
        self.metric_5_graph = deque(maxlen=self.max_graph_size)
        self.metric_6_graph = deque(maxlen=self.max_graph_size)

        self.figures = []
        self.launch_graphs()

    def online_metrics_inputs(self, im, im_h, im_w, tlwhs, obj_ids, frame_id=0):
        self.tracking_datas.append([im, [im_w, im_h], tlwhs, obj_ids])
        self.frame_id = frame_id

        self.select_online_metrics()
        self.dump_metrics()
        self.stack_trim()

        # Return an annotated image
        return self.tracking_datas[-1][0]

    def dump_metrics(self):
        if not self.args.dump_metrics:
            return
        if len(self.metric_1_graph) == 0:
            metric_1 = None
        else:
            metric_1 = self.metric_1_graph[-1]

        if len(self.metric_2_graph) == 0:
            metric_2 = None
        else:
            metric_2 = self.metric_2_graph[-1]

        if len(self.metric_5_graph) == 0:
            metric_5 = None
        else:
            metric_5 = self.metric_5_graph[-1]

        if len(self.metric_6_graph) == 0:
            metric_6 = None
        else:
            metric_6 = self.metric_6_graph[-1]

        dump_swarm_metrics(self.visualization_folder, self.frame_id,  metric_1, metric_2, metric_5, metric_6)





    """
        The various queues used in SWARMMetrics are limited in size: 'stack_trim' is used to remove the oldest item 
        of a queues when his maximum size is reached.
    """
    def stack_trim(self):
        if len(self.tracking_datas) > self.max_queue_size:
            self.tracking_datas.popleft()
        if len(self.metric_1_stack) > self.max_queue_size:
            self.metric_1_stack.popleft()
        if len(self.metric_2_stack) > self.max_queue_size:
            self.metric_2_stack.popleft()
        if len(self.metric_5_stack) > self.max_queue_size:
            self.metric_5_stack.popleft()
        if len(self.metric_6_stack) > self.max_queue_size:
            self.metric_6_stack.popleft()
        if len(self.metric_1_graph) > self.max_graph_size:
            self.metric_1_graph.popleft()
        if len(self.metric_2_graph) > self.max_graph_size:
            self.metric_2_graph.popleft()
        if len(self.metric_5_graph) > self.max_graph_size:
            self.metric_5_graph.popleft()
        if len(self.metric_6_graph) > self.max_graph_size:
            self.metric_6_graph.popleft()

    def select_online_metrics(self, ):
        # Add here additional metrics
        # Note that metric_2 require metric_1: see compute_metric_1() function
        # Same comment for metric_4 : see compute_metric_3() function
        # Same comment for metric_5 : see compute_metric_3() function
        if self.args.swarm_metric_1:
            self.compute_metric_1()
        if self.args.swarm_metric_3:
            self.compute_metric_3()
        if self.args.swarm_metric_6:
            self.compute_metric_6()
        self.update_graphs()

    """
        'Immediate' center of mass location
    """
    def compute_metric_1(self, ):
        sum_mass = 0
        sum_x = 0
        sum_y = 0
        # from https://stackoverflow.com/questions/12801400/find-the-center-of-mass-of-points
        for i, tlwh in enumerate(self.tracking_datas[-1][2]):
            x1, y1, w, h = tlwh
            # sum_mass += tlwh_mass
            sum_mass += 1
            sum_x += x1 + w / 2
            sum_y += y1 + h / 2
        float_center = [sum_x / sum_mass, sum_y / sum_mass]
        self.metric_1_stack.append(float_center)
        self.metric_1_graph.append(float_center)
        center = tuple(map(int, float_center))
        cv2.circle(self.tracking_datas[-1][0], center, radius=5, color=(0, 0, 255), thickness=7)
        if self.args.swarm_metric_2:
            self.compute_metric_2()

    """
        Moving average of the center masses locations over the last x=max_queue_size frames
    """
    def compute_metric_2(self, ):
        sum_x = 0
        sum_y = 0
        metric_1_stack_size = len(self.metric_1_stack)
        for _, center in enumerate(self.metric_1_stack):
            x, y = center
            sum_x += x
            sum_y += y
        mean_center_float = (sum_x / metric_1_stack_size, sum_y / metric_1_stack_size)
        mean_center = tuple(map(int, mean_center_float))
        self.metric_2_stack.append(mean_center_float)
        self.metric_2_graph.append(mean_center_float)
        cv2.circle(self.tracking_datas[-1][0], mean_center, radius=5, color=(0, 255, 0), thickness=7)

    """
        Raw velocity vectors 
        metric_3 display the velocity vector of each moving entity
        metric_4 highlight the fastest moving entity on a given frame (require metric_3)
        metric_5 compute a global velocity vector; the tail of that vector is given by the inst. center of masse.
    """
    def compute_metric_3(self, ):
        vector_scale = 2
        sum_velocity_vector_x = 0
        sum_velocity_vector_y = 0
        fastest_entity = [None, 0, [0, 0]]
        current_entity_number = len(self.tracking_datas)
        if current_entity_number >= 2:
            for i, tlwh_current in enumerate(self.tracking_datas[-1][2]):
                obj_id_current = int(self.tracking_datas[-1][3][i])
                for j, tlwh_previous in enumerate(self.tracking_datas[-2][2]):
                    obj_id_previous = int(self.tracking_datas[-2][3][j])
                    if obj_id_previous == obj_id_current:
                        x1, y1, w1, h1 = tlwh_current
                        x2, y2, w2, h2 = tlwh_previous
                        center_current = tuple(map(int, (x1 + w1 / 2, y1 + h1 / 2)))
                        center_previous = tuple(map(int, (x2 + w2 / 2, y2 + h2 / 2)))
                        x_delta = center_current[0] - center_previous[0]
                        y_delta = center_current[1] - center_previous[1]
                        norm = x_delta * x_delta + y_delta * y_delta
                        vector_tip = tuple(map(int, (center_current[0] + x_delta * vector_scale,
                                                     center_current[1] + y_delta * vector_scale)))
                        if self.args.swarm_metric_4 and norm >= fastest_entity[1]:
                            fastest_entity = obj_id_current, norm, center_current

                        sum_velocity_vector_x += x_delta
                        sum_velocity_vector_y += y_delta

                        color = get_color(abs(obj_id_previous))
                        cv2.line(self.tracking_datas[-1][0], center_current, vector_tip, color, 3)
                        cv2.circle(self.tracking_datas[-1][0], center_current, radius=3, color=(0, 255, 0), thickness=3)
            if fastest_entity[0] is not None:
                cv2.circle(self.tracking_datas[-1][0], fastest_entity[2], radius=40, color=(0, 0, 255), thickness=2)
            global_velocity_vector_tip = tuple(
                map(int, (self.metric_1_stack[-1][0] + vector_scale * sum_velocity_vector_x / current_entity_number,
                          self.metric_1_stack[-1][1] + vector_scale * sum_velocity_vector_y / current_entity_number)))

            if self.args.swarm_metric_5 and self.args.swarm_metric_1:
                instantaneous_center_of_masses = tuple(
                    map(int, (self.metric_1_stack[-1][0], self.metric_1_stack[-1][1])))
                cv2.line(self.tracking_datas[-1][0], instantaneous_center_of_masses, global_velocity_vector_tip,
                         (0, 0, 255), 3)
                global_velocity_deltas_float = [sum_velocity_vector_x / current_entity_number,
                                                sum_velocity_vector_y / current_entity_number]
                self.metric_5_stack.append(global_velocity_deltas_float)
                self.metric_5_graph.append(global_velocity_deltas_float)

    """
    Moving average of the global velocity vector ; displayed with "moving average of center of masses" as the vector's tail
    """
    def compute_metric_6(self, ):
        if len(self.metric_5_stack) > 1:
            sum_dx = 0
            sum_dy = 0
            vector_scale = 2
            metric_5_stack_size = len(self.metric_5_stack)
            for _, deltas in enumerate(self.metric_5_stack):
                dx, dy = deltas
                sum_dx += dx
                sum_dy += dy
            mean_deltas_float = [sum_dx / metric_5_stack_size, sum_dy / metric_5_stack_size]
            self.metric_6_stack.append(mean_deltas_float)
            self.metric_6_graph.append(mean_deltas_float)
            vector_tail = tuple(map(int, (self.metric_2_stack[-1][0], self.metric_2_stack[-1][1])))
            vector_tip = tuple(map(int, (vector_tail[0] + vector_scale * self.metric_6_stack[-1][0], vector_tail[1]
                                         + vector_scale * self.metric_6_stack[-1][1])))
            cv2.line(self.tracking_datas[-1][0], vector_tail, vector_tip, (0, 255, 0), 3)

    def update_graphs(self):
        self.frame_count += 1
        if self.frame_count >= 1:
            x_values = len(self.metric_1_graph)
            x_time = np.linspace(0, x_values, x_values)
            self.figures[0][1][0].set_xdata(x_time)
            self.figures[0][1][0].set_ydata([x_location[0] for x_location in self.metric_1_graph])
            self.figures[0][1][1].set_xdata(x_time)
            self.figures[0][1][1].set_ydata([y_location[1] for y_location in self.metric_1_graph])

            x_values = len(self.metric_2_graph)
            x_time = np.linspace(0, x_values, x_values)
            self.figures[1][1][0].set_xdata(x_time)
            self.figures[1][1][0].set_ydata([x_location[0] for x_location in self.metric_2_graph])
            self.figures[1][1][1].set_xdata(x_time)
            self.figures[1][1][1].set_ydata([y_location[1] for y_location in self.metric_2_graph])

            x_values = len(self.metric_5_graph)
            x_time = np.linspace(0, x_values, x_values)
            self.figures[2][1][0].set_xdata(x_time)
            self.figures[2][1][0].set_ydata([x_location[0] for x_location in self.metric_5_graph])
            self.figures[2][1][1].set_xdata(x_time)
            self.figures[2][1][1].set_ydata([y_location[1] for y_location in self.metric_5_graph])

            x_values = len(self.metric_6_graph)
            x_time = np.linspace(0, x_values, x_values)
            self.figures[3][1][0].set_xdata(x_time)
            self.figures[3][1][0].set_ydata([x_location[0] for x_location in self.metric_6_graph])
            self.figures[3][1][1].set_xdata(x_time)
            self.figures[3][1][1].set_ydata([y_location[1] for y_location in self.metric_6_graph])

            self.figures[0][0].canvas.draw()
            self.figures[0][0].canvas.flush_events()

            # The color profile need to be converted from BGR to RGB
            self.figures[4][0].imshow(cv2.cvtColor(self.tracking_datas[-1][0], cv2.COLOR_BGR2RGB))
            time.sleep(0.05)

    def launch_graphs(self):
        # From https://www.delftstack.com/howto/matplotlib/how-to-plot-in-real-time-using-matplotlib/
        # From https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html
        plt.ion()

        figure1, axs = plt.subplots(2, 2, figsize=(12, 8), gridspec_kw={'width_ratios': [1, 1], 'height_ratios': [1, 1]})
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.35, hspace=0.40)
        figure1.suptitle('Online metrics', size=22)
        figure1.canvas.set_window_title('Online swarm metrics')

        axs[0, 0].set_title('Centers of mass (x,y)')
        axs[0, 0].set(xlabel='Frames', ylabel='x(Orange),y(green)')
        x0_0 = np.linspace(0, self.max_graph_size, self.max_graph_size)
        y0_0 = np.linspace(0, 1280, self.max_graph_size)
        line0_0_1, = axs[0, 0].plot(x0_0, y0_0, 'tab:orange')
        line0_0_2, = axs[0, 0].plot(x0_0, y0_0, 'tab:green')
        self.figures.append([figure1, [line0_0_1, line0_0_2]])

        # 'self.max_queue_size' define the size of the window for MA
        axs[0, 1].set_title('Moving Average over 7 frames')
        axs[0, 1].set(xlabel='Frames', ylabel='x(Orange),y(green)')
        x0_1 = np.linspace(0, self.max_graph_size, self.max_graph_size)
        y0_1 = np.linspace(0, 1280, self.max_graph_size)
        line0_1_1, = axs[0, 1].plot(x0_1, y0_1, 'tab:orange')
        line0_1_2, = axs[0, 1].plot(x0_1, y0_1, 'tab:green')
        self.figures.append([figure1, [line0_1_1, line0_1_2]])

        axs[1, 0].set_title('Global velocity vector (Dx,Dy)')
        axs[1, 0].set(xlabel='Frames', ylabel='Dx(Blue),Dy(Red)')
        x1_0 = np.linspace(0, self.max_graph_size, self.max_graph_size)
        y1_0 = np.linspace(-20, 20, self.max_graph_size)
        line1_0_1, = axs[1, 0].plot(x1_0, y1_0, 'tab:blue')
        line1_0_2, = axs[1, 0].plot(x1_0, y1_0, 'tab:red')
        self.figures.append([figure1, [line1_0_1, line1_0_2]])

        axs[1, 1].set_title('Moving Average over 7 frames')
        axs[1, 1].set(xlabel='Frames', ylabel='Dx(Blue),Dy(Red)')
        x1_1 = np.linspace(0, self.max_graph_size, self.max_graph_size)
        y1_1 = np.linspace(-20, 20, self.max_graph_size)
        line1_1_1, = axs[1, 1].plot(x1_1, y1_1, 'tab:blue')
        line1_1_2, = axs[1, 1].plot(x1_1, y1_1, 'tab:red')
        self.figures.append([figure1, [line1_1_1, line1_1_2]])

        figure2, ax = plt.subplots(1, 1, figsize=(16, 12), gridspec_kw={'width_ratios': [1], 'height_ratios': [1]})
        plt.subplots_adjust(left=0.040, bottom=0.0125, right=0.975, top=0.9875, wspace=0.025, hspace=0.0125)
        figure2.suptitle('Annotated output image')
        ax.set(xlabel='x', ylabel='y')
        self.figures.append([plt, [None, None]])

        # Image plot

        #self.figures.append([figure1, [line0_2_1, line0_2_2]])

        # axs[0, 1].label_outer() # Hide x labels and tick labels for top plots and y ticks for right plots.
