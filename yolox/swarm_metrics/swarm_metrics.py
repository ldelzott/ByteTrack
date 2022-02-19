import numpy as np
from yolox.swarm_metrics.swarm_metrics_utils import get_color
import cv2
from collections import deque


class SWARMMetrics(object):
    def __init__(self, args):
        self.args = args
        self.max_queue_size = 7
        self.tracking_datas = deque(maxlen=self.max_queue_size)

        # Contains center of masses
        self.metric_1_stack = deque(maxlen=self.max_queue_size)
        # Contains moving average of center of masses
        self.metric_2_stack = deque(maxlen=self.max_queue_size)
        # Contains delta_x and delta_y of inst. global velocity vector
        self.metric_5_stack = deque(maxlen=self.max_queue_size)
        # Contains moving average of delta_x and delta_y of inst. global velocity vector
        self.metric_6_stack = deque(maxlen=self.max_queue_size)

    def online_metrics_inputs(self, im, im_h, im_w, tlwhs, obj_ids):
        self.tracking_datas.append([im, [im_w, im_h], tlwhs, obj_ids])
        self.select_online_metrics()
        self.stack_trim()
        # Return an annotated image
        return self.tracking_datas[-1][0]

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

    def select_online_metrics(self,):
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

    """
    'Immediate' center of mass location
    """
    def compute_metric_1(self,):
        sum_mass = 0
        sum_x = 0
        sum_y = 0
        # from https://stackoverflow.com/questions/12801400/find-the-center-of-mass-of-points
        for i, tlwh in enumerate(self.tracking_datas[-1][2]):
            x1, y1, w, h = tlwh
            #sum_mass += tlwh_mass
            sum_mass += 1
            sum_x += x1 + w/2
            sum_y += y1 + h/2
        center = tuple(map(int, (sum_x/sum_mass, sum_y/sum_mass)))
        self.metric_1_stack.append([sum_x/sum_mass, sum_y/sum_mass])
        cv2.circle(self.tracking_datas[-1][0], center, radius=5, color=(0, 0, 255), thickness=7)
        if self.args.swarm_metric_2:
            self.compute_metric_2()

    '''
        Moving average of the center masses locations over the last x=max_queue_size frames
    '''
    def compute_metric_2(self, ):
        sum_x = 0
        sum_y = 0
        metric_1_stack_size = len(self.metric_1_stack)
        for _, center in enumerate(self.metric_1_stack):
            x, y = center
            sum_x += x
            sum_y += y
        mean_center = tuple(map(int, (sum_x/metric_1_stack_size, sum_y/metric_1_stack_size)))
        self.metric_2_stack.append(mean_center)
        cv2.circle(self.tracking_datas[-1][0], mean_center, radius=5, color=(0, 255, 0), thickness=7)


    '''
        Raw velocity vectors 
        metric_3 display the velocity vector of each moving entity
        metric_4 highlight the fastest moving entity on a given frame (require metric_3)
        metric_5 compute a global velocity vector; the tail of that vector is given by the inst. center of masse.
    '''
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
                        x_delta = center_current[0]-center_previous[0]
                        y_delta = center_current[1]-center_previous[1]
                        norm = x_delta*x_delta + y_delta*y_delta
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
            global_velocity_vector_tip = tuple(map(int, (self.metric_1_stack[-1][0] + vector_scale*sum_velocity_vector_x / current_entity_number,
                                self.metric_1_stack[-1][1] + vector_scale*sum_velocity_vector_y / current_entity_number)))

            if self.args.swarm_metric_5 and self.args.swarm_metric_1:
                instantaneous_center_of_masses = tuple(map(int, (self.metric_1_stack[-1][0], self.metric_1_stack[-1][1])))
                cv2.line(self.tracking_datas[-1][0], instantaneous_center_of_masses, global_velocity_vector_tip, (0, 0, 255), 3)
                self.metric_5_stack.append([sum_velocity_vector_x / current_entity_number,
                                            sum_velocity_vector_y / current_entity_number])

    '''
    Moving average of the global velocity vector ; displayed with "moving average of center of masses" as the vector's tail
    '''
    def compute_metric_6(self,):
        if len(self.metric_5_stack) > 1:
            sum_dx = 0
            sum_dy = 0
            vector_scale = 2
            metric_5_stack_size = len(self.metric_5_stack)
            for _, deltas in enumerate(self.metric_5_stack):
                dx, dy = deltas
                sum_dx += dx
                sum_dy += dy
            self.metric_6_stack.append([sum_dx/metric_5_stack_size, sum_dy/metric_5_stack_size])
            vector_tail = tuple(map(int, (self.metric_2_stack[-1][0], self.metric_2_stack[-1][1])))
            vector_tip = tuple(map(int, (vector_tail[0] + vector_scale*self.metric_6_stack[-1][0], vector_tail[1]
                                         + vector_scale*self.metric_6_stack[-1][1])))
            cv2.line(self.tracking_datas[-1][0], vector_tail, vector_tip, (0, 255, 0), 3)

