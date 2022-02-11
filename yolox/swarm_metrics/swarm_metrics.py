import numpy as np
from yolox.swarm_metrics.swarm_metrics_utils import get_color
import cv2


class SWARMMetrics(object):
    def __init__(self, args):
        self.previous_im = None
        self.previous_tlwhs = None
        self.previous_obj_ids = None
        self.im = None
        self.tlwhs = None
        self.obj_ids = None
        self.im_h = None
        self.im_w = None
        self.args = args

    def online_metrics_inputs(self, im, im_h, im_w, tlwhs, obj_ids):
        self.im = im
        self.im_h = im_h
        self.im_w = im_w
        self.tlwhs = tlwhs
        self.obj_ids = obj_ids
        self.select_online_metrics()
        self.previous_im = im
        self.previous_tlwhs = tlwhs
        self.previous_obj_ids = obj_ids
        return self.im

    def select_online_metrics(self,):
        # Add here additional metrics
        if self.args.swarm_metric_1:
            self.compute_metric_1()
        return self.im

    def compute_metric_1(self,):
        sum_mass = 0
        sum_x = 0
        sum_y = 0
        # from https://stackoverflow.com/questions/12801400/find-the-center-of-mass-of-points
        for i, tlwh in enumerate(self.tlwhs):
            x1, y1, w, h = tlwh
            #sum_mass += tlwh_mass
            sum_mass += 1
            sum_x += x1 + w/2
            sum_y += y1 + h/2
        center = tuple(map(int, (sum_x/sum_mass, sum_y/sum_mass)))
        cv2.circle(self.im, center, radius=5, color=(0, 0, 255), thickness=6)

