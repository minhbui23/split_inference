import contextlib
from copy import deepcopy
from typing import Union, List

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from ultralytics import YOLO
from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import ops

class SplitDetectionModel(nn.Module):
    def __init__(self, cfg=YOLO('yolov8n.pt').model,split_layer=-1):
        super().__init__()
        self.model = cfg.model
        self.save = cfg.save
        self.stride = cfg.stride
        self.inplace = cfg.inplace
        self.names = cfg.names
        self.yaml = cfg.yaml
        self.nc = len(self.names)  # cfg.nc
        self.task = cfg.task
        self.pt = True

        if split_layer > 0:
            self.head = self.model[:split_layer]
            self.tail = self.model[split_layer:]

    def forward_head(self, x, output_from=()):
        y, dt = [], [] # outputs
        for i, m in enumerate(self.head):
            if m.f != -1: # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f] # from earlier layers
            x = m(x) # run
            if (m.i in self.save) or (i in output_from):
                y.append(x)
            else:
                y.append(None)

        for mi in range(len(y)):
            if mi not in output_from:
                y[mi] = None

        if y[-1] is None:
            y[-1] = x
        return {"layers_output": y, "last_layer_idx": len(y) - 1}

    def forward_tail(self, x):
        y = x["layers_output"]
        x = y[x["last_layer_idx"]]
        for m in self.tail:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x) # run
            y.append(x if m.i in self.save else None)

        y = x
        if isinstance(y, (list, tuple)):
            return self.from_numpy(y[0] if len(y) == 1 else [self.from_numpy(x) for x in y])
        else:
            return self.from_numpy(y)

    def _predict_once(self, x):
        y, dt = [], [] # outputs
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)
        return x

    def forward(self, x):
        return self._predict_once(x)

    def from_numpy(self, x):
        return torch.tensor(x).to(self.device) if isinstance(x, np.ndarray) else x

class SplitDetectionPredictor(DetectionPredictor):
    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        model.fp16 = self.args.half
        self.model = model

    def postprocess(self, preds, img, orig_imgs=None, path=None):
        """Post-processes predictions and returns a list of Results objects."""
        """Choose the best bounding boxes from the output."""
        preds = ops.non_max_suppression(preds, # output from model
                                        self.args.conf,
                                        self.args.iou,
                                        self.args.classes,
                                        self.args.agnostic_nms,
                                        max_det=self.args.max_det,
                                        nc=len(self.model.names),
                                        )
        if orig_imgs is not None and not isinstance(orig_imgs, list): # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        return self.construct_results(preds, img, orig_imgs, path)

    def construct_results(self, preds, img, orig_imgs, path):
        return [
            self.construct_result(pred, img, orig_img, img_path)
            for pred, orig_img, img_path in zip(preds, orig_imgs, path)
        ]

    def construct_result(self, pred, img, orig_img, img_path):
        pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        return Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6])



