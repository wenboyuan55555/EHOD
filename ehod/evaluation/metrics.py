from __future__ import annotations

import math
from typing import Dict

import numpy as np
import torch


class StreamingMetrics(object):
    def __init__(self) -> None:
        self.n = 0
        self.absolute = 0.0
        self.square = 0.0
        self.sum_y = 0.0
        self.sum_p = 0.0
        self.sum_y2 = 0.0
        self.sum_p2 = 0.0
        self.sum_yp = 0.0
        self.target_total = 0.0
        self.prediction_total = 0.0
        self.target_zero = 0
        self.predicted_zero = 0
        self.positive_absolute = 0.0
        self.positive_n = 0
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        p = prediction.detach().double().reshape(-1).cpu()
        y = target.detach().double().reshape(-1).cpu()
        error = p - y
        self.n += int(y.numel())
        self.absolute += float(error.abs().sum().item())
        self.square += float(error.square().sum().item())
        self.sum_y += float(y.sum().item())
        self.sum_p += float(p.sum().item())
        self.sum_y2 += float(y.square().sum().item())
        self.sum_p2 += float(p.square().sum().item())
        self.sum_yp += float((y * p).sum().item())
        self.target_total += float(y.sum().item())
        self.prediction_total += float(p.sum().item())
        target_positive = y >= 0.5
        predicted_positive = p >= 0.5
        self.target_zero += int((~target_positive).sum().item())
        self.predicted_zero += int((~predicted_positive).sum().item())
        if bool(target_positive.any()):
            self.positive_absolute += float(error[target_positive].abs().sum().item())
            self.positive_n += int(target_positive.sum().item())
        self.tp += int((target_positive & predicted_positive).sum().item())
        self.fp += int((~target_positive & predicted_positive).sum().item())
        self.fn += int((target_positive & ~predicted_positive).sum().item())

    def compute(self) -> Dict[str, float]:
        n = max(self.n, 1)
        covariance = self.sum_yp - self.sum_y * self.sum_p / n
        variance_y = self.sum_y2 - self.sum_y * self.sum_y / n
        variance_p = self.sum_p2 - self.sum_p * self.sum_p / n
        pcc = covariance / math.sqrt(max(variance_y * variance_p, 1.0e-12))
        precision = self.tp / max(self.tp + self.fp, 1)
        recall = self.tp / max(self.tp + self.fn, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1.0e-12)
        return {
            "n": self.n,
            "mae": self.absolute / n,
            "rmse": math.sqrt(self.square / n),
            "pcc": pcc,
            "activation_f1": f1,
            "false_activation_rate": self.fp / max(self.target_zero, 1),
            "miss_activation_rate": self.fn / max(self.positive_n, 1),
            "positive_mae": self.positive_absolute / max(self.positive_n, 1),
            "target_zero_rate": self.target_zero / n,
            "predicted_zero_rate": self.predicted_zero / n,
            "target_total": self.target_total,
            "prediction_total": self.prediction_total,
            "total_flow_absolute_percentage_error": abs(
                self.prediction_total - self.target_total
            )
            / max(self.target_total, 1.0e-12)
            * 100.0,
        }
