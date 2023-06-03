# -*- coding: utf-8 -*-
from dataclasses import dataclass


@dataclass
class Metrics:
    """Class for storing error metrics from training NeuralProphet."""

    mae_train: float
    mae_val: float
