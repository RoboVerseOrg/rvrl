from __future__ import annotations

import math
from typing import Any

import torch
from loguru import logger as log
from torchmetrics import MeanMetric, Metric


class MetricAggregator:
    """A metric aggregator class to aggregate metrics to be tracked.
    Args:
        metrics (Optional[Dict[str, Metric]]): Dict of metrics to aggregate.

    Copied from https://github.com/Eclectic-Sheep/sheeprl/blob/1952c8e4ee1411580de18ac5387e5cf2b301723c/sheeprl/utils/metric.py#L17
    """

    disabled: bool = False

    def __init__(self, metrics: dict[str, Metric] | None = None):
        self.metrics: dict[str, Metric] = {}
        if metrics is not None:
            self.metrics = metrics

    def __iter__(self):
        return iter(self.metrics.keys())

    @torch.no_grad()
    def update(self, name: str, value: Any) -> None:
        """Update the metric with the value

        Args:
            name (str): Name of the metric
            value (Any): Value to update the metric with.

        Raises:
            ValueError: If the metric does not exist.
        """
        if not self.disabled:
            if name not in self.metrics:
                self.metrics[name] = MeanMetric()
            self.metrics[name].update(value)

    def reset(self):
        """Reset all metrics to their initial state"""
        if not self.disabled:
            for metric in self.metrics.values():
                metric.reset()

    def to(self, device: str | torch.device = "cpu") -> MetricAggregator:
        """Move all metrics to the given device
        Args:
            device (str |torch.device, optional): Device to move the metrics to. Defaults to "cpu".
        """
        if not self.disabled:
            if self.metrics:
                for k, v in self.metrics.items():
                    self.metrics[k] = v.to(device)
        return self

    @torch.no_grad()
    def compute(self) -> dict[str, Any]:
        """Reduce the metrics to a single value
        Returns:
            Reduced metrics
        """
        reduced_metrics = {}
        if not self.disabled:
            if self.metrics:
                for k, v in self.metrics.items():
                    reduced = v.compute()
                    is_tensor = torch.is_tensor(reduced)
                    if is_tensor and reduced.numel() == 1:
                        reduced_metrics[k] = reduced.item()
                    else:
                        if not is_tensor:
                            log.warning(
                                f"The reduced metric {k} is not a scalar tensor: type={type(reduced)}. "
                                "This may create problems during the logging phase.",
                                category=RuntimeWarning,
                            )
                        else:
                            log.warning(
                                f"The reduced metric {k} is not a scalar: size={v.size()}. "
                                "This may create problems during the logging phase.",
                                category=RuntimeWarning,
                            )
                        reduced_metrics[k] = reduced

                    is_tensor = torch.is_tensor(reduced_metrics[k])
                    if (is_tensor and torch.isnan(reduced_metrics[k]).any()) or (
                        not is_tensor and math.isnan(reduced_metrics[k])
                    ):
                        reduced_metrics.pop(k, None)
        return reduced_metrics
