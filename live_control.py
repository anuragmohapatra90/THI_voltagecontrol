"""Prepare the reference ADG MLP on the same physical grid used by the dashboard."""
from active_grid.controller import net_bus_pq, voltage_violation


def prepare_controller(lab, samples=3000):
    if lab.voltage_controller is None:
        from active_grid.training import train_controller
        train_controller(lab, samples=samples)
    return lab.voltage_controller
