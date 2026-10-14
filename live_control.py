"""Dashboard controller entry point: ADG_datadriven's net-PQ MLP and signed-PQ MPC."""
from active_grid.controller import MLP_MPC_Controller,net_bus_pq,voltage_violation
from active_grid.training import train_controller


def prepare_controller(lab,samples=3000):
    if getattr(lab,'voltage_controller',None) is None:train_controller(lab,samples=samples)
    return lab.voltage_controller
