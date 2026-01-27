from datetime import datetime

import numpy as np

from simglucose.controller.base import Action
from simglucose.controller.event_triggered_nmpc import EventTriggeredNMPCController


class _DummyController:
    def reset(self):
        return None

    def policy(self, observation, reward, done, **info):
        # return some nonzero basal so wrapper isn't trivially 0
        return Action(basal=0.01, bolus=0.0)


class _Obs:
    def __init__(self, cgm):
        self.CGM = cgm


def test_event_triggered_controller_returns_action():
    base = _DummyController()
    ctrl = EventTriggeredNMPCController(base, verbose=False)
    obs = _Obs(cgm=150.0)
    info = {
        "sample_time": 5,
        "time": datetime(2025, 1, 1, 0, 0, 0),
        "bg": 150.0,
    }
    a = ctrl.policy(obs, 0.0, False, **info)
    assert isinstance(a, tuple)
    assert hasattr(a, "basal") and hasattr(a, "bolus")
    assert np.isfinite(a.basal)
    assert np.isfinite(a.bolus)

