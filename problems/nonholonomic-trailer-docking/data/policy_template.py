"""Policy template for nonholonomic-trailer-docking.

Copy this shape to /tmp/output/policy.py and replace the controller.
"""


def act(obs):
    # Return normalized [drive, steering] commands in [-1, 1].
    _ = obs
    return [0.0, 0.0]
