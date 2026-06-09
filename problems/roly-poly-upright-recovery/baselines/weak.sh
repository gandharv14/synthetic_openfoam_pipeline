#!/usr/bin/env bash
set -euo pipefail

mkdir -p /tmp/output

# Compiles but crown heavier than base (fails structure) and bang-bang torque.
cat > /tmp/output/model.xml <<'XML'
<?xml version="1.0"?>
<mujoco model="weak_roly">
  <option timestep="0.004" integrator="RK4" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.05"/>
    <body name="doll" pos="0 0 0.1">
      <site name="pivot" pos="0 0 -0.1" size="0.005"/>
      <joint name="tilt" type="hinge" axis="0 1 0" pos="0 0 -0.1" damping="0.1"/>
      <body name="weighted_base" pos="0 0 -0.03">
        <geom name="rocker" type="sphere" size="0.07" mass="0.15"/>
      </body>
      <body name="crown" pos="0 0 0.1">
        <geom type="capsule" fromto="0 0 -0.02 0 0 0.06" size="0.03" mass="0.9"/>
      </body>
    </body>
  </worldbody>
  <actuator><motor name="base_torque" joint="tilt" ctrlrange="-12 12"/></actuator>
  <sensor>
    <jointpos name="tilt_pos" joint="tilt"/>
    <jointvel name="tilt_vel" joint="tilt"/>
    <framezaxis name="upright_axis" objtype="body" objname="crown"/>
  </sensor>
</mujoco>
XML

cat > /tmp/output/policy.py <<'PY'
def act(obs):
    return 8.0 if float(obs.get("tilt_angle", 0.0)) > 0 else -8.0
PY
