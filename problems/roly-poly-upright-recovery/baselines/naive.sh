#!/usr/bin/env bash
set -euo pipefail

mkdir -p /tmp/output

cat > /tmp/output/model.xml <<'XML'
<?xml version="1.0"?>
<mujoco model="bad_roly">
  <option timestep="0.01" integrator="Euler"/>
  <worldbody>
    <geom name="floor" type="plane" size="1 1 0.05"/>
    <body name="doll">
      <joint name="tilt" type="hinge" axis="0 1 0"/>
      <body name="weighted_base"><geom type="sphere" size="0.05" mass="0.2"/></body>
      <body name="crown"><geom type="sphere" size="0.02" mass="0.5"/></body>
      <site name="pivot" pos="0 0 0" size="0.005"/>
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
    return 0.0
PY
