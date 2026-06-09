# Underwater Glider Thermocline Tracking

This MuJoCo task asks for a deterministic controller for a buoyancy/pitch underwater glider. The controller must collect ordered thermocline samples with a short continuous in-window scan while dealing with hidden ocean-current perturbations and unsafe plume regions.

The scorer gates the headline score by the weakest hidden survey's sample
completion and finish quality, and per-survey credit is safety-gated by plume
and workspace clearance. A high average tracking score is not enough unless
every hidden profile is completed without clipping the hidden corridor hazards.
The headline completion gate starts at weakest ordered-sample score 0.70 and
finish score 0.20, and reaches full strength at 0.90 and 0.50 respectively.
Per-survey sample, tracking, and finish credit are also zeroed when safety is
below 0.20 and ungated once safety reaches 0.78.

The reference solution is a predictive depth controller with current cancellation and local repulsion from plume and workspace boundaries. The scorer evaluates hidden profiles and currents only.
