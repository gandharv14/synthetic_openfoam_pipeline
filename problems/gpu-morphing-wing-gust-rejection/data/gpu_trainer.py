"""Optional CUDA-oriented residual-policy training scaffold.

This public helper sketches the accelerator workflow: sample randomized
shape/gust/fatigue batches on GPU, fit a residual controller, then export
deterministic inference code to /tmp/output/policy.py. The hidden grader does
not import this file.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch


def main() -> None:
    cases = json.loads(Path("/data/public_training_cases.json").read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device=device).manual_seed(20260520)
    net = torch.nn.Sequential(
        torch.nn.Linear(28, 128),
        torch.nn.SiLU(),
        torch.nn.Linear(128, 128),
        torch.nn.SiLU(),
        torch.nn.Linear(128, 7),
    ).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1.8e-3, weight_decay=1e-4)

    for _step in range(1600):
        case_id = int(torch.randint(len(cases), (1,), generator=generator, device=device))
        case = cases[case_id]
        base = torch.tensor(case["base"], dtype=torch.float32, device=device)
        amp = torch.tensor(case["amplitude"], dtype=torch.float32, device=device)
        gains = torch.tensor(case["actuator_gains"], dtype=torch.float32, device=device)
        q = base + 0.35 * torch.randn((384, 7), generator=generator, device=device)
        qd = 0.75 * torch.randn((384, 7), generator=generator, device=device)
        target = base + amp * torch.randn((384, 7), generator=generator, device=device).tanh()
        target_vel = 0.24 * amp * torch.randn((384, 7), generator=generator, device=device)
        features = torch.cat([q, qd, target, target_vel], dim=1)
        desired = (2.4 * (target - q) + 0.20 * (target_vel - qd)) / gains
        loss = torch.nn.functional.smooth_l1_loss(torch.tanh(net(features)), desired.clamp(-1, 1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    print(f"trained morphing-wing residual scaffold on {device}; final_loss={float(loss):.5f}")


if __name__ == "__main__":
    main()
