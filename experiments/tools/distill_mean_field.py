# Distil the diffusion mean field into a small MLP that can be used as the training dynamics.

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "sablas"))
from modules.diffusion_repair.diffusion import load_uncertainty_model

class MeanFieldMLP(nn.Module):

    def __init__(self, n_state=8, m_control=3, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_state + m_control, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_state),
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=1))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--diffusion", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--train_steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--n_state", type=int, default=8)
    ap.add_argument("--m_control", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    z = np.load(args.data, allow_pickle=True)
    st = np.asarray(z["state"], dtype=np.float32)
    ac = np.asarray(z["action"], dtype=np.float32)

    model = load_uncertainty_model(args.diffusion, map_location="cpu")
    model.eval()
    labels = []
    with torch.no_grad():
        for i in range(0, len(st), 512):
            v = model.sample_vectors(torch.from_numpy(st[i:i + 512]),
                                     torch.from_numpy(ac[i:i + 512]),
                                     num_samples=args.samples)
            v = np.asarray(v.detach().cpu())
            labels.append(v.mean(axis=1) if v.ndim == 3 else v)
    y = np.concatenate(labels, axis=0)
    print("mean-field labels: %s  median norm %.3f" % (y.shape, float(np.median(np.linalg.norm(y, axis=1)))))

    mlp = MeanFieldMLP(n_state=args.n_state, m_control=args.m_control, hidden=args.hidden)
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    st_t, ac_t, y_t = torch.from_numpy(st), torch.from_numpy(ac), torch.from_numpy(y)
    for step in range(args.train_steps):
        idx = torch.randint(0, len(st_t), (args.batch,))
        pred = mlp(st_t[idx], ac_t[idx])
        loss = torch.mean((pred - y_t[idx]) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % 2000 == 0:
            print("step %d/%d loss %.6f" % (step + 1, args.train_steps, float(loss)))

    torch.save({"state_dict": mlp.state_dict(), "hidden": args.hidden,
                "n_state": args.n_state, "m_control": args.m_control}, args.out)
    resid = float(np.median(np.linalg.norm((mlp(st_t[:1000], ac_t[:1000]) - y_t[:1000]).detach().numpy(), axis=1)))
    print("distilled residual vs diffusion mean: %.4f" % resid)
    print("saved %s" % args.out)
    with open(os.path.join(ROOT, "experiments/results/mean_field_distill.json"), "w", encoding="utf-8") as fh:
        json.dump({"data": args.data, "diffusion": args.diffusion, "samples": args.samples,
                   "train_steps": args.train_steps, "residual_vs_mean": resid}, fh, indent=2)

if __name__ == "__main__":
    main()
