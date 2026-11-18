# Recompute the main results table from the shipped archives.

import os

import numpy as np

def score(path):
    z = np.load(path, allow_pickle=True)
    h = np.asarray(z["per_step_h"], dtype=float)
    ep = np.asarray(z["per_step_episode_id"])
    dt = float(np.asarray(z["dt"]).ravel()[0]) if "dt" in z.files else 0.02
    hs = h.min(axis=1)
    d = np.zeros_like(hs)
    d[:-1] = np.diff(hs)
    d[1:][ep[1:] != ep[:-1]] = 0.0
    viol = ((d / dt) + hs)[:-1] < 0
    return dict(SR=float(np.asarray(z["safety_rate"])),
                BR=float(np.mean(hs >= 0.0)),
                NDR=float(1.0 - viol.mean()),
                std_h=float(hs.std()),
                viol=int(viol.sum()))

ROWS = [
    ("drone", "initialized", "experiments/results/gp30_eval/init200k.npz"),
    ("drone", "CertPM", "experiments/results/gp30_eval/rm4_certpm.npz"),
    ("drone", "PredPM", "experiments/results/gp30_eval/rm4_predpm.npz"),
    ("drone", "ours", "experiments/results/gp30_eval/ours_dt002.npz"),
    ("ship", "initialized", "experiments/results/ship_eval/ship_init.npz"),
    ("ship", "CertPM", "experiments/results/ship_eval/ship_certpm.npz"),
    ("ship", "PredPM", "experiments/results/ship_eval/ship_predpm.npz"),
    ("ship", "ours", "experiments/results/ship_eval/ship_boost_08.npz"),
]

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    print("%-6s %-12s %8s %8s %8s %9s %8s" % ("env", "method", "SR", "BR", "NDR", "std(h)", "viol"))
    for env, name, rel in ROWS:
        p = os.path.join(here, rel)
        if not os.path.isfile(p):
            print("%-6s %-12s   (archive missing)" % (env, name))
            continue
        m = score(p)
        print("%-6s %-12s %8.4f %8.4f %8.4f %9.4f %8d"
              % (env, name, m["SR"], m["BR"], m["NDR"], m["std_h"], m["viol"]))

if __name__ == "__main__":
    main()
