# Selective Repair of Neural Certificates under Learned Dynamics

Anonymous code release accompanying the submission. It contains the core implementation and one
trimmed evaluation archive per row of the main results table, which is enough to recompute every
number we report.

## What is in here

```
sablas/
  envs/            drone and ship simulators (true dynamics, observation model)
  modules/         networks, SABLAS trainer, dataset, and the diffusion/repair code
  scripts/         training and evaluation entry points
experiments/tools/ diagnostic and figure scripts used for the paper
experiments/results/  trimmed per-episode archives (see below)
recompute_table.py    reproduces the main table from the archives
```

The evaluation archives keep only `per_step_h`, `per_step_derivative`, `per_step_episode_id`,
`per_step_safe_mask`, `per_step_physical_safe`, plus the scalars `safety_rate`, `barrier_rate`,
`nondecreasing_rate`, `goal_rate` and `dt`.

Model weights, training logs, figures and paper sources are **not** included.

## Archive to table mapping

| archive | paper row |
|---|---|
| `experiments/results/gp30_eval/init200k.npz` | Table 1, drone, SABLAS (initialized, 200k steps) |
| `experiments/results/gp30_eval/rm4_certpm.npz` | Table 1, drone, CertPM |
| `experiments/results/gp30_eval/rm4_predpm.npz` | Table 1, drone, PredPM |
| `experiments/results/gp30_eval/ours_dt002.npz` | Table 1, drone, ours |
| `experiments/results/ship_eval/ship_init.npz` | Table 1, ship, SABLAS (initialized) |
| `experiments/results/ship_eval/ship_certpm.npz` | Table 1, ship, CertPM |
| `experiments/results/ship_eval/ship_predpm.npz` | Table 1, ship, PredPM |
| `experiments/results/ship_eval/ship_boost_08.npz` | Table 1, ship, ours |

Each archive is a full evaluation run (100 episodes for drone, 100 episodes of 1000 steps for ship),
so the per-step rows can be re-aggregated any way the reader prefers.

## Metric definitions

With `h(x) = min_k h_k(x)`:

| metric | definition |
|---|---|
| SR | fraction of steps whose pre-action state is farther than `safe_dist` from every obstacle |
| BR | fraction of steps with `h(x_t) >= 0` (label consistency) |
| **NDR** | fraction of steps with `(h(x_{t+1}) - h(x_t)) / dt + h(x_t) >= 0` (agent level) |

The evaluators also print an **all-slot** NDR (`nondecreasing_rate`), which requires all eight slots
to satisfy the condition. It is a different, stricter quantity; every number in the paper uses the
agent-level definition above.

## Reproducing the main table

```bash
pip install -r requirements.txt
python recompute_table.py
```

The script prints SR, BR, NDR, the standard deviation of `h` along each trajectory (a diagnostic
that separates a structured certificate from a near-constant one) and the violating step count, for
the four drone rows and the four ship rows. No training is required.

## Retraining (optional)

Training needs the identified nominal model and the learned diffusion model, which are not shipped.
The entry points are:

```bash
# SABLAS baseline (initialized)
python sablas/scripts/train_drone.py --param <nominal.npz> --steps 200000 --out_prefix <prefix>

# our repair loop
python sablas/scripts/diffusion_repair/train_dgsr_repair.py --env_family drone --env drone \
  --train_data <train.npz> --validation_data <validation.npz> \
  --dynamics_model <diffusion.pth> --mean_field <mean_field.pth> --credible_mode meanfield \
  --cbf <start_cbf.pth> --controller <start_controller.pth> \
  --out_prefix <prefix> --max_stage1_rounds 4

# monitor baselines (canonical runtime-monitoring implementation)
python sablas/scripts/monitoring/repair_monitor.py --monitor certpm \
  --cbf <start_cbf.pth> --controller <start_controller.pth> \
  --train_data <train.npz> --out_cbf <cbf.pth> --out_controller <ctl.pth> --max_rounds 6
```

## Hardware

All reported numbers were produced on an Intel Core i5-12450H CPU with 15.8 GB of RAM. The 4 GB
laptop GPU is too small for the diffusion sampling used in the credible-set construction, so every
training and evaluation run used the CPU.
