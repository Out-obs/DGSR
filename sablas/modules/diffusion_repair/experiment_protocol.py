# Shared, testable definitions for DGSR experiment protocol and reporting.

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

SELECTION_METRICS = ("safety_rate", "barrier_rate", "active_nondecreasing_rate", "active_robust_rate")
LEGACY_METRICS = ("safety_rate", "barrier_rate", "nondecreasing_rate", "robust_rate")

DEFAULT_C1C2_RATE_LIMIT = 0.3

def conformal_order_statistic(scores, coverage=0.95):
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("at least one calibration score is required")
    if not 0.0 < float(coverage) < 1.0:
        raise ValueError("coverage must be in (0, 1)")
    if not np.all(np.isfinite(values)):
        raise ValueError("calibration scores must be finite")
    rank = int(math.ceil((values.size + 1) * float(coverage)))
    if rank > values.size:
        return float("inf")
    return float(np.partition(values, rank - 1)[rank - 1])

def split_indices_by_group(group_ids, calibration_fraction=0.2, seed=0):
    group_ids = np.asarray(group_ids).reshape(-1)
    if not 0.0 < float(calibration_fraction) < 1.0:
        raise ValueError("calibration_fraction must be in (0, 1)")
    unique = np.unique(group_ids)
    if unique.size < 2 or np.any(group_ids < 0):
        raise ValueError("at least two valid episode IDs are required for a grouped split")
    rng = np.random.RandomState(int(seed))
    rng.shuffle(unique)
    n_cal_groups = int(round(unique.size * float(calibration_fraction)))
    n_cal_groups = min(max(n_cal_groups, 1), unique.size - 1)
    cal_mask = np.isin(group_ids, unique[:n_cal_groups])
    return np.flatnonzero(~cal_mask), np.flatnonzero(cal_mask)

def _rate(numerator, denominator):
    return float(numerator / denominator) if denominator else float("nan")

def episode_safety_metrics(physical_safe, episode_ids=None, completed_episode_ids=None):
    safe = np.asarray(physical_safe, dtype=bool).reshape(-1)
    if episode_ids is None:
        ids = np.zeros(safe.size, dtype=np.int64)
    else:
        ids = np.asarray(episode_ids, dtype=np.int64).reshape(-1)
        if ids.shape != safe.shape:
            raise ValueError("episode_ids must have the same length as physical_safe")
    unique = [int(x) for x in np.unique(ids)]
    by_episode = {episode: safe[ids == episode] for episode in unique}
    safe_flags = [bool(np.all(by_episode[episode])) for episode in unique]
    if completed_episode_ids is None:
        completed = unique
    else:
        completed = [int(x) for x in completed_episode_ids]
        missing = set(completed) - set(unique)
        if missing:
            raise ValueError("completed_episode_ids not present in episode_ids")
    completed_flags = [bool(np.all(by_episode[episode])) for episode in completed]
    unsafe = ~safe
    if safe.size:
        episode_boundary = np.concatenate(([True], ids[1:] != ids[:-1]))
        previous_safe = np.concatenate(([True], safe[:-1]))
        starts = unsafe & (episode_boundary | previous_safe)
    else:
        starts = unsafe
    return {
        "evaluated_episode_count": int(len(unique)),
        "collision_free_episode_count": int(sum(safe_flags)),
        "episode_collision_free_rate": float(np.mean(safe_flags)) if safe_flags else float("nan"),
        "unsafe_episode_count": int(len(unique) - sum(safe_flags)),
        "completed_episode_count_for_safety": int(len(completed)),
        "completed_collision_free_episode_count": int(sum(completed_flags)),
        "completed_episode_collision_free_rate": (float(np.mean(completed_flags))
                                                   if completed_flags else float("nan")),
        "unsafe_step_count": int(unsafe.sum()),
        "unsafe_event_count": int(starts.sum()),
    }

def certificate_metrics(h, derivative, rho, safe_mask, danger_mask):
    h = np.asarray(h, dtype=np.float64)
    derivative = np.asarray(derivative, dtype=np.float64)
    rho = np.asarray(rho, dtype=np.float64)
    safe = np.asarray(safe_mask, dtype=bool)
    danger = np.asarray(danger_mask, dtype=bool)
    if h.ndim != 2 or derivative.shape != h.shape or rho.shape != h.shape:
        raise ValueError("h, derivative and rho must have the same (N,K) shape")
    if safe.shape != h.shape or danger.shape != h.shape:
        raise ValueError("safe_mask and danger_mask must match h shape")
    if np.any(safe & danger):
        raise ValueError("safe and danger masks must be disjoint")

    active = h >= 0.0
    c1_fail = safe & ~active
    c2_fail = danger & active
    c3_fail = active & (rho < 0.0)
    state_c1 = np.any(c1_fail, axis=1)
    state_c2 = np.any(c2_fail, axis=1)
    state_union = state_c1 | state_c2
    active_n = int(active.sum())
    safe_n = int(safe.sum())
    danger_n = int(danger.sum())
    metrics = {
        "barrier_rate": float(np.mean(np.all(active, axis=1))) if h.shape[0] else float("nan"),
        "nondecreasing_rate": float(np.mean(np.all(derivative >= 0.0, axis=1))) if h.shape[0] else float("nan"),
        "robust_rate": float(np.mean(np.all(rho >= 0.0, axis=1))) if h.shape[0] else float("nan"),
        "active_nondecreasing_rate": _rate(np.sum(active & (derivative >= 0.0)), active_n),
        "active_robust_rate": _rate(np.sum(active & (rho >= 0.0)), active_n),
        "active_pair_count": active_n,
        "c1_pair_violation_rate": _rate(c1_fail.sum(), safe_n),
        "c1_safe_barrier_coverage": _rate(np.sum(safe & active), safe_n),
        "c1_pair_count": safe_n,
        "c2_pair_violation_rate": _rate(c2_fail.sum(), danger_n),
        "c2_false_safe_rate": _rate(c2_fail.sum(), danger_n),
        "c2_danger_negative_rate": _rate(np.sum(danger & ~active), danger_n),
        "c2_pair_count": danger_n,
        "c3_active_pair_violation_rate": _rate(c3_fail.sum(), active_n),
        "condition1_state_violation_rate": float(np.mean(state_c1)) if h.shape[0] else float("nan"),
        "condition2_state_violation_rate": float(np.mean(state_c2)) if h.shape[0] else float("nan"),
        "condition12_state_violation_rate": float(np.mean(state_union)) if h.shape[0] else float("nan"),
        "condition1_state_violations": int(state_c1.sum()),
        "condition2_state_violations": int(state_c2.sum()),
        "condition12_state_violations": int(state_union.sum()),
        "condition3_active_pair_violations": int(c3_fail.sum()),
        "evaluated_steps": int(h.shape[0]),
    }
    if not np.all(np.isfinite(rho)):
        for name in ("robust_rate", "active_robust_rate", "c3_active_pair_violation_rate"):
            metrics[name] = float("nan")
        metrics["condition3_active_pair_violations"] = -1
    return metrics

def wilson_interval(successes, trials, confidence_z=1.959963984540054):
    n = int(trials)
    if n <= 0:
        return float("nan"), float("nan")
    p = float(successes) / n
    z2 = float(confidence_z) ** 2
    denominator = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denominator
    radius = (float(confidence_z) * math.sqrt((p * (1.0 - p) / n) + z2 / (4.0 * n * n))) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)

def validation_score(metrics: Mapping[str, float]) -> float:
    values = np.asarray([metrics.get(k, np.nan) for k in SELECTION_METRICS], dtype=float)
    if not np.all(np.isfinite(values)):
        return float("nan")
    return float(values.mean())

def pooled_pair_rate(rows, count_key, rate_keys):
    total_count = 0
    total_failures = 0
    complete = True
    for row in rows:
        try:
            count_value = float(row.get(count_key, np.nan))
        except (TypeError, ValueError):
            complete = False
            continue
        if not np.isfinite(count_value) or count_value < 0 or not count_value.is_integer():
            complete = False
            continue
        count = int(count_value)
        total_count += count
        if count == 0:
            continue
        rate = float("nan")
        for key in rate_keys:
            try:
                candidate = float(row.get(key, np.nan))
            except (TypeError, ValueError):
                continue
            if np.isfinite(candidate):
                rate = candidate
                break
        if not np.isfinite(rate) or not 0.0 <= rate <= 1.0:
            complete = False
            continue
        failures = int(round(rate * count))
        if not np.isclose(rate * count, failures, atol=1e-5, rtol=0.0):
            complete = False
            continue
        total_failures += failures
    rate = (total_failures / total_count
            if total_count > 0 and complete else float("nan"))
    return {"failure_count": total_failures, "pair_count": total_count,
            "rate": float(rate), "complete": bool(complete)}

def certificate_gate_status(metrics, c12_rate_limit=DEFAULT_C1C2_RATE_LIMIT,
                            c1_rate_limit=DEFAULT_C1C2_RATE_LIMIT,
                            c2_rate_limit=DEFAULT_C1C2_RATE_LIMIT,
                            min_c1_pairs=30, min_c2_pairs=30,
                            c3_state_violation_rate=None, c3_rate_limit=None):
    def first_number(keys):
        for key in keys:
            value = metrics.get(key)
            if value is not None:
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(value):
                    return value
        return float("nan")

    c1_rate = first_number(("c1_pair_violation_rate", "c1_pair_failure_rate"))
    c1_count = first_number(("c1_pair_count", "c1_safe_pair_count"))
    c2_rate = first_number(("c2_false_safe_rate", "c2_pair_violation_rate",
                            "c2_pair_false_safe_rate"))
    c2_count = first_number(("c2_pair_count", "c2_danger_pair_count"))
    c12_rate = first_number(("condition12_state_violation_rate",))
    c12_max_rate = first_number(("max_condition12_state_violation_rate",))
    if not np.isfinite(c12_rate):
        failures = first_number(("num_retrain_cbf_controller",))
        checked = first_number(("num_checked",))
        if np.isfinite(failures) and np.isfinite(checked) and checked > 0:
            c12_rate = failures / checked
    if not np.isfinite(c12_max_rate):
        c12_max_rate = c12_rate

    c3_rate = first_number(("c3_state_violation_rate",
                            "condition3_state_violation_rate",
                            "c3_active_pair_violation_rate"))
    if c3_state_violation_rate is not None:
        try:
            explicit_c3 = float(c3_state_violation_rate)
        except (TypeError, ValueError):
            explicit_c3 = float("nan")
        if np.isfinite(explicit_c3):
            c3_rate = explicit_c3
    c3_gate_enabled = c3_rate_limit is not None
    diagnostics = {
        "condition12_state_violation_rate": c12_rate,
        "max_condition12_state_violation_rate": c12_max_rate,
        "c12_rate_limit": float(c12_rate_limit),
        "c1_pair_violation_rate": c1_rate,
        "c1_pair_count": int(c1_count) if np.isfinite(c1_count) else 0,
        "c2_false_safe_rate": c2_rate,
        "c2_pair_count": int(c2_count) if np.isfinite(c2_count) else 0,
        "c1_rate_limit": float(c1_rate_limit),
        "c2_rate_limit": float(c2_rate_limit),
        "min_c1_pairs": int(min_c1_pairs),
        "min_c2_pairs": int(min_c2_pairs),
        "c3_state_violation_rate": c3_rate,
        "c3_rate_limit": (float(c3_rate_limit) if c3_gate_enabled else None),
        "c3_gate_enabled": bool(c3_gate_enabled),
    }
    observed_failure = (
        (np.isfinite(c12_rate) and c12_rate > c12_rate_limit)
        or (np.isfinite(c12_max_rate) and c12_max_rate > c12_rate_limit)
        or (np.isfinite(c1_rate) and c1_rate > c1_rate_limit
            and c1_count >= min_c1_pairs)
        or (np.isfinite(c2_rate) and c2_rate > c2_rate_limit
            and c2_count >= min_c2_pairs)
        or (c3_gate_enabled and np.isfinite(c3_rate)
            and c3_rate > float(c3_rate_limit))
    )

    c1_support_ok = np.isfinite(c1_count) and c1_count >= min_c1_pairs
    c2_support_ok = np.isfinite(c2_count) and c2_count >= min_c2_pairs
    c3_support_ok = (not c3_gate_enabled) or np.isfinite(c3_rate)
    support_adequate = (np.isfinite(c12_rate) and np.isfinite(c1_rate)
                        and np.isfinite(c2_rate) and c1_support_ok
                        and c2_support_ok and c3_support_ok)
    diagnostics["c1_support_adequate"] = bool(c1_support_ok)
    diagnostics["c2_support_adequate"] = bool(c2_support_ok)
    diagnostics["c1_pair_status"] = (
        "indeterminate" if not c1_support_ok
        else ("fail" if (np.isfinite(c1_rate) and c1_rate > c1_rate_limit) else "pass"))
    diagnostics["c2_pair_status"] = (
        "indeterminate" if not c2_support_ok
        else ("fail" if (np.isfinite(c2_rate) and c2_rate > c2_rate_limit) else "pass"))
    if observed_failure:
        status = "fail"
    elif not support_adequate:
        status = "indeterminate"
    else:
        status = "pass"
    diagnostics["status"] = status
    return diagnostics

def select_checkpoint_with_certificate_gate(
        candidates, c12_rate_limit=0.015, c1_rate_limit=0.015,
        c2_rate_limit=0.015, min_c1_pairs=30, min_c2_pairs=30,
        split="validation"):
    if str(split).lower() != "validation":
        raise ValueError("checkpoint selection is validation-only; test data are forbidden")
    if not candidates:
        raise ValueError("no checkpoint candidates supplied")
    enriched = []
    for candidate in candidates:
        metrics = candidate["metrics"]
        gate = certificate_gate_status(
            metrics, c12_rate_limit=c12_rate_limit,
            c1_rate_limit=c1_rate_limit, c2_rate_limit=c2_rate_limit,
            min_c1_pairs=min_c1_pairs, min_c2_pairs=min_c2_pairs)
        score = validation_score(metrics)
        try:
            c12 = float(metrics.get("max_condition12_state_violation_rate",
                                    metrics.get("condition12_state_violation_rate", np.nan)))
        except (TypeError, ValueError):
            c12 = float("nan")
        enriched.append((candidate, gate, c12, score))
    feasible = [item for item in enriched if item[1]["status"] == "pass"]
    if feasible:
        chosen = max(feasible, key=lambda x: (np.nan_to_num(x[3], nan=-np.inf), -x[2]))
        return chosen[0], True, chosen[1]
    chosen = min(enriched, key=lambda x: (
        np.inf if not np.isfinite(x[2]) else x[2],
        -np.nan_to_num(x[3], nan=-np.inf)))
    return chosen[0], False, chosen[1]

def select_checkpoint(candidates, c12_gate=0.015, split="validation"):
    if str(split).lower() != "validation":
        raise ValueError("checkpoint selection is validation-only; test data are forbidden")
    if not candidates:
        raise ValueError("no checkpoint candidates supplied")
    enriched = []
    for candidate in candidates:
        metrics = candidate["metrics"]
        c12 = float(metrics["condition12_state_violation_rate"])
        score = validation_score(metrics)
        enriched.append((candidate, c12, score))
    feasible = [x for x in enriched if np.isfinite(x[1]) and x[1] <= c12_gate]
    if feasible:
        chosen = max(feasible, key=lambda x: (np.nan_to_num(x[2], nan=-np.inf), -x[1]))
        return chosen[0], True
    chosen = min(enriched, key=lambda x: (np.inf if not np.isfinite(x[1]) else x[1],
                                          -np.nan_to_num(x[2], nan=-np.inf)))
    return chosen[0], False

def hierarchical_bootstrap(values, n_boot=10000, seed=0, confidence=0.95):
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or min(array.shape) < 2:
        raise ValueError("bootstrap input must be a 2-D matrix with >=2 clusters per axis")
    if not np.all(np.isfinite(array)):
        raise ValueError("bootstrap values must be finite")
    rng = np.random.RandomState(int(seed))
    samples = np.empty(int(n_boot), dtype=np.float64)
    for i in range(int(n_boot)):
        rows = rng.randint(0, array.shape[0], size=array.shape[0])
        cols = rng.randint(0, array.shape[1], size=array.shape[1])
        samples[i] = array[np.ix_(rows, cols)].mean()
    alpha = (1.0 - float(confidence)) / 2.0
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)),
        "ci_low": float(np.quantile(samples, alpha)),
        "ci_high": float(np.quantile(samples, 1.0 - alpha)),
        "n_pipeline_seeds": int(array.shape[0]),
        "n_scenes_per_pipeline": int(array.shape[1]),
    }

def paired_hierarchical_bootstrap(method_a, method_b, n_boot=10000, seed=0, confidence=0.95):
    a = np.asarray(method_a, dtype=np.float64)
    b = np.asarray(method_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("paired methods must share the same pipeline/scene grid")
    return hierarchical_bootstrap(a - b, n_boot=n_boot, seed=seed, confidence=confidence)

def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()

def write_json(path, payload):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=True), encoding="utf-8")
