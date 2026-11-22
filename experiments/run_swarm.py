"""Swarm study: coverage scaling, vehicle loss, sensor faults, and the
screen-then-confirm policy with real classifier outputs replayed per zone.

Inputs (written by the training scripts):
  <cache>/preds/codebrim_<vision tag>.npz   test-split probabilities
  <cache>/preds/ascan_<ut tag>.npz          cross-specimen A-scan probabilities

Usage: python experiments/run_swarm.py --vision <tag> --ut Shear_Rot00 --trials 200
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataclasses import replace

from experiments.swarm_params import CARRIER, COUPLING, SCENARIO, SCOUT
from inspectswarm import swarm as S
from inspectswarm.manifest import save_result
from inspectswarm.paths import CACHE, RESULTS

UT_TEST = ("pk266", "pk401")  # specimens with objects; each scored by a model that never saw it


def load_pools(vision_tag, ut_tag):
    t = np.load(CACHE / "codebrim_test.npz")
    p = np.load(CACHE / "preds" / f"codebrim_{vision_tag}.npz")["test"]
    p_any = 1 - np.prod(1 - p, axis=1)
    dmg = t["y"].sum(1) > 0
    u = np.load(CACHE / "preds" / f"ascan_{ut_tag}.npz")
    pos, neg = [], []
    for s in UT_TEST:
        m, lab, pr = u[f"valid_{s}"], u[f"label_{s}"], u[f"p_{s}"]
        pos.append(pr[m & (lab == 1)])
        neg.append(pr[m & (lab == 0)])
    return {"vis_dmg": p_any[dmg], "vis_clean": p_any[~dmg], "ut_pos": np.concatenate(pos), "ut_neg": np.concatenate(neg)}


def world(rng, pools, k=3, **over):
    c = {**COUPLING, **over}
    return S.make_world(rng, SCENARIO["nx"], SCENARIO["ny"], SCENARIO["zone"], prevalence=c["prevalence"],
                        p_hidden_damaged=c["p_hidden_damaged"], p_hidden_clean=c["p_hidden_clean"],
                        vis_prob_damaged=pools["vis_dmg"], vis_prob_clean=pools["vis_clean"],
                        ut_prob_pos=pools["ut_pos"], ut_prob_neg=pools["ut_neg"], k_contacts=k,
                        cluster=c.get("cluster", 0.5))


def mean_std(rows):
    keys = rows[0].keys()
    return {k: {"mean": float(np.nanmean([r[k] for r in rows])), "std": float(np.nanstd([r[k] for r in rows]))} for k in keys}


# ----------------------------------------------------------- experiments

def coverage_scaling(pools, trials, seed):
    """Tier 1 time, sorties and energy against the number of scouts and the
    per-sortie loss probability."""
    out = {}
    for p_loss in (0.0, 0.05, 0.10, 0.20):
        plat = replace(SCOUT, p_loss=p_loss)
        for n in (1, 2, 4, 8, 12, 16):
            rows = []
            for t in range(trials if p_loss else 1):
                rng = np.random.default_rng(seed + t)
                w = world(rng, pools)
                r = S.run_tier(w, S.split_path(S.boustrophedon(w.nx, w.ny), n), plat, rng)
                rows.append({"time_min": r["time"] / 60, "sorties": r["sorties"], "energy_wh": r["energy_wh"],
                             "lost": r["lost"], "coverage": float((~np.isnan(r["visit"])).mean())})
            out[f"loss{p_loss}_n{n}"] = mean_std(rows)
    return out


def confirm_policies(pools, trials, seed, k=3, ut_noise=0.0, tau=0.5, n_scouts=8, **coupling):
    """Hidden-defect detection and cost for each tier-2 selection policy."""
    c = {**COUPLING, **coupling}
    base_rate = 0.5  # ultrasonic outputs are calibrated to a 50% prior, so their odds are likelihood ratios
    acc = {}
    for t in range(trials):
        rng = np.random.default_rng(seed + t)
        w = world(rng, pools, k=k, **coupling)
        if ut_noise > 0:  # degrade the second modality: blend its output with uninformative noise
            w.p_ut = (1 - ut_noise) * w.p_ut + ut_noise * rng.random(w.p_ut.shape)
        t1 = S.run_tier(w, S.split_path(S.boustrophedon(w.nx, w.ny), n_scouts), SCOUT, rng)
        prior = S.hidden_prior(w.p_vision, c["p_hidden_damaged"], c["p_hidden_clean"])
        flagged = np.where(w.p_vision >= tau)[0]
        budget = len(flagged)
        order = np.argsort(-prior)
        policies = {
            "vision_only": np.array([], int),
            "exhaustive": np.arange(w.n),
            "flagged": flagged,
            "flagged_plus_10pct_audit": np.union1d(flagged, rng.choice(np.setdiff1d(np.arange(w.n), flagged),
                                                                       max(1, int(0.1 * (w.n - budget))), replace=False)),
            "random_same_budget": rng.choice(w.n, budget, replace=False),
            "oracle_surface_truth": np.where(w.surface_damage)[0],
            "top_prior_half_budget": order[: max(1, budget // 2)],
        }
        for name, zones in policies.items():
            post = prior.copy()
            cost = {"time": 0.0, "energy_wh": 0.0, "sorties": 0}
            if len(zones):
                post[zones] = S.fuse(prior[zones], w.p_ut[zones], base_rate)
                tour = zones[S.nearest_neighbour_tour(w.xy(zones), (0.0, 0.0))]
                plat = replace(CARRIER, dwell=CARRIER.dwell * k)
                cost = S.run_tier(w, [tour], plat, rng)
            m = S.detection_metrics(w.hidden_defect, post >= 0.5)
            m2 = S.detection_metrics(w.hidden_defect, post >= 0.2)  # cautious operating point
            acc.setdefault(name, []).append({
                "hidden_recall": m["recall"], "hidden_precision": m["precision"], "hidden_f1": m["f1"],
                "hidden_recall_thr02": m2["recall"], "hidden_precision_thr02": m2["precision"],
                "false_positives": m["fp"], "zones_contacted": len(zones), "contacts": len(zones) * k,
                "tier2_time_min": cost["time"] / 60, "tier2_energy_wh": cost["energy_wh"],
                "tier1_time_min": t1["time"] / 60, "tier1_energy_wh": t1["energy_wh"],
                "total_time_min": (t1["time"] + cost["time"]) / 60, "n_hidden": int(w.hidden_defect.sum())})
    return {k_: mean_std(v) for k_, v in acc.items()}


def sensor_fault(pools, trials, seed, n_scouts=8):
    """One scout has a faulty camera pipeline (uninformative outputs).
    Compare single coverage, double coverage with averaging, and double
    coverage with peer cross-check that isolates the faulty scout."""
    acc = {}
    for t in range(trials):
        rng = np.random.default_rng(seed + t)
        w = world(rng, pools)
        path = S.boustrophedon(w.nx, w.ny)
        own1 = np.empty(w.n, int)
        for a, z in enumerate(S.split_path(path, n_scouts)):
            own1[z] = a
        own2 = (own1 + 1 + (np.arange(w.n) % (n_scouts - 1))) % n_scouts  # second look by a different scout
        v1 = w.p_vision
        v2 = np.where(w.surface_damage, rng.choice(pools["vis_dmg"], w.n), rng.choice(pools["vis_clean"], w.n))
        bad = int(rng.integers(n_scouts))
        f1 = np.where(own1 == bad, rng.random(w.n), v1)
        f2 = np.where(own2 == bad, rng.random(w.n), v2)
        # peer cross-check: a scout's disagreement rate with the second look at its zones
        dis = np.array([np.mean(np.concatenate([(f1 >= .5)[own1 == a] != (f2 >= .5)[own1 == a],
                                                (f2 >= .5)[own2 == a] != (f1 >= .5)[own2 == a]])) for a in range(n_scouts)])
        med, mad = np.median(dis), np.median(np.abs(dis - np.median(dis))) + 1e-9
        suspects = np.where((dis - med) / (1.4826 * mad) > 3.5)[0]
        ok1, ok2 = ~np.isin(own1, suspects), ~np.isin(own2, suspects)
        fused = np.where(ok1 & ok2, (f1 + f2) / 2, np.where(ok1, f1, np.where(ok2, f2, (f1 + f2) / 2)))
        variants = {"no_fault_single": v1, "fault_single": f1, "fault_double_average": (f1 + f2) / 2,
                    "fault_double_crosscheck": fused, "no_fault_double_average": (v1 + v2) / 2}
        for name, p in variants.items():
            m = S.detection_metrics(w.surface_damage, p >= 0.5)
            acc.setdefault(name, []).append({"recall": m["recall"], "precision": m["precision"], "f1": m["f1"]})
        acc.setdefault("diagnosis", []).append({"faulty_scout_isolated": float(bad in suspects),
                                                "healthy_scouts_wrongly_isolated": float(len(set(suspects) - {bad}))})
    return {k: mean_std(v) for k, v in acc.items()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--vision", required=True)
    ap.add_argument("--ut", default="Shear_Rot00")
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    pools = load_pools(a.vision, a.ut)
    res = {"pool_sizes": {k: int(len(v)) for k, v in pools.items()},
           "tier_accuracy": {
               "vision_damage_recall": float((pools["vis_dmg"] >= .5).mean()),
               "vision_false_alarm": float((pools["vis_clean"] >= .5).mean()),
               "ut_recall_single_contact": float((pools["ut_pos"] >= .5).mean()),
               "ut_false_alarm_single_contact": float((pools["ut_neg"] >= .5).mean())}}
    print(res["tier_accuracy"])
    res["coverage_scaling"] = coverage_scaling(pools, a.trials, a.seed)
    res["confirm_policies"] = confirm_policies(pools, a.trials, a.seed)
    for name, r in res["confirm_policies"].items():
        print(f"{name:28s} recall {r['hidden_recall']['mean']:.3f} prec {r['hidden_precision']['mean']:.3f} "
              f"contacts {r['contacts']['mean']:.0f} tier2 {r['tier2_time_min']['mean']:.1f} min")
    res["sensitivity"] = {
        "k_contacts": {str(k): confirm_policies(pools, a.trials // 2, a.seed, k=k) for k in (1, 2, 3, 5)},
        "ut_noise": {str(n): confirm_policies(pools, a.trials // 2, a.seed, ut_noise=n) for n in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)},
        "lift": {f"{pd}_{pc}": confirm_policies(pools, a.trials // 2, a.seed, p_hidden_damaged=pd, p_hidden_clean=pc)
                 for pd, pc in ((0.3, 0.3 * 0.2 / 0.2), (0.3, 0.10), (0.3, 0.03), (0.3, 0.01), (0.5, 0.01))},
        "prevalence": {str(pv): confirm_policies(pools, a.trials // 2, a.seed, prevalence=pv) for pv in (0.05, 0.1, 0.2, 0.4)},
        "vision_threshold": {str(tau): confirm_policies(pools, a.trials // 2, a.seed, tau=tau) for tau in (0.2, 0.35, 0.5, 0.65, 0.8)},
    }
    res["sensor_fault"] = sensor_fault(pools, a.trials, a.seed)
    print({k: {m: round(v["mean"], 3) for m, v in r.items()} for k, r in res["sensor_fault"].items()})
    save_result(RESULTS / "swarm" / f"swarm_{a.vision}__{a.ut}.json", res, vars(a), a.seed)
