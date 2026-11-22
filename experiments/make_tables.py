"""Print the paper's tables as Markdown from the JSON files in results_v2/.

Usage: python experiments/make_tables.py > results_v2/tables.md
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inspectswarm.paths import RESULTS

NICE = {"convnext_tiny": "ConvNeXt-Tiny", "resnet50": "ResNet50", "mobilenetv3_small_100": "MobileNetV3-Small",
        "nano32": "NanoNet-32", "nano16": "NanoNet-16"}


def load(rel):
    p = RESULTS / rel
    return json.loads(p.read_text())["result"] if p.exists() else None


def pm(mean, std, scale=100, d=1):
    return f"{scale * mean:.{d}f} ± {scale * std:.{d}f}"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def vision():
    groups = {}
    for f in sorted(glob.glob(str(RESULTS / "vision" / "codebrim_*.json"))):
        r = json.loads(Path(f).read_text())["result"]
        tag = r["tag"].rsplit("_s", 1)[0]
        name, rest = tag.split("_r", 1)
        res = rest.split("_")[0]
        groups.setdefault((name, res, "_kd-" in tag), {"params": r["params"], "runs": []})["runs"].append(r["test"])
    order = list(NICE)
    rows = []
    for (name, res, kd), g in sorted(groups.items(), key=lambda kv: (order.index(kv[0][0]), kv[0][2])):
        def ms(k, d=1):
            a = np.array([t[k] for t in g["runs"]])
            return pm(a.mean(), a.std(), d=d)

        rows.append([NICE[name], f"{g['params'] / 1e6:.2f}", res, "yes" if kd else "no", len(g["runs"]),
                     ms("multi_target_acc"), ms("macro_f1"), ms("damage_recall"), ms("false_alarm_rate")])
    print("**Source table.** CODEBRIM test results, mean ± standard deviation over seeds (%).\n")
    print(table(["Model", "Parameters (M)", "Input (px)", "Distilled", "Seeds", "Multi-target accuracy", "Macro F1",
                 "Damage recall", "False alarms on background"], rows))
    pc = {}
    for (name, res, kd), g in groups.items():
        if name in ("convnext_tiny", "nano32", "nano16", "mobilenetv3_small_100") and (kd or name == "convnext_tiny"):
            pc[NICE[name]] = {c: np.mean([t["per_class"][c]["f1"] for t in g["runs"]]) for c in g["runs"][0]["per_class"]}
    if pc:
        classes = list(next(iter(pc.values())))
        print("**Source table.** Per-class F1 on the CODEBRIM test set (teacher and distilled students, mean over seeds).\n")
        print(table(["Model"] + classes, [[k] + [f"{v[c]:.3f}" for c in classes] for k, v in pc.items()]))


def edge():
    e = load("vision/edge_benchmark.json")
    if not e:
        return
    rows = []
    for tag, r in e.items():
        name = tag.split("_r")[0]
        rows.append([NICE[name] + (" (distilled)" if "_kd-" in tag else ""), r["res"], f"{r['fp32']['size_kb'] / 1024:.2f}",
                     f"{r['int8']['size_kb'] / 1024:.2f}", f"{100 * r['fp32']['multi_target_acc']:.1f}",
                     f"{100 * r['int8']['multi_target_acc']:.1f}", f"{r['fp32']['latency_ms_median']:.1f}",
                     f"{r['int8']['latency_ms_median']:.1f}"])
    print("**Source table.** Edge budget of the seed-0 models: file size, multi-target accuracy and median single-image "
          "latency on one laptop CPU thread, for 32-bit float and 8-bit integer versions.\n")
    print(table(["Model", "Input (px)", "Size float (MB)", "Size 8-bit (MB)", "Accuracy float (%)", "Accuracy 8-bit (%)",
                 "Latency float (ms)", "Latency 8-bit (ms)"], rows))


def leak():
    r = load("vision/leak_study.json")
    if not r:
        return
    s = r["summary"]
    rows = []
    for key, name in (("tile_model_on_tile_val", "Original tile split (leaky)"),
                      ("grouped_model_on_grouped_val", "Grouped split, originals only")):
        rows.append([name] + [pm(s[f"{key}.{m}"]["mean"], s[f"{key}.{m}"]["std"]) if m != "macro_f1" else
                              pm(s[f"{key}.{m}"]["mean"], s[f"{key}.{m}"]["std"], 1, 3)
                              for m in ("acc", "macro_f1", "acc_sdnet_source", "acc_raidathmane_source")])
    print("**Source table.** The v1 recipe retrained under two protocols (three seeds, mean ± standard deviation).\n")
    print(table(["Protocol", "Accuracy (%)", "Macro F1", "Accuracy on crack/intact source (%)",
                 "Accuracy on corrosion/spalling source (%)"], rows))
    for k in ("tile_model_on_tile_val", "grouped_model_on_grouped_val"):
        print(k, "per-class F1 mean:", {c: round(float(np.mean([run[k]["f1"][c] for run in r["runs"]])), 3)
                                         for c in r["runs"][0][k]["f1"]}, " n =", r["runs"][0][k]["n"])
    print()


def cross():
    r = load("vision/cross_source.json")
    if not r:
        return
    print("**Cross-source A** (CODEBRIM models on SDNET2018 / Raidathmane originals), n =", r["n_external"], "\n")
    keys = ["damage_recall", "false_alarm_rate_intact", "balanced_acc_damage_vs_intact", "flag_recall_crack",
            "class_recall_crack", "flag_recall_corrosion", "class_recall_corrosion", "flag_recall_spalling", "class_recall_spalling"]
    print(table(["Model"] + keys, [[t] + [f"{100 * v[k]:.1f}" for k in keys] for t, v in r["A_codebrim_models_on_sdnet_raidathmane"].items()]))
    print("**Cross-source B** (v1.0 classifier on CODEBRIM test):", json.dumps(r["B_v1_model_on_codebrim_test"], indent=0), "\n")


def ultrasonic():
    rows = []
    for f in sorted(glob.glob(str(RESULTS / "ultrasonic" / "detect_*.json"))):
        r = json.loads(Path(f).read_text())["result"]
        tag = Path(f).stem.replace("detect_", "").replace("_Rot", ", ").replace("Long", "Longitudinal")
        fo = r["folds"]
        rows.append([tag + "°"] + [pm(fo[s][k]["mean"], fo[s][k]["std"], 1, 3) for s in ("pk266", "pk401") for k in ("auroc", "f1")]
                    + [f"{fo[s]['classical']['auroc']:.3f}" for s in ("pk266", "pk401")]
                    + [pm(fo[s]["specificity"]["mean"], fo[s]["specificity"]["std"]) for s in ("pk266", "pk401", "pk050")])
    if rows:
        print("**Source table.** Embedded-object detection from single A-scans, cross-specimen (three seeds).\n")
        print(table(["Wave, orientation", "AUROC ducts (Pk266)", "F1 ducts", "AUROC cuboids (Pk401)", "F1 cuboids",
                     "Baseline AUROC ducts", "Baseline AUROC cuboids", "Specificity Pk266 (%)", "Specificity Pk401 (%)",
                     "Specificity plain Pk050 (%)"], rows))
        for f in sorted(glob.glob(str(RESULTS / "ultrasonic" / "detect_*.json"))):
            r = json.loads(Path(f).read_text())["result"]["folds"]
            print(Path(f).stem, {s: {cv: round(float(np.mean([run["recall_by_cover_mm"][cv] for run in r[s]["runs"]])), 2)
                                     for cv in r[s]["runs"][0]["recall_by_cover_mm"]} for s in ("pk266", "pk401")},
                  "recall", {s: round(r[s]["recall"]["mean"], 3) for s in ("pk266", "pk401")},
                  "precision", {s: round(r[s]["precision"]["mean"], 3) for s in ("pk266", "pk401")},
                  "uncal F1", {s: round(float(np.mean([run["uncalibrated"]["f1"] for run in r[s]["runs"]])), 3) for s in ("pk266", "pk401")})
        print()
    t = load("ultrasonic/thickness.json")
    if t:
        rows = []
        for k, v in t.items():
            for s in ("pk266", "pk401"):
                e = v[s]["plain_k1"]
                e5 = v[s]["plain_k5"]
                rows.append([k.replace("_Rot", ", ").replace("Long", "Longitudinal") + "°", f"{v['c_ms']:.0f}", s.replace("pk", "Pk"), e["n"],
                             f"{e['median_ae_mm']:.1f}", f"{e['mae_mm']:.1f}", f"{100 * e['within_10mm']:.0f}",
                             f"{100 * e['within_5pct']:.0f}", f"{e5['median_ae_mm']:.1f}", f"{100 * e5['within_5pct']:.0f}"])
        print("**Source table.** Thickness from single A-scans with the wave speed calibrated on Pk050 only.\n")
        print(table(["Wave, orientation", "c from Pk050 (m/s)", "Test specimen", "A-scans", "Median error (mm)", "Mean error (mm)",
                     "Within 10 mm (%)", "Within 5% (%)", "Median error, 5 contacts (mm)", "Within 5%, 5 contacts (%)"], rows))
        print("Pk050 step 2 check:", {k: round(v["pk050_step2_check"]["estimated_median_mm"], 1) for k, v in t.items()}, "\n")


def swarm():
    for f in sorted(glob.glob(str(RESULTS / "swarm" / "swarm_*.json"))):
        r = json.loads(Path(f).read_text())["result"]
        print(f"### {Path(f).stem}\n\ntier accuracy: {json.dumps({k: round(v, 3) for k, v in r['tier_accuracy'].items()})}\n")
        cs = r["coverage_scaling"]
        ns = (1, 2, 4, 8, 12, 16)
        rows = [[n, f"{cs[f'loss0.0_n{n}']['time_min']['mean']:.1f}", f"{cs[f'loss0.0_n{n}']['sorties']['mean']:.0f}",
                 f"{cs[f'loss0.0_n{n}']['energy_wh']['mean']:.2f}"] +
                [f"{cs[f'loss{p}_n{n}']['time_min']['mean']:.1f} / {100 * cs[f'loss{p}_n{n}']['coverage']['mean']:.1f}" for p in (0.05, 0.1, 0.2)]
                for n in ns]
        print("**Source table.** Tier 1 coverage of 480 zones against the number of scouts.\n")
        print(table(["Scouts", "Time (min)", "Sorties", "Flight energy (Wh)", "5% loss: time (min) / coverage (%)",
                     "10% loss: time / coverage", "20% loss: time / coverage"], rows))
        pol = r["confirm_policies"]
        names = {"vision_only": "Vision only", "random_same_budget": "Random zones, same budget", "top_prior_half_budget": "Highest prior, half budget",
                 "flagged": "Flagged zones", "flagged_plus_10pct_audit": "Flagged + 10% audit", "exhaustive": "Exhaustive",
                 "oracle_surface_truth": "Perfect scout (oracle)"}
        rows = [[lab, f"{pol[k]['contacts']['mean']:.0f}", f"{pol[k]['tier2_time_min']['mean']:.0f}", f"{pol[k]['tier2_energy_wh']['mean']:.0f}",
                 pm(pol[k]["hidden_recall"]["mean"], pol[k]["hidden_recall"]["std"], d=0),
                 pm(pol[k]["hidden_precision"]["mean"], pol[k]["hidden_precision"]["std"], d=0),
                 f"{100 * pol[k]['hidden_recall_thr02']['mean']:.0f}", f"{100 * pol[k]['hidden_precision_thr02']['mean']:.0f}"]
                for k, lab in names.items()]
        print("**Source table.** Confirmation policies (mean ± standard deviation over simulated structures).\n")
        print(table(["Policy", "Contacts", "Tier 2 time (min)", "Tier 2 energy (Wh)", "Hidden-defect recall (%)", "Precision (%)",
                     "Recall at threshold 0.2 (%)", "Precision at threshold 0.2 (%)"], rows))
        print("n_hidden mean:", round(pol["flagged"]["n_hidden"]["mean"], 1), " tier1 time:", round(pol["flagged"]["tier1_time_min"]["mean"], 1),
              "min, tier1 energy:", round(pol["flagged"]["tier1_energy_wh"]["mean"], 2), "Wh\n")
        for sname, sens in r["sensitivity"].items():
            rows = [[k] + [f"{100 * v[p]['hidden_recall']['mean']:.0f} / {100 * v[p]['hidden_precision']['mean']:.0f} / {v[p]['contacts']['mean']:.0f}"
                           for p in ("flagged", "random_same_budget", "exhaustive")] for k, v in sens.items()]
            print(f"Sensitivity: {sname} (recall % / precision % / contacts)\n")
            print(table([sname, "flagged", "random same budget", "exhaustive"], rows))
        sf = r["sensor_fault"]
        print("**Source table.** One faulty scout among eight.\n")
        print(table(["Configuration", "Recall (%)", "Precision (%)", "F1 (%)"],
                    [[k] + [pm(v[m]["mean"], v[m]["std"]) for m in ("recall", "precision", "f1")] for k, v in sf.items() if k != "diagnosis"]))
        print("diagnosis:", json.dumps({k: round(v["mean"], 3) for k, v in sf["diagnosis"].items()}), "\n")


if __name__ == "__main__":
    for fn in (leak, vision, edge, cross, ultrasonic, swarm):
        fn()
