"""Builds README charts from reports/metrics.json ->  docs/accuracy_summary.png, docs/error_by_case.png"""
import json, argparse, os, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt, numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("--metrics", default="reports/metrics.json"); ap.add_argument("--out", default="docs")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); R = json.load(open(a.metrics))
T = ["max_deformation_mm", "von_mises_MPa", "max_principal_MPa", "factor_of_safety"]
lab = ["Max\ndeformation", "von Mises\nstress", "Max principal\nstress", "Factor of\nsafety"]

fig, ax = plt.subplots(figsize=(7.5, 4)); x = np.arange(4); w = .27
for i, (k, nm, c) in enumerate([("median_APE_pct", "Median error", "#2a9d8f"), ("MAPE_pct", "Mean error", "#e9c46a"), ("p95_APE_pct", "95th-percentile error", "#e76f51")]):
    v = [R[t]["test"][k] for t in T]; b = ax.bar(x + (i - 1) * w, v, w, label=nm, color=c)
    ax.bar_label(b, fmt="%.1f", fontsize=8, padding=2)
ax.set_xticks(x); ax.set_xticklabels(lab); ax.set_ylabel("ML vs ANSYS error (%)"); ax.set_title("Held-out test accuracy of the ML surrogate")
ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False); plt.tight_layout(); plt.savefig(f"{a.out}/accuracy_summary.png", dpi=150); plt.close()

d = R["by_case_MAPE_pct"]["von_mises_MPa"]; items = sorted(d.items(), key=lambda kv: kv[1])
fig, ax = plt.subplots(figsize=(7.5, 4.6)); col = ["#2a9d8f" if v < 3 else "#e9c46a" if v < 6 else "#e76f51" for _, v in items]
b = ax.barh([k.replace("/", "  |  ") for k, _ in items], [v for _, v in items], color=col); ax.bar_label(b, fmt="%.1f%%", fontsize=8, padding=3)
ax.set_xlabel("Mean von Mises error vs ANSYS (%)"); ax.set_title("Error by load | support | section (test set)")
ax.spines[["top", "right"]].set_visible(False); plt.tight_layout(); plt.savefig(f"{a.out}/error_by_case.png", dpi=150); plt.close()
print("wrote", a.out)
