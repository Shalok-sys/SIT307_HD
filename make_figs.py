import warnings, json; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

# Monochrome palette: the report is printed in black only, so figures use
# greyscale fills plus hatching to separate series.
C = dict(plum="#000000", terra="#000000", ochre="#FFFFFF", sage="#8C8C8C",
         cream="#F2F2F2", ink="#000000", mist="#D9D9D9")
plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.grid": True,
                     "grid.alpha": .25, "grid.color": "#999999",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "black", "text.color": "black",
                     "axes.labelcolor": "black", "xtick.color": "black",
                     "ytick.color": "black", "font.family": "DejaVu Sans",
                     "hatch.linewidth": 0.6})

P1 = json.load(open("results_part1.json"))
P2 = json.load(open("results_part2.json"))
df = pd.read_csv("data/heart1025.csv")
FEAT = list(df.columns[:-1])
groups = pd.factorize(df[FEAT].astype(str).agg("|".join, axis=1))[0]
ORDER = ["LR", "DT", "RF", "XGB", "NB", "KNN", "Stacking", "DPCS"]

# ---------------- fig1: dataset anatomy ----------------
fig, ax = plt.subplots(1, 3, figsize=(13, 3.5))
mult = pd.Series(groups).value_counts().value_counts().sort_index()
b = ax[0].bar(mult.index.astype(str), mult.values, color="#4D4D4D", edgecolor="black", width=.62)
for bb, v in zip(b, mult.values):
    ax[0].text(bb.get_x() + bb.get_width()/2, v + 2, str(v), ha="center", fontsize=7.5)
ax[0].set_xlabel("copies of the same patient record"); ax[0].set_ylabel("number of patients")
ax[0].set_title("Every one of the 302 patients\nappears more than once", fontsize=9.5)

vals = [1025, 302]
b = ax[1].bar(["rows in the file", "distinct patients"], vals,
              color=["#4D4D4D", "white"], edgecolor="black", width=.5)
for bb, v in zip(b, vals):
    ax[1].text(bb.get_x() + bb.get_width()/2, v + 20, f"{v:,}", ha="center", fontsize=9)
ax[1].set_ylim(0, 1180); ax[1].set_ylabel("count")
ax[1].set_title(f"{P1['dataset']['duplicate_share']:.1f} % of rows are exact duplicates", fontsize=9.5)

tw = P2["leakage"]["test_rows_with_a_twin_in_train"]
ax[2].barh(["has a twin in training", "genuinely unseen"], [tw, 100 - tw],
           color=["#4D4D4D", "white"], edgecolor="black", height=.5)
ax[2].text(tw - 4, 0, f"{tw:.1f} %", va="center", ha="right", color="white", fontsize=11)
ax[2].text(100 - tw + 2, 1, f"{100-tw:.1f} %", va="center", fontsize=9)
ax[2].set_xlim(0, 108); ax[2].set_xlabel("% of the 205 test rows")
ax[2].set_title("Under a random 80:20 split the test set\nis almost entirely memorised", fontsize=9.5)
plt.tight_layout(); plt.savefig("fig1_dataset.png", bbox_inches="tight"); plt.close()

# ---------------- fig2: reproduction vs paper ----------------
rep = P1["reproduction_seed42"]; paper = P1["paper_reported"]
fig, ax = plt.subplots(1, 2, figsize=(13, 3.8))
names = ["LR", "DT", "RF", "XGB", "NB", "KNN", "Stacking"]
xp = np.arange(len(names)); w = .38
ours = [rep[n]["Accuracy"] for n in names]
pap = [paper[n] if paper[n] else np.nan for n in names]
ax[0].bar(xp - w/2, ours, w, color="#4D4D4D", edgecolor="black", label="this reproduction")
ax[0].bar(xp + w/2, pap, w, color="white", edgecolor="black", hatch="///", label="reported in the paper")
for i, v in enumerate(ours):
    ax[0].text(i - w/2, v + .006, f"{v:.3f}", ha="center", fontsize=7)
for i, v in enumerate(pap):
    if not np.isnan(v): ax[0].text(i + w/2, v + .006, f"{v:.3f}", ha="center", fontsize=7)
ax[0].set_xticks(xp); ax[0].set_xticklabels(names, fontsize=8)
ax[0].set_ylim(.7, 1.06); ax[0].set_ylabel("test accuracy")
ax[0].legend(fontsize=7.5, loc="lower right")
ax[0].set_title("Single 80:20 split, seed 42. The paper does not report LR, NB or KNN", fontsize=9.5)

raw = P1["seed_sensitivity_raw"]
data = [raw[n]["Accuracy"] for n in names]
bp = ax[1].boxplot(data, labels=names, patch_artist=True, widths=.55)
for patch, n in zip(bp["boxes"], names):
    patch.set_facecolor("#4D4D4D" if n in ("DT", "RF", "XGB", "Stacking") else "#D9D9D9")
    patch.set_alpha(.75)
for m in bp["medians"]: m.set_color("black")
ax[1].tick_params(labelsize=8); ax[1].set_ylabel("test accuracy")
ax[1].set_title("30 random splits. Dark = models able to memorise duplicates", fontsize=9.5)
plt.tight_layout(); plt.savefig("fig2_reproduction.png", bbox_inches="tight"); plt.close()

# ---------------- fig3: protocol A vs B ----------------
r, g = P2["random_cv"], P2["grouped_cv"]
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
xp = np.arange(len(ORDER)); w = .38
ra = [r[n]["Accuracy"]["mean"] for n in ORDER]
ga = [g[n]["Accuracy"]["mean"] for n in ORDER]
rs = [r[n]["Accuracy"]["sd"] for n in ORDER]
gs = [g[n]["Accuracy"]["sd"] for n in ORDER]
ax[0].bar(xp - w/2, ra, w, yerr=rs, capsize=2, color="white", edgecolor="black", hatch="///", label="groups ignored (paper style)")
ax[0].bar(xp + w/2, ga, w, yerr=gs, capsize=2, color="#4D4D4D", edgecolor="black", label="leakage aware grouped CV")
ax[0].set_xticks(xp); ax[0].set_xticklabels(ORDER, fontsize=8, rotation=20)
ax[0].set_ylim(.6, 1.06); ax[0].set_ylabel("accuracy, mean of 25 folds")
ax[0].legend(fontsize=7.5, loc="lower left")
ax[0].set_title("The same models under the two protocols", fontsize=9.5)

drop = [ra[i] - ga[i] for i in range(len(ORDER))]
cols = ["#333333" if d > .15 else "#BFBFBF" for d in drop]
o = np.argsort(drop)
ax[1].barh([ORDER[i] for i in o], [drop[i] for i in o], color=[cols[i] for i in o], height=.6)
for k, i in enumerate(o):
    ax[1].text(drop[i] + .004, k, f"{drop[i]:+.4f}", va="center", fontsize=8)
ax[1].set_xlim(0, .30); ax[1].set_xlabel("accuracy lost when the leakage is removed")
ax[1].set_title("The drop measures how much each model was memorising", fontsize=9.5)
plt.tight_layout(); plt.savefig("fig3_protocol.png", bbox_inches="tight"); plt.close()

# ---------------- fig4: DPCS internals ----------------
info = P2["dpcs_example"]
fig, ax = plt.subplots(1, 2, figsize=(12.5, 4))
au = pd.Series(info["aucs"]).sort_values()
cols = ["#333333" if k in info["selected"] else "#CCCCCC" for k in au.index]
ax[0].barh(au.index, au.values, color=cols, height=.6)
for i, v in enumerate(au.values):
    ax[0].text(v + .004, i, f"{v:.4f}", va="center", fontsize=8)
ax[0].set_xlim(.6, .95); ax[0].set_xlabel("inner, group aware out of fold AUC")
ax[0].set_title(f"Base learner quality. Dark = kept by DPCS ({', '.join(info['selected'])})", fontsize=9.5)

ks = list(info["err_corr"].keys()); names6 = ["LR", "DT", "RF", "XGB", "NB", "KNN"]
M = np.eye(6)
for k, v in info["err_corr"].items():
    a, b_ = k.split("|"); i, j = names6.index(a), names6.index(b_)
    M[i, j] = M[j, i] = v
im = ax[1].imshow(M, cmap="Greys", vmin=0, vmax=1)
ax[1].set_xticks(range(6)); ax[1].set_xticklabels(names6, fontsize=8)
ax[1].set_yticks(range(6)); ax[1].set_yticklabels(names6, fontsize=8)
for i in range(6):
    for j in range(6):
        ax[1].text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=7,
                   color="white" if M[i, j] > .55 else "black")
ax[1].set_title("Pairwise error correlation, the diversity term", fontsize=9.5); ax[1].grid(False)
plt.colorbar(im, ax=ax[1], fraction=.046)
plt.tight_layout(); plt.savefig("fig4_dpcs.png", bbox_inches="tight"); plt.close()

# ---------------- fig5: final comparison ----------------
mets = ["Accuracy", "Precision", "Recall", "F1", "AUC"]
sel = ["Stacking", "DPCS", "LR"]
lab = ["paper style stacking", "proposed DPCS", "logistic regression"]
colr = ["white", "#4D4D4D", "#B3B3B3"]
fig, ax = plt.subplots(figsize=(9, 4))
xp = np.arange(len(mets)); w = .26
for k, (n, l, c) in enumerate(zip(sel, lab, colr)):
    vals = [g[n][m]["mean"] for m in mets]
    errs = [g[n][m]["sd"] for m in mets]
    ax.bar(xp + (k - 1) * w, vals, w, yerr=errs, capsize=2, color=c,
           edgecolor="black", hatch="///" if k == 0 else None, label=l)
    for i, v in enumerate(vals):
        ax.text(i + (k - 1) * w, v + .012, f"{v:.3f}", ha="center", fontsize=6.8)
ax.set_xticks(xp); ax.set_xticklabels(mets)
ax.set_ylim(.6, 1.0); ax.set_ylabel("mean over 25 leakage aware folds")
ax.legend(fontsize=8); ax.set_title("Final comparison under the honest protocol", fontsize=10)
plt.tight_layout(); plt.savefig("fig5_final.png", bbox_inches="tight"); plt.close()
print("figures written")
