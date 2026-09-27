"""Part 2: Diversity-Pruned Calibrated Stacking under a leakage-aware protocol.

Two contributions:
  (1) A grouped evaluation protocol. Every distinct feature vector is one patient
      group, so the 1025 rows collapse to 302 groups. StratifiedGroupKFold keeps
      all copies of a patient on the same side of every split, which removes the
      duplicate leakage that inflates the published result.
  (2) DPCS, a stacking design built for the honest protocol: base learners are
      pruned by an accuracy and diversity criterion, the survivors are probability
      calibrated, the meta learner sees calibrated out of fold probabilities, and
      the decision threshold is tuned on inner folds rather than left at 0.5.
"""
import warnings, json, itertools
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import clone
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, brier_score_loss)

RS = 42
df = pd.read_csv("data/heart1025.csv")
FEAT = list(df.columns[:-1])
X = df[FEAT].values
y = df["target"].values

# ---- patient grouping: identical feature vectors are the same patient ----
key = df[FEAT].astype(str).agg("|".join, axis=1)
groups = pd.factorize(key)[0]
R = {}
R["grouping"] = {
    "rows": int(len(df)), "groups": int(len(np.unique(groups))),
    "mean_copies_per_group": round(float(len(df) / len(np.unique(groups))), 3),
    "max_copies": int(pd.Series(groups).value_counts().max()),
    "singleton_groups": int((pd.Series(groups).value_counts() == 1).sum()),
}
print(json.dumps(R["grouping"], indent=1))

# how bad is the leakage? probability a test row has a twin in train, 80:20
from sklearn.model_selection import train_test_split
tr_i, te_i = train_test_split(np.arange(len(df)), test_size=.2, random_state=RS, stratify=y)
twin = np.isin(groups[te_i], np.unique(groups[tr_i])).mean()
R["leakage"] = {"test_rows_with_a_twin_in_train": round(float(twin) * 100, 2)}
print("test rows whose exact duplicate also appears in training: "
      f"{twin*100:.2f} %")


def base_models(seed=RS):
    return [
        ("LR", Pipeline([("sc", StandardScaler()),
                         ("m", LogisticRegression(max_iter=5000, random_state=seed))])),
        ("DT", DecisionTreeClassifier(random_state=seed)),
        ("RF", RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)),
        ("XGB", XGBClassifier(n_estimators=100, learning_rate=0.3, max_depth=6,
                              eval_metric="logloss", random_state=seed, n_jobs=-1,
                              verbosity=0)),
        ("NB", Pipeline([("sc", StandardScaler()), ("m", GaussianNB())])),
        ("KNN", Pipeline([("sc", StandardScaler()),
                          ("m", KNeighborsClassifier(n_neighbors=5))])),
    ]


def met(yt, yp, pr):
    return dict(Accuracy=accuracy_score(yt, yp), Precision=precision_score(yt, yp, zero_division=0),
                Recall=recall_score(yt, yp), F1=f1_score(yt, yp), AUC=roc_auc_score(yt, pr),
                Brier=brier_score_loss(yt, pr))


# ============================ DPCS ============================
def dpcs_fit_predict(Xa, ya, Xb, seed, ga=None, lam=0.5, inner=5, return_info=False):
    """Diversity pruned calibrated stacking.

    1. inner CV out of fold probabilities for every candidate base learner
    2. score each learner by AUC; score each pair by error correlation
    3. greedily keep the subset maximising mean AUC minus lam * mean error correlation
    4. calibrate survivors, fit meta LR on their calibrated OOF probabilities
    5. choose the decision threshold that maximises inner fold F1
    """
    # The inner CV must respect patient groups too. If it does not, duplicate rows
    # leak inside the training fold, every tree model scores an inner AUC near 1.0,
    # and the selection step picks the memorisers. Leakage corrupts model selection,
    # not only the final estimate.
    if ga is not None:
        splits = list(StratifiedGroupKFold(inner, shuffle=True,
                                           random_state=seed).split(Xa, ya, ga))
    else:
        splits = list(StratifiedKFold(inner, shuffle=True, random_state=seed).split(Xa, ya))
    cv = splits
    cands = base_models(seed)
    oof, aucs = {}, {}
    for n, m in cands:
        p = cross_val_predict(m, Xa, ya, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
        oof[n] = p
        aucs[n] = roc_auc_score(ya, p)
    names = [n for n, _ in cands]
    err = {n: (oof[n] > .5).astype(int) != ya for n in names}
    corr = {}
    for a, b in itertools.combinations(names, 2):
        ea, eb = err[a].astype(float), err[b].astype(float)
        c = np.corrcoef(ea, eb)[0, 1] if ea.std() > 0 and eb.std() > 0 else 0.0
        corr[tuple(sorted((a, b)))] = 0.0 if np.isnan(c) else c

    def score(sub):
        if len(sub) < 2: return -np.inf
        a = np.mean([aucs[n] for n in sub])
        pc = [corr[tuple(sorted(p))] for p in itertools.combinations(sub, 2)]
        return a - lam * float(np.mean(pc))

    best, bestv = None, -np.inf
    for k in range(2, len(names) + 1):
        for sub in itertools.combinations(names, k):
            v = score(list(sub))
            if v > bestv: bestv, best = v, list(sub)

    sel = dict(cands)
    def inner_cv(Xs, ys, gs):
        if gs is None:
            return list(StratifiedKFold(3, shuffle=True, random_state=seed).split(Xs, ys))
        return list(StratifiedGroupKFold(3, shuffle=True, random_state=seed).split(Xs, ys, gs))

    # calibrated out of fold probabilities, computed by hand so that the inner
    # calibration folds are built on the sub training set rather than on absolute
    # indices of Xa, and so that they also respect the patient groups
    def oof_calibrated(model, Xs, ys, gs, outer):
        p = np.zeros(len(ys))
        for tr_i, te_i in outer:
            g_sub = None if gs is None else gs[tr_i]
            c = CalibratedClassifierCV(clone(model), method="sigmoid",
                                       cv=inner_cv(Xs[tr_i], ys[tr_i], g_sub))
            c.fit(Xs[tr_i], ys[tr_i])
            p[te_i] = c.predict_proba(Xs[te_i])[:, 1]
        return p

    cal = {n: CalibratedClassifierCV(clone(sel[n]), method="sigmoid",
                                     cv=inner_cv(Xa, ya, ga)).fit(Xa, ya) for n in best}
    Za = np.column_stack([oof_calibrated(sel[n], Xa, ya, ga, splits) for n in best])
    meta = LogisticRegression(max_iter=5000, C=1.0, random_state=seed).fit(Za, ya)
    pa = meta.predict_proba(Za)[:, 1]
    ths = np.linspace(.2, .8, 61)
    thr = float(ths[np.argmax([f1_score(ya, (pa >= t).astype(int)) for t in ths])])
    Zb = np.column_stack([cal[n].predict_proba(Xb)[:, 1] for n in best])
    pb = meta.predict_proba(Zb)[:, 1]
    if return_info:
        return pb, thr, {"selected": best, "aucs": {k: float(v) for k, v in aucs.items()},
                         "err_corr": {f"{a}|{b}": float(v) for (a, b), v in corr.items()},
                         "threshold": thr}
    return pb, thr


# ==================== protocol comparison ====================
def evaluate(protocol, n_repeats=5):
    """protocol: 'random' ignores groups, 'grouped' keeps patients together."""
    out = {n: {k: [] for k in ["Accuracy", "Precision", "Recall", "F1", "AUC", "Brier"]}
           for n in [n for n, _ in base_models()] + ["Stacking", "DPCS"]}
    for rep in range(n_repeats):
        if protocol == "grouped":
            splitter = StratifiedGroupKFold(5, shuffle=True, random_state=rep)
            it = splitter.split(X, y, groups)
        else:
            splitter = StratifiedKFold(5, shuffle=True, random_state=rep)
            it = splitter.split(X, y)
        for tr, te in it:
            Xa, Xb, ya, yb = X[tr], X[te], y[tr], y[te]
            for n, m in base_models():
                m.fit(Xa, ya)
                mm = met(yb, m.predict(Xb), m.predict_proba(Xb)[:, 1])
                for k in mm: out[n][k].append(mm[k])
            st = StackingClassifier(estimators=base_models(),
                                    final_estimator=LogisticRegression(max_iter=5000, random_state=RS),
                                    cv=5, stack_method="predict_proba", n_jobs=-1).fit(Xa, ya)
            mm = met(yb, st.predict(Xb), st.predict_proba(Xb)[:, 1])
            for k in mm: out["Stacking"][k].append(mm[k])
            ga = groups[tr] if protocol == 'grouped' else None
            pb, thr = dpcs_fit_predict(Xa, ya, Xb, seed=RS, ga=ga)
            mm = met(yb, (pb >= thr).astype(int), pb)
            for k in mm: out["DPCS"][k].append(mm[k])
        print(f"  {protocol} repeat {rep} done", flush=True)
    return {n: {k: {"mean": float(np.mean(v)), "sd": float(np.std(v))} for k, v in d.items()}
            for n, d in out.items()}, out


print("\n=== protocol A: standard stratified CV, groups ignored (the paper's style) ===")
R["random_cv"], raw_rand = evaluate("random")
for n, d in R["random_cv"].items():
    print(f"{n:9s} acc {d['Accuracy']['mean']:.4f}+-{d['Accuracy']['sd']:.4f}  "
          f"F1 {d['F1']['mean']:.4f}  AUC {d['AUC']['mean']:.4f}")

print("\n=== protocol B: leakage aware StratifiedGroupKFold ===")
R["grouped_cv"], raw_grp = evaluate("grouped")
for n, d in R["grouped_cv"].items():
    print(f"{n:9s} acc {d['Accuracy']['mean']:.4f}+-{d['Accuracy']['sd']:.4f}  "
          f"F1 {d['F1']['mean']:.4f}  AUC {d['AUC']['mean']:.4f}")

R["raw_random"] = {n: {k: list(map(float, v)) for k, v in d.items()} for n, d in raw_rand.items()}
R["raw_grouped"] = {n: {k: list(map(float, v)) for k, v in d.items()} for n, d in raw_grp.items()}

# one DPCS fit on a grouped split, to expose what it selected
sg = StratifiedGroupKFold(5, shuffle=True, random_state=0)
tr, te = next(iter(sg.split(X, y, groups)))
_, _, info = dpcs_fit_predict(X[tr], y[tr], X[te], seed=RS, ga=groups[tr], return_info=True)
R["dpcs_example"] = info
print("\nDPCS selected base learners:", info["selected"], " threshold", round(info["threshold"], 3))

json.dump(R, open("results_part2.json", "w"), indent=1)
print("\nsaved results_part2.json")
