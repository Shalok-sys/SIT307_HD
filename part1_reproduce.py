"""Part 1: reproduction of Bhagat, Sharma and Agarwal (2024).

Protocol recovered from the paper's reported accuracies, all of which are exact
integer fractions of 205 (190/205 = 92.68 %, 186/205 = 90.73 %, 202/205 = 98.53 %).
205 is 20 % of 1025, so the dataset is the 1025 row Kaggle heart disease file and
the split is a random 80:20 hold out.
"""
import warnings, json, time
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix)

RS = 42
DATA = "data/heart1025.csv"
df = pd.read_csv(DATA)
X = df.drop(columns="target").values
y = df["target"].values
FEATURES = list(df.columns[:-1])

R = {}
R["dataset"] = {
    "rows": int(len(df)), "features": int(X.shape[1]),
    "target_counts": {int(k): int(v) for k, v in df.target.value_counts().items()},
    "exact_duplicate_rows": int(df.duplicated().sum()),
    "unique_rows": int(len(df.drop_duplicates())),
    "duplicate_share": round(100 * df.duplicated().sum() / len(df), 2),
}
print(json.dumps(R["dataset"], indent=1))


def base_models(seed=RS):
    """The six classifiers named in the paper.

    Scaling is wrapped in a pipeline for the three distance or gradient based
    models. Tree ensembles are scale invariant so they are left unscaled.
    """
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


def metrics(y_true, y_pred, y_prob):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_prob),
    }


# ---------------------------------------------------------------- headline run
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.20, random_state=RS, stratify=y)
R["split"] = {"train": len(Xtr), "test": len(Xte),
              "test_positive": int(ytr.sum()), "note": "205 test rows, matching the paper"}

rows = {}
for name, mdl in base_models():
    t0 = time.time()
    mdl.fit(Xtr, ytr)
    p = mdl.predict(Xte)
    pr = mdl.predict_proba(Xte)[:, 1]
    rows[name] = metrics(yte, p, pr)
    rows[name]["seconds"] = round(time.time() - t0, 3)
    rows[name]["correct_of_205"] = int((p == yte).sum())

stack = StackingClassifier(
    estimators=base_models(),
    final_estimator=LogisticRegression(max_iter=5000, random_state=RS),
    cv=5, stack_method="predict_proba", n_jobs=-1)
t0 = time.time()
stack.fit(Xtr, ytr)
p = stack.predict(Xte); pr = stack.predict_proba(Xte)[:, 1]
rows["Stacking"] = metrics(yte, p, pr)
rows["Stacking"]["seconds"] = round(time.time() - t0, 3)
rows["Stacking"]["correct_of_205"] = int((p == yte).sum())
R["reproduction_seed42"] = rows

paper = {"LR": None, "DT": 0.9268, "RF": 0.9268, "XGB": 0.9073,
         "NB": None, "KNN": None, "Stacking": 0.9853}
R["paper_reported"] = paper
print("\n--- Part 1 reproduction, single 80:20 split (seed 42) ---")
for k, v in rows.items():
    pv = paper.get(k)
    d = f"  paper {pv:.4f}  diff {v['Accuracy']-pv:+.4f}" if pv else ""
    print(f"{k:9s} acc {v['Accuracy']:.4f} ({v['correct_of_205']}/205)  "
          f"P {v['Precision']:.4f} R {v['Recall']:.4f} F1 {v['F1']:.4f} AUC {v['AUC']:.4f}{d}")

# ------------------------------------------------- sensitivity to the split seed
seeds = list(range(30))
sens = {n: {m: [] for m in ["Accuracy", "Precision", "Recall", "F1", "AUC"]}
        for n, _ in base_models() + [("Stacking", None)]}
for s in seeds:
    a, b, c, d_ = train_test_split(X, y, test_size=.2, random_state=s, stratify=y)
    for name, mdl in base_models(seed=RS):
        mdl.fit(a, c)
        mm = metrics(d_, mdl.predict(b), mdl.predict_proba(b)[:, 1])
        for k in mm: sens[name][k].append(mm[k])
    st = StackingClassifier(estimators=base_models(seed=RS),
                            final_estimator=LogisticRegression(max_iter=5000, random_state=RS),
                            cv=5, stack_method="predict_proba", n_jobs=-1).fit(a, c)
    mm = metrics(d_, st.predict(b), st.predict_proba(b)[:, 1])
    for k in mm: sens["Stacking"][k].append(mm[k])
    print(f"  seed {s:2d} done", flush=True)

R["seed_sensitivity"] = {n: {k: {"mean": float(np.mean(v)), "sd": float(np.std(v)),
                                 "min": float(np.min(v)), "max": float(np.max(v))}
                             for k, v in d_.items()} for n, d_ in sens.items()}
R["seed_sensitivity_raw"] = {n: {k: list(map(float, v)) for k, v in d_.items()}
                             for n, d_ in sens.items()}
print("\n--- Across 30 random 80:20 splits ---")
for n in sens:
    a = R["seed_sensitivity"][n]["Accuracy"]
    print(f"{n:9s} acc {a['mean']:.4f} +- {a['sd']:.4f}   range {a['min']:.4f} to {a['max']:.4f}")

json.dump(R, open("results_part1.json", "w"), indent=1)
print("\nsaved results_part1.json")
