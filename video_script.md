# 5-minute video presentation script

Show face + screen share. Timings are cumulative.

## 0:00–0:35 — The claim
"The paper I reproduced is Bhagat, Sharma and Agarwal, 2024. They stack six
classifiers for heart attack prediction and report 98.53% accuracy, against
92.68% for their best single model. My question was simple: does that hold up?"

## 0:35–1:20 — Recovering the protocol (screen: README section)
"The paper never states its train/test split. But every accuracy it reports is
an exact fraction of 205 — 190/205, 186/205, 202/205. 205 is 20% of 1025, so
it's a random 80:20 hold-out. That let me reimplement it faithfully."

## 1:20–2:10 — The reproduction (screen: run part1_reproduce.py, Table 2)
"I didn't just match the paper — I beat it. Random forest, XGBoost and stacking
all hit 100.00%. Over 30 random seeds they average 99.48%. The paper's 92.68%
for random forest sits seven standard deviations below that."

## 2:10–3:00 — The diagnosis (screen: fig1_dataset.png)
"So I looked at the data. 1,025 rows — but only 302 distinct patients. 723 exact
duplicates. Under a random 80:20 split, 98.54% of test rows have an identical
twin in training. The models aren't generalising, they're recalling."

## 3:00–3:50 — The correction (screen: fig3_protocol.png)
"My fix: treat each distinct feature vector as a patient group and use stratified
group k-fold. Watch what happens. Decision tree drops 24 points. Stacking drops
20. But naive Bayes — which has 27 parameters and can't memorise anything —
drops 1.4. That ordering is only explicable as leakage."

## 3:50–4:35 — My method (screen: fig4_dpcs.png + Table 5)
"DPCS prunes base learners on accuracy and error diversity, calibrates them, and
tunes the threshold — all under a group-aware inner resampler. That last part
mattered: my first version leaked inside the training fold, picked the
memorisers, and gained almost nothing. Fixed, it beats the paper's stacking on
all six metrics — accuracy 0.8075 vs 0.7915, recall up 2.6 points, Brier down
21%."

## 4:35–5:00 — The honest conclusion
"But I have to report this: plain logistic regression still beats my ensemble,
at 0.8162. On 302 effective patients, ensemble machinery isn't justified. The
paper's contribution doesn't survive an honest evaluation — and that, not a
higher score, is the finding."
