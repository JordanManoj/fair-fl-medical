# Results log

Rules: ≥5 seeds per configuration, report mean ± 95% CI (t-interval), never a
single run. Preprocessing statistics are fit on training data only (FLamby
normalizes each center with that center's train mean/std; pooled uses pooled
train stats). Seeds vary model init + batch order; the train/test split is
FLamby's fixed split (`random_state=43`).

Metrics on sex, all gaps are **female − male**: SPD = P(ŷ=1|F) − P(ŷ=1|M),
EOD = TPR_F − TPR_M. `n/a` = undefined (group or class missing from that test set).
Headline "all-centers" = union of the 4 hospital test sets, each normalized with
its own hospital's train stats. Per-seed rows live in `results/*.csv`.

## 2026-10-04 · FedAvg baseline (`scripts/run_fedavg.py`, seeds 42–46)

FLamby hyperparameters: SGD, lr 1e-3, batch 4, 100 local updates × 15 rounds,
logistic regression.

**Reproduction check.** On FLamby's pooled test, accuracy is **0.748 ± 0.027 (std)**,
which matches the published 0.748 ± 0.027. 24 of the 25 per-seed/per-site
accuracies equal FLamby's published CSVs exactly. The exception is seed 45 on
Cleveland: 0.721 vs 0.731, a difference of 1 patient out of 104, most likely a
prediction at the 0.5 threshold that rounds differently under torch 2.14.

| test set | n | n female | acc | balanced acc | SPD (F−M) | EOD (F−M) |
|---|---|---|---|---|---|---|
| all-centers | 254 | 56 | 0.705 ± 0.014 | 0.705 ± 0.014 | −0.216 ± 0.186 | −0.127 ± 0.257 |
| Cleveland | 104 | 34 | 0.729 ± 0.027 | 0.729 ± 0.028 | −0.161 ± 0.202 | −0.017 ± 0.257 |
| Hungary | 89 | 19 | 0.719 ± 0.041 | 0.736 ± 0.038 | −0.279 ± 0.173 | −0.398 ± 0.257 |
| Switzerland | 16 | 0 | 0.650 ± 0.141 | 0.813 ± 0.075* | n/a | n/a |
| Long Beach | 45 | 3 | 0.640 ± 0.063 | 0.690 ± 0.070 | −0.338 ± 0.380 | −0.318 ± 0.538 |
| FLamby pooled test | 254 | 56 | 0.748 ± 0.033 | 0.748 ± 0.033 | −0.271 ± 0.169 | −0.170 ± 0.238 |

\* Switzerland's test set has only 1 negative, so its balanced accuracy rests on a single patient.

**Observations**
1. **FLamby's "Pooled Test" overstates FL accuracy.** It normalizes the test
   data with pooled-train statistics, which no FL client has. Under the
   per-hospital normalization the model was trained with, the same models
   score 0.705, not 0.748. We use all-centers as the headline number.
2. **FedAvg under-predicts disease for women.** SPD is −0.22 and its CI
   excludes 0. Part of this gap reflects women's lower disease prevalence in
   the data. EOD, which conditions on true disease, is −0.13 but its CI
   includes 0, so 5 seeds cannot yet show a TPR gap.
3. **Fairness varies a lot across seeds.** SPD ranges from −0.03 to −0.36.
   FedAvg at lr 1e-3 for 15 rounds is probably not converged, so the next
   step is 10+ seeds and a convergence check before comparing methods.

## 2026-10-04 · Federated normalization (`run_fedavg.py --norm ...`, seeds 42–51)

Question: is FedAvg's gap to pooled training (0.705 vs 0.795) partly caused by
each hospital z-scoring with its own statistics? Three modes, all fit on train only:
- **local**: FLamby default, each hospital uses its own mean and std.
- **federated**: global mean and std computed by the server from per-hospital
  (count, Σx, Σx²). These equal the pooled-train statistics to float precision
  (max |Δstd| = 6e-5). No patient rows leave a hospital.
- **federated-impute**: like federated, but zeros in chol and trestbps (UCI's
  missing-value code) are first replaced by the global mean of the observed
  values. This is the control for a shortcut: chol is 0 for 100% of
  Switzerland's patients and about 25% of Long Beach's, and those patients are
  mostly positive. Under global stats, "missing chol" becomes an extreme
  z-score the model could learn to read as "disease".

All-centers test set (n = 254, 56 women), mean ± 95% CI over 10 seeds:

| norm | acc | balanced acc | SPD (F−M) | EOD (F−M) |
|---|---|---|---|---|
| local | 0.704 ± 0.010 | 0.704 ± 0.009 | −0.160 ± 0.096 | −0.049 ± 0.121 |
| federated | 0.767 ± 0.013 | 0.767 ± 0.013 | −0.266 ± 0.077 | −0.142 ± 0.115 |
| federated-impute | 0.765 ± 0.014 | 0.764 ± 0.014 | −0.242 ± 0.078 | −0.111 ± 0.102 |

Paired difference vs local (same seeds), with paired t-test:

| norm | Δacc | ΔSPD | ΔEOD |
|---|---|---|---|
| federated | +0.063 ± 0.019 (p<1e-4, 10/10 seeds up) | −0.106 ± 0.029 (p<1e-4, 10/10 worse) | −0.094 ± 0.043 (p=0.0008) |
| federated-impute | +0.060 ± 0.019 (p=1e-4, 10/10 seeds up) | −0.082 ± 0.031 (p=2e-4, 10/10 worse) | −0.063 ± 0.053 (p=0.025) |

Per hospital (federated-impute vs local): accuracy is Cleveland 0.737 vs 0.713,
Hungary 0.779 vs 0.733, Switzerland 0.850 vs 0.688, Long Beach 0.771 vs 0.633.
See `results/fedavg_heart_<norm>.csv`.

**Findings**
1. **Federated normalization gives +6 accuracy points in every one of 10 seeds**
   and costs only one extra round of summary statistics. 0.765 closes most of
   the gap to pooled training (0.795) and is close to FLamby's FedYogi (0.779).
2. **The missing-cholesterol shortcut explains almost none of the gain.**
   Imputation keeps +0.060 of the +0.063.
3. **The accuracy gain makes the model less fair.** SPD gets worse in 10/10
   seeds and EOD worsens significantly. A preprocessing choice alone moves the
   model along the accuracy–fairness trade-off, which is the setting
   Fed-FUEL and FairTrade target.
4. **Part of the per-site gain is majority-class prediction.** At Long Beach
   accuracy rises 0.633 → 0.771 while balanced accuracy *falls* 0.671 → 0.635
   (78% of its patients are positive). Overall balanced accuracy still rises by
   6 points, so the headline gain is real.
5. **Hypothesis (not yet tested):** local z-scoring centers each hospital's
   features at zero, which erases differences between patient populations
   (e.g. Long Beach's older, sicker cohort) that carry label signal. Global
   statistics keep those differences. That would also mean the gain partly
   comes from recognizing which site a patient is from, which may not transfer
   to a new hospital. A leave-one-hospital-out test would check this.

## 2026-10-04 · Leave-one-hospital-out (`scripts/run_loho.py`, seeds 42–51)

Tests finding 5 above. FedAvg trains on 3 hospitals and is evaluated on all
patients (train + test split) of the 4th, which contributes nothing to training
or to the federated statistics. Under `local` the held-out site z-scores with
its own (unlabeled) feature statistics.

| held out (n, women) | norm | acc | balanced acc | SPD (F−M) |
|---|---|---|---|---|
| Cleveland (303, 97) | local | 0.704 ± 0.039 | 0.706 ± 0.038 | −0.168 ± 0.109 |
| | federated-impute | 0.735 ± 0.028 | 0.731 ± 0.028 | −0.202 ± 0.084 |
| Hungary (261, 69) | local | 0.705 ± 0.035 | 0.725 ± 0.030 | −0.240 ± 0.115 |
| | federated-impute | 0.766 ± 0.025 | 0.754 ± 0.024 | −0.261 ± 0.074 |
| Switzerland (46, 3) | local | 0.557 ± 0.051 | 0.773 ± 0.026 | −0.358 ± 0.149 |
| | federated | 0.724 ± 0.150 | 0.810 ± 0.107 | −0.364 ± 0.097 |
| | federated-impute | 0.787 ± 0.045 | 0.842 ± 0.099 | −0.431 ± 0.058 |
| Long Beach (130, 5) | local | 0.631 ± 0.019 | 0.660 ± 0.021 | −0.203 ± 0.234 |
| | federated-impute | 0.745 ± 0.019 | 0.596 ± 0.043 | −0.064 ± 0.103 |

Paired federated-impute − local:

| held out | Δacc | Δbalanced acc |
|---|---|---|
| Cleveland | +0.031 ± 0.028 (p=0.03, 8/10 up) | +0.025 ± 0.023 (p=0.04, 8/10 up) |
| Hungary | +0.061 ± 0.029 (p=0.001, 10/10 up) | +0.029 ± 0.026 (p=0.03, 8/10 up) |
| Switzerland | +0.230 ± 0.049 (p<1e-4, 10/10 up) | +0.069 ± 0.118 (p=0.22, n.s.) |
| Long Beach | +0.114 ± 0.019 (p<1e-4, 10/10 up) | **−0.064 ± 0.038** (p=0.004, 1/10 up) |

**Findings**
1. **The hospital-recognition hypothesis is rejected as the main explanation.**
   At unseen hospitals with balanced disease rates (Cleveland, Hungary),
   federated normalization still improves balanced accuracy by about 3 points.
   That is about half the in-distribution gain (+6.0), so part of the original
   gain does not transfer.
2. **At high-prevalence sites the accuracy gain is a prevalence effect, not
   better discrimination.** At Long Beach (78% positive), accuracy rises by
   11 points while balanced accuracy *falls* by 6. At Switzerland (93%
   positive), balanced accuracy does not change significantly. Global
   statistics keep the population's shift toward older, sicker patients, so
   the model predicts "disease" more often there.
3. **Imputation matters on unseen sites.** Without it, held-out Switzerland
   (cholesterol 100% zero-coded) is unstable: 0.724 ± 0.150 vs 0.787 ± 0.045
   with imputation. `federated-impute` is the default from now on.
4. **Consequence for the rest of the project:** balanced accuracy is reported
   alongside accuracy everywhere, and claims about accuracy gains rest on
   balanced accuracy.

## 2026-10-04 · Local and pooled baselines (`scripts/run_baselines.py`, seeds 42–51)

FLamby settings: Adam lr 1e-3, batch 4, 50 epochs, one shared init per seed.
**Reproduction:** all 100 local-model accuracies (4 models × 5 test sets × seeds
42–46) and the pooled model's Pooled Test accuracies match FLamby's published
CSVs exactly. The pooled model's per-hospital numbers differ by design. FLamby
evaluates it on each hospital's test set with that hospital's own statistics,
not the pooled statistics it was trained with; e.g. at Long Beach FLamby
reports 0.644, while with consistent normalization it is 0.822.

All-centers test set (n = 254, 56 women), 10 seeds:

| method | acc | balanced acc | SPD (F−M) | EOD (F−M) |
|---|---|---|---|---|
| Local Cleveland | 0.735 ± 0.006 | 0.736 ± 0.006 | −0.199 ± 0.019 | −0.119 ± 0.043 |
| Local Hungary | 0.691 ± 0.006 | 0.695 ± 0.006 | −0.227 ± 0.015 | −0.107 ± 0.011 |
| Local Switzerland | 0.497 ± 0.043 | 0.488 ± 0.042 | −0.036 ± 0.146 | −0.054 ± 0.159 |
| Local Long Beach | 0.566 ± 0.019 | 0.555 ± 0.020 | −0.136 ± 0.071 | +0.005 ± 0.068 |
| FedAvg (local norm) | 0.704 ± 0.010 | 0.704 ± 0.009 | −0.160 ± 0.096 | −0.049 ± 0.121 |
| FedAvg (federated-impute) | 0.765 ± 0.014 | 0.764 ± 0.014 | −0.242 ± 0.078 | −0.111 ± 0.102 |
| Pooled (upper bound) | 0.796 ± 0.002 | 0.795 ± 0.002 | −0.322 ± 0.003 | −0.156 ± 0.003 |

**Findings**
1. **The more accurate the model, the less fair it is.** From FedAvg-local to
   FedAvg-federated to Pooled, accuracy rises (0.704 → 0.765 → 0.796) and SPD
   worsens (−0.16 → −0.24 → −0.32). Better use of the data amplifies the sex
   gap already in the labels, which is exactly why a fairness-aware method is
   needed.
2. **FedAvg with FLamby's default normalization loses to Cleveland alone**
   (0.704 vs 0.735). With federated normalization it beats every single
   hospital.
3. **Local models from small hospitals are useless elsewhere.** Switzerland's
   model is at chance (0.497), since its training labels are all positive.

## 2026-10-04 · FairTrade vs FedAvg (`scripts/run_fairtrade.py`, seeds 42–51)

Port of FairTrade (`fairfl/fairtrade.py`). Every client minimizes
BCE + α · demographic-parity penalty, with full-batch local Adam, 30 rounds ×
10 local steps, and size-weighted FedAvg. MOBO over (α, lr) uses 8 Sobol +
16 qLogNEHVI evaluations to maximize validation balanced accuracy and
−|SPD|. Per seed: a fresh 25% validation split of each hospital's train data,
with federated-impute statistics fit on the remaining 75%. FedAvg is the same
trainer with α = 0 and lr tuned on validation. **All selection uses validation
metrics; test metrics are only reported.** `tests/test_fairtrade.py` checks
that the ported penalty equals the original `DemographicParityLoss`
numerically.

Defects in the challenge implementation that the port fixes:
1. MOBO objectives were computed on the **test set**.
2. The hypervolume reference point [0.001, 0.001] lay above every attainable
   −|SPD| ≤ 0, so every hypervolume improvement was 0 and the "BO" was
   effectively random search.
3. Each candidate evaluation also ran an FL round on the shared global
   model, so the final model mixed hyperparameters across evaluations.
4. Double sigmoid: the model ends in `Sigmoid` but feeds `BCEWithLogitsLoss`,
   and the penalty applies sigmoid again.
5. Unweighted client averaging.
6. `AverageTreatmentEffectLoss` calls `super(EqualOpportunityLoss, ...)`, a class
   that doesn't exist, so `--fairness_notion ate` crashes.

All-centers test set (n = 254, 56 women), 10 seeds, mean ± 95% CI:

| method | balanced acc | SPD (F−M) | EOD (F−M) | test hypervolume |
|---|---|---|---|---|
| FedAvg (same trainer, α = 0) | 0.787 ± 0.016 | −0.342 ± 0.057 | −0.191 ± 0.085 | 0.047 ± 0.013 |
| FairTrade, most accurate with val \|SPD\| ≤ 0.10 | 0.718 ± 0.035 | +0.022 ± 0.029 | **+0.138 ± 0.024** | 0.133 ± 0.004 |
| FairTrade, most accurate with val \|SPD\| ≤ 0.05 | 0.700 ± 0.060 | +0.014 ± 0.038 | +0.089 ± 0.055 | |

Paired FairTrade(τ = 0.10) − FedAvg: Δ balanced acc −0.069 ± 0.041, Δ|SPD|
−0.307 ± 0.061. The constraint was met on validation in 10/10 seeds
(7/10 for τ = 0.05). Hypervolume is the area of the (balanced acc, −|SPD|)
plane dominated by the validation-selected Pareto front re-scored on test,
with reference point (0.5, −0.5).

**Findings**
1. **FairTrade works on real hospitals.** It removes the sex gap in prediction
   rates (SPD −0.34 → +0.02, CI includes 0) at a cost of 7 balanced-accuracy
   points, and its Pareto front nearly triples the hypervolume
   (0.047 → 0.133, 10/10 seeds).
2. **Demographic parity is the wrong fairness target for this diagnosis.**
   Women's disease rate in the data is 24.7%, men's is 60.1%. Equalizing
   prediction rates forces over-diagnosis of women, so sick women are
   detected *more* often than sick men: EOD +0.14, CI excludes 0. Part of
   the 7-point accuracy cost is the price of matching rates that really
   differ. The clinically meaningful notion is equal opportunity (equal
   detection rate for sick patients), the notion the original code tried to
   offer but crashed on.
3. **Selection on a small validation set is noisy.** It has about 123
   patients and about 23 women per seed; candidate-level val→test
   correlation is 0.89 for balanced accuracy but only 0.69 for SPD. 2/10
   seeds picked α at the upper bound with poor accuracy (0.62–0.69).
4. **FedAvg through this trainer beats FLamby's FedAvg** (0.787 vs 0.764
   balanced accuracy under the same normalization). Full-batch Adam with a
   tuned learning rate optimizes better than SGD at lr 1e-3 for 15 rounds.
   Comparisons below always use the same trainer for both methods.

**Next (milestone 8):** an equal-opportunity objective and penalty, and
fairness statistics pooled across hospitals. After the validation split,
Long Beach has 1 woman and Switzerland 2 in their fit data, so their local
penalties carry almost no information.

## Raw runs

Per-seed rows: `results/fedavg_heart_<norm>.csv`, `results/loho_heart.csv`,
`results/baselines_heart.csv`. The 5-seed reproduction above is seeds 42–46
of `fedavg_heart_local.csv`.
