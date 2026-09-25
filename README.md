# MEDAI — Story of a Clinical Data Project

### From 60 models with brilliant metrics to a system that knows how to say "I don't know"

*A data-analysis story. No code included. Every figure in this document is measured against the project's files and reproducible with the commands in section 9.*

---

## Why it matters

Specialist waitlists are long, and every day spent waiting for a diagnosis is a day where a treatable condition can quietly become an emergency. A single analysis often can't rule an illness in or out on its own — so a physician orders test after test, in sequence, before reaching a confident answer. That sequence is where time, and sometimes patients, get lost.

MEDAI was built to shorten that gap. Instead of one model producing one verdict, each of the **12 conditions** in the catalog is checked by **5 predictive models running in parallel** — more than 60 models in total — covering angles that a single test can miss and cross-checking each other before a probability ever reaches a screen. The goal: put a fuller, evidence-backed picture in front of the physician in one sitting, with the underlying data and limitations attached, so what used to take a string of appointments and lab orders can be reviewed in minutes rather than weeks.

That is the *design goal* the architecture serves — it is not yet a measured clinical outcome. Nothing in this document claims MEDAI has reduced waitlists or saved lives; that claim would need a prospective external cohort MEDAI doesn't have yet. What follows is an honest account of what is actually proven today, what is still a hypothesis, and the four-layer architecture built specifically so that distinction never gets blurred by an impressive-looking score.

---

## What MEDAI is

MEDAI started as an oncology machine-learning MVP. Instead of replacing those models, I built a system around them — a layer designed to make their outputs traceable, reproducible, and explicit about their limitations.

The architecture keeps four things strictly separate:

- **Model output** — what the trained artifact produces
- **Dataset evidence** — what the underlying data can actually support
- **Validation evidence** — how robust the result is under different tests
- **Clinical interpretation** — what can, and cannot, be claimed medically

That separation is fundamental. A model can produce a prediction. A dataset can contain a strong signal. A validation procedure can reveal how stable that signal is. None of those, by themselves, establish clinical validity — so MEDAI never collapses them into a single score.

**What's built into the system:**

- 🩺 8 clinical flows, each running multiple machine-learning models in parallel
- 🔬 Model and dataset auditing: repeated data partitions, duplication checks, label-permutation controls, and single-variable baselines
- 📊 Validation evidence tracked separately from model output — measured values are shown only when the corresponding test actually ran
- 🗂️ Model provenance and artifact status, distinguishing technical verification from clinical validation
- ⚖️ Decision thresholds, model disagreement, missing values, and imputation are visible rather than hidden behind a final prediction
- 🌐 English, French, and Spanish supported across both the interface and the clinical analyses, from a single source of truth
- 🔬 A rehearsal mode so the full clinical workflow can be tested before real clinical data and verified feature contracts are in place

And one rule runs through the entire platform: **if the evidence doesn't exist, MEDAI does not manufacture it.** No invented external-validation score. No hidden imputation. No unexplained weighting. No presenting technical model performance as clinical performance.

MEDAI is decision support, not a diagnostic system. The objective is not to make an AI model look impressive — it's to make the entire path from data → model → evidence → interpretation inspectable.

This is the first version. The next stage is external validation, verified feature contracts, and real data for each condition.

What follows is that audit, told with its numbers.

---

## 0. Reading rule

Before any number, three rules govern everything that follows:

**A figure that no trained artifact produced is not shown.** If the data is missing, the gap is declared — it is never filled with noise, an average, or a constant. *(SPEC §19)*

> "200 random splits demonstrate stability against the split, but they do not demonstrate clinical validity or the absence of every kind of leakage."

---

## 1. The why: a clinical question, not a machine-learning question

Clinical reasoning is a sequence: symptoms first, then data and test results, and with that, the disease is confirmed — or ruled out. A physician doesn't guess from one isolated test; they use it to confirm a suspicion that already came from the consultation.

Two requirements follow from that, and a model can either meet them or not:

1. **The model confirms, it doesn't diagnose.** Its place is at the end of the sequence, supporting a decision that still belongs to the clinician.
2. **Confirmation requires confirmatory data.** If a model is trained on answers to a questionnaire, the most it can learn is to reproduce the questionnaire — not to confirm a disease.

This is the story of how a set of models that looked extraordinarily good turned out not to be doing that, and what it took to find out, measure it, and fix it.

---

## 2. The problem: the starting point

The project started with a large, seemingly valuable inheritance:

| What existed | Number |
|---|---|
| Model artifacts (.pkl) in the catalog | 60 |
| Diseases covered by the catalog | 12 |
| Generation scripts audited one by one | 79 |
| Artifacts with a provable clinical contract | **0** |

Zero out of sixty. And the problem wasn't technical — all 60 artifacts loaded and ran without errors. The problem was that none of them could prove where their data came from or what label they were actually predicting. A model that reports AUC 0.98 and can't show the dataset that produced it isn't clinical evidence: it's an orphaned number.

Three checks made the gap visible:

- **Clinical contract:** no artifact declared a variable schema, units, ranges, encoding, or required tests. A `schema_confirmed` flag that had been set without any measured basis was also found and removed.
- **Origin script:** 60 of 60 artifacts had an attributed synthetic origin and 0 had their own contract; none declared their variable names. The one script that did open a real dataset — the prostate one — fabricated its label with `load_breast_cancer()`, so it didn't prove anything either. The rule I set: "No PKL enters the clinical catalog without an origin script that opens a real dataset; and if the script fabricates the data, the artifact doesn't prove anything even if its name says wdbc" *(SPEC §16)*.
- **Served figures:** the app computed the "ensemble" in the browser. Only the logistic regression came from measured coefficients; the other four probabilities were generated as noise around the logistic value — the number the physician saw hadn't been produced by any trained model. It was replaced with a real query to the inference engine, and if the engine doesn't respond, the app doesn't estimate: it shows only the logistic regression, labels it as such, and hides the model table and the consensus badge. That's where the permanent rule §19 came from.

---

## 3. How I found it

### 3.1 The instrument: four layers that never mix

The first job wasn't about models — it was about the architecture of the evidence. I split what gets said into four layers and forbade any one from leaning on the one next to it:

```
Model Output          "what probability comes out"     (the artifact)
      ↓
Dataset Evidence      "what corpus it comes from"       (provenance, label, nulls, license)
      ↓
Validation Evidence   "how much that figure can hold"   (splits, permutation, leakage, calibration)
      ↓
Clinical Interpretation "what can, and can't, be claimed with it"
```

Without that separation, a figure from a synthetic corpus ends up presented as clinical performance in three clicks. With it, every number carries its origin along with it.

### 3.2 The question that ordered everything: what data was each flow trained on?

I sorted the flows by the nature of their data, and the pattern showed up immediately:

| Flow | Corpus | Data type | Confirmatory? | Measured holdout AUC |
|---|---|---|---|---|
| Breast | WDBC (FNA) | fine-needle cytology | Yes | 0.9993 |
| Renal | UCI kidney | lab work (blood/urine) | Yes | 1.0000 |
| Diabetes (Pima) | Pima Indians | OGTT + lab work | Yes | 0.8259 |
| Cardiac | Cleveland | clinical + angiography | Yes | 0.9481 |
| Liver (ILPD) | UCI 225 ILPD | lab work (bilirubin, enzymes) | Yes | 0.8388 |
| Early diabetes | questionnaire | self-reported symptoms | No | 0.9988 |
| Lung | survey | 15 self-reported answers | No | 0.9421 |
| Liver (legacy) | UCI 423 HCC Survival | survival outcome | No *(it's prognostic)* | 0.7615 |
| Cervical | behavior + biopsy | mixed, no signal | No | 0.6155 |

The finding: the flows that fall apart are, one by one, the ones that don't use confirmatory data. And the two cases with near-perfect AUC and non-confirmatory data (early diabetes 0.9988, lung 0.9421) show that AUC isn't a reliable criterion by itself — it measures how well the model ranks, not whether what it ranks makes clinical sense.

### 3.3 The extremes: why 1.0000 and why 0.6155

An AUC of 1.0000 and an AUC of 0.6 both had to be explained, not assumed or hidden. I measured them flow by flow with 200 random splits, label permutation, duplicate grouping, and a single-variable ceiling check:

| Flow | AUC | Measured verdict | What backs it up |
|---|---|---|---|
| Renal | 1.0000 | corpus nearly separable by construction | 200 splits 0.9979 · 26.5% land exactly at 1.0 · the `sc` variable alone already gives 0.937 · 0 duplicates |
| Breast | 0.9993 | nearly separable corpus | 200 splits 0.9949 · `worst perimeter` alone gives 0.9874 · 0 duplicates |
| Early diabetes | 0.9988 | leakage from repeated rows + circular label | 72 of 104 test rows have an exact twin · deduplicating drops it to 0.9735 → 0.9514 · Polyuria=1 & Polydipsia=1 → 193 of 193 positive |
| Cardiac | 0.9481 | real, stable signal | 0 duplicates · 0% of splits reach 1.0 · grouped mean 0.9000 · holdout sits 0.05 above its own mean → the citable figure is ~0.90 |
| Lung | 0.9421 | mild leakage + unusable operating point | 34 repeated rows, 10 train/test crossovers, 1 vector with a contradictory label · grouping moves it by −0.0019 (leakage isn't the cause) · the real damage is the threshold: specificity 0.75 |
| Diabetes (Pima) | 0.8259 | moderate real signal | — |
| Liver (ILPD) | 0.8388 | moderate real signal | full limits in §4.4 |
| Liver (legacy) | 0.7615 | inadequate label | hemoglobin alone (0.8654) beats the full model · 200 splits with a minimum of 0.4654 · predicts 1-year mortality, not diagnosis |
| Cervical | 0.6155 | corpus with no signal | permutation p=0.24 (null 0.4874) · 200 splits, mean 0.5393, minimum 0.175 · Brier skill −1.2692 · the best of its 23 variables alone gives 0.5802 |

Three conclusions that change how the whole project should be read:

1. **1.0000 isn't overfitting or leakage — it's corpus separability.** The danger isn't that the model memorized; it's that the corpus doesn't resemble the clinic, and the number won't transfer.
2. **The real inflation was in early_diabetes**, the questionnaire flow: leakage from row overlap plus a label that's derivable from two symptoms. Two forms of circularity, not one.
3. **Cervical's 0.6155 isn't a weak model — it's a corpus with no signal.** With that corpus, no reasonable algorithm is going to work; the fix belongs in the data, not the model.

### 3.4 The question AUC hides: how many healthy people does it flag?

AUC doesn't answer the question a clinician actually asks: "if I run this on 100 patients without the disease, how many will I tell they have it?" I measured it at the threshold actually served, with an exact upper bound (Clopper-Pearson, 95% one-sided):

| Flow | False positives per 100 healthy | 95% upper bound | Sick patients missed, per 100 |
|---|---|---|---|
| Renal | 0 (0 of 30) | 9.5 | 0 |
| Early diabetes | 2.5 (1 of 40) | 11.3 | 0 |
| Breast | 2.78 (2 of 72) | 8.5 | 0 |
| Cardiac | 3.03 (1 of 33) | 13.6 | 14.3 |
| Cervical | 7.45 (12 of 161) | 11.8 | 81.8 |
| Liver (ILPD) | 8.82 (3 of 34) | 21.3 | 38.5 |
| Lung | 25.0 (2 of 8) | 60.0 | 7.4 |
| Diabetes (Pima) | 29.0 (29 of 100) | 37.4 | 18.5 |
| Liver (legacy) | 50.0 (10 of 20) | 69.8 | 0 |

Two details AUC doesn't show, and a health system would notice right away: 0 false positives out of 30 healthy people isn't "0%" — the bound says it could run as high as 9.5 per 100 — and a flow can flag half of all healthy people while still keeping a decent AUC.

And the second question, the one that decides whether the tool holds up outside the holdout: at a realistic prevalence (10%), lung's positive predictive value drops to 0.29 and legacy liver's to 0.18, against the 0.96 and 0.57 they showed off in their own holdout. Base rate isn't a technicality — it's the difference between a useful tool and a machine that scares healthy people.

### 3.5 The healthy patient with normal labs

Tested against the running app, not in the abstract: a typical healthy patient passes in all 9 flows. But a healthy patient with labs in the 95th percentile (high values, still within physiological range) gets flagged as positive in 7 of the 9 flows. This is what a clinical reviewer needs to hear before they find it themselves: specificity is the weak point of the set, not AUC.

### 3.6 Auditing the auditor

The review didn't stop at the models. My own audit tool turned up three defects, and all three were fixed in the code, not in the report's wording:

1. A field called "false positives per 100 declared" wasn't measuring false positives per 100 healthy people — it was measuring the false fraction of declared positives (100·FP/(TP+FP)): two different clinical questions. It affected 8 of 9 flows.
2. The "95% bound" being published was the rule-of-three applied blindly: for cervical it gave 1.86 when the measured rate was 7.45 — a bound below the observed value, which is impossible. It was replaced with the exact Clopper-Pearson bound, which matches the rule-of-three when there are no failures (0 of 30 → 9.5 per 100) and is an actual bound when there are.
3. The validation index covered 4 of 9 flows, and in those four the cross-validation column was empty; 3 flows had no provenance file at all.

---

## 4. How I designed the solution

### 4.1 The architecture, in one sentence

An app where the clinician fills in a patient's chart and gets back, per disease, a probability with its full evidence hanging beneath it: what corpus it comes from, how it was validated, what its limits are, and what can't be claimed with it.

### 4.2 The three rules holding the system up

| Rule | What it prevents |
|---|---|
| **§16** — real origin script | an artifact with a famous dataset's name entering the catalog without opening that dataset |
| **§17** — declared ceiling, no external cohort | internal validation being read as clinical validity: no second cohort → the ceiling is declared, not filled in |
| **§19** — no unproduced figures | the interface showing a number no trained artifact generated |
| **§20** — labeled rehearsal | a contract-less artifact being read as clinical: rehearsal flows carry `schema_status: rehearsal_no_contract` and the interface declares it on every run |

### 4.3 What the app refuses to do

It doesn't diagnose, doesn't recommend treatment, doesn't do autonomous triage, and doesn't show a probability without its provenance card. When data is missing, the app says so — the gap gets declared.

### 4.4 Proof that the method changes the outcome

The best example isn't a flow that always worked — it's one that was wrong and got replaced:

| Aspect | Legacy liver | Replaced liver (ILPD) |
|---|---|---|
| Corpus | UCI 423 HCC Survival | UCI 225 ILPD (DOI 10.24432/C5D02C) |
| Subjects | 165 patients already diagnosed with HCC | 583 patients (441 M / 142 F) |
| Label | `died` — 1-year mortality | liver disease diagnosis via biochemical markers |
| Holdout AUC | 0.7615 | 0.8388 (95% CI 0.7704–0.9008) |
| Beats a single variable? | No: hemoglobin alone gets 0.8654 | Yes: the best single variable (`ast`) gets 0.7539 |
| False positives per 100 healthy | 50 | 8.82 |
| License | — | CC BY 4.0, verified against the primary record |

The new flow, measured in depth:

- **Holdout:** 117 rows (83 with disease) · AUC 0.8388 · AUPRC 0.9387 (CI 0.9105–0.9638) · Brier 0.1506 · Brier skill vs. prevalence 0.2694 · ECE 0.0876 · sensitivity 0.6145 · precision 0.9444 · confusion matrix TP 51 · FP 3 · TN 31 · FN 32.
- **Stability:** 200 splits, mean 0.747 (minimum 0.6258) · 5×3 cross-validation across three seeds at 0.7507 / 0.7687 / 0.7487 · label permutation p = 0 with a null of 0.5203.
- **The holdout is optimistic, and it's declared:** the validation report itself flags 0.8388 (holdout) against 0.7034 (out-of-fold predictions) — the citable figure is the lower one, not the higher one.
- **Limits, unvarnished:** no external cohort; 5 identical vectors cross train/test; 13 duplicate rows (2.2%); the served threshold (0.72) was chosen on the same holdout; 32 of 83 sick patients go undetected; the isotonic calibrator was measured (ΔBrier 0.0072) but isn't served with the artifact; effective EPV 13.3 with 10 variables.
- **A fairness nuance already measured:** in the holdout, the 90-case subgroup with `gender=1` scores 0.8738 and the 27-case subgroup with `gender=0` scores 0.7434 — performance isn't equal across subgroups, and that's declared as such.

---

## 5. What's been done (a verifiable inventory)

| Done | Verified status |
|---|---|
| Flows served with a real corpus and provenance card | 9 |
| Evidence-layer self-test | 42 of 42 green |
| Browser harness, liver flow | 25 of 25 checks |
| Control harness (breast) | 25 of 25 checks |
| Diseases tested in rehearsal mode, in-app, across three languages | 12 of 12 (5 models each, es/en/fr) |
| Licenses verified against the primary record | 8 of 9 (CC BY 4.0); lung remains UNVERIFIED, not backfilled |
| Cleanup of intermediate files | 89.4 MB, with a prior manifest and sha256 check |
| Extreme-value audit reports | 3 data-science-agent reports + an item-by-item verification addendum |
| Tooling defects fixed | 3 (two mislabeled fields + one false bound) |

And what was not done, stated explicitly: no flagged flow was pulled from service without a replacement, and one flow (cervical) is still served at AUC 0.6155 while its removal is decided.

---

## 6. What's NOT resolved

Unsoftened, because a reviewer will find it anyway:

1. **No external cohort exists** for any flow. Everything is internal validation on a single corpus. That's the declared ceiling *(SPEC §17)*.
2. **5 identical vectors cross train/test** in the liver flow. Removing them requires retraining; it's measured and declared, not hidden.
3. **Cervical is still served at AUC 0.6155** (p=0.24; 200 splits, minimum 0.175), and its stored value (0.5957) doesn't reproduce from its five member models: the discrepancy is in the aggregation rule, not the members, and the rule hasn't been located yet.
4. **The validation index covers 4 of 9 flows**, with an empty cross-validation column; 3 flows have no provenance file.
5. **Lung is served at a 0.30 threshold**, which flags 25 of every 100 healthy people and has a negative net clinical benefit (DCA −0.0230): warning everyone would beat using the model.
6. **Legacy liver (HCC) is still in the catalog** with a mortality label; it should be retired, not relabeled.
7. **Probability reliability is measured; clinical utility is not**, because there's no clinical outcome and no external cohort.
8. **Of 334 legacy artifacts surveyed, 318 load and 16 don't:** 12 for missing `lightgbm` in this environment and 4 for old pickles that fail to deserialize. The legacy catalog is still pending a decision.

---

## 7. In the language of a health system

Told to a clinical reviewer or a digital-health lead, this project is:

- **Problem** — teams accumulate models with impressive internal metrics and no traceability of data or label; the consequence is they can't be brought into a clinical decision, however high the number looks.
- **Requirements** — every figure must trace back to its corpus, its label, its split, and its license; and the system must declare what it can't claim.
- **Workflow** — the clinician enters symptoms and test results into a per-disease chart; the system returns a probability with its evidence hanging beneath it (provenance, validation, limits) and walks through the case with a guided assistant, in Spanish, English, and French.
- **Solution** — a four-layer evidence architecture, a per-flow provenance catalog, a reproducible audit engine, and an interface that refuses to show figures without an origin.
- **Decision Support** — the system doesn't decide: it organizes the information and surfaces the uncertainty, including the uncertainty that works against the system itself.

What it would take to move from prototype to clinical use: a prospective external cohort, ethics-committee review, model registration with a code hash and date, and a subgroup-fairness monitoring plan. None of that is claimed as done here.

---

## 8. Publication: what's public and what isn't

| Published | Not published |
|---|---|
| This story (narrative and figures) | Backend and frontend source code |
| Architecture and rules (§16/§17/§20) | Trained artifacts (.pkl) |
| Measured figures and their limits | Downloaded corpora and intermediate files |
| Provenance and licenses of the public corpora | Credentials, tokens, or keys (none were observed) |
| Interface screenshots with synthetic cases | Patient data (there is none — the corpora are public and anonymized) |

The corpora cited are public and used under their license (CC BY 4.0, with attribution to the UCI Machine Learning Repository). The code stays private: what's open is the evidence and the method, not the implementation.

---

## 9. How to verify it yourself

```bash
# Evidence-layer self-test (42 checks)
PYTHONPATH=. ./.venv/Scripts/python.exe medai/tools/selftest.py

# Metrics audit across the 9 flows (200 splits, permutation, leakage, FP bounds)
PYTHONPATH=. ./.venv/Scripts/python.exe medai/tools/audit_metricas_flujos.py
```

And the files where each figure lives:

| Figure | File |
|---|---|
| AUC, leakage, stability, false positives, and bounds for the 9 flows | `medai/reports/audit_metricas_8_flujos.json` |
| Liver-flow validation (holdout, CV, calibration, limits) | `medai/validation/reports/liver_ilpd.json` |
| Provenance, license, and units for the liver corpus | `data/liver_ilpd.provenance.json` |
| Corpus integrity | `sha256 = 69ff1a69a3f3aca42f10691cb7dca0257242f56f605a47d2e0d51febaae37bb7` |
| Extreme-value audit (own and agents') | `medai/reports/extremos_medicion_propia.md`, `agente_extremos_altos_1.md`, `agente_extremos_altos_2.md`, `agente_extremos_bajos.md` |
| System rules | `MEDAI-CLINICAL-AI-SPEC.md` (§16 artifact admission, §17 external cohort, §19 served probability, §20 rehearsal mode) |

---

## Appendix · Minimal glossary

- **AUC (AUROC):** the probability the model ranks a sick patient above a healthy one. It is not accuracy, probability, or clinical utility.
- **AUPRC:** the same idea, focused on the sick class; it only means something read against its trivial value, which is prevalence.
- **False positives per 100 healthy:** out of 100 patients without the disease, how many get flagged as sick. This is the clinical question screening actually asks.
- **Upper bound (Clopper-Pearson):** with few healthy patients, "0 failures" doesn't mean "0%"; the bound states how high the real rate could actually be.
- **Prevalence (base rate):** the proportion of sick patients in the population. Lowering it wrecks a test's predictive value even when its AUC doesn't change.
- **Leakage:** test-set information the model already saw during training (for example, duplicate rows present in both).
- **Corpus separability:** the dataset separates the classes by construction. An AUC of 1.0 there doesn't measure model quality — it measures the dataset.
- **EPV (events per variable):** how many minority-class cases exist per predictor variable. Below 10, the model is leaning on chance.
- **Brier score / ECE:** measure whether the probability that comes out is the probability that actually happens (reliability), not just whether the ranking is correct.
- **DCA (decision-curve analysis):** compares the model's net benefit against "treat everyone" and "treat no one." A negative DCA means using the model is worse than not using it.

---

*Document generated from measurements taken on the project. The figures in this version correspond to the 2026-09-17 audit; any figure that changes must be re-measured before being cited again.*
