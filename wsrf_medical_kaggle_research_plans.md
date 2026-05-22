# WSRF · Medical Research Plans (Kaggle Benchmarks)

**Started:** 2026-05-22
**Goal:** Benchmark the WSRF library family against public Kaggle datasets across multiple medical domains, looking specifically for tasks whose underlying structure matches WSRF's strength — combinatorial / hidden-parity-style signals where single-feature marginals are weak but specific feature combinations are predictive.

**Why broad, not narrow:** Many medical problems share the same mathematical shape: epistasis, gene-gene interactions, drug combinations, comorbidity patterns, multi-marker disease signatures. Testing across domains shows whether WSRF's documented 38pp gap on synthetic hidden-parity tasks is a general phenomenon or specific to one biology. One area's null result paired with another's positive result is still a publishable, informative outcome.

**Working principle:** One dataset → one notebook → one honest write-up. Positive, null, and negative results all logged the same way. No domain advances to its next phase until the current one is documented.

---

## Phase 1 · Bone marrow / hematologic cancers (2 projects)

### 1A. Golub Gene Expression Dataset — AML vs. ALL
- **Link:** https://www.kaggle.com/datasets/crawford/gene-expression
- **What it is:** The canonical 1999 Golub dataset. 7,129 gene expression features × 72 bone-marrow samples. Binary classification: acute myeloid vs. acute lymphoblastic leukemia.
- **Why this one:** Smallest, fastest to run, decades of published baselines (~85% original, ~95%+ modern). Standard reference point — any meaningful result is immediately legible to bioinformaticians.
- **Combinatorial-structure angle:** Discriminating gene sets are believed to involve interaction effects beyond marginal expression levels. Standard methods already perform well; the test is whether WSRF reaches the same performance via a different mechanism, or improves on it.
- **Status:** Not started
- **Notebook:** —
- **Result:** —

### 1B. Multiple Myeloma Survival (Kaggle competition)
- **Link:** https://www.kaggle.com/competitions/multiple-myeloma-survival
- **What it is:** Survival-time prediction for multiple myeloma patients. Semi-supervised learning angle, continuous censored outcome.
- **Why this one:** Pairs with 1A to cover both major bone-marrow cancer categories (leukemia + myeloma) AND two distinct problem shapes (classification + survival). Has a public leaderboard, so the comparison is direct.
- **Combinatorial-structure angle:** Patient survival depends on combinations of genetic risk markers, treatment response, and comorbidities — exactly the kind of multi-factor interaction surface WSRF is designed for.
- **Status:** Not started
- **Notebook:** —
- **Result:** —
- **Caveat:** Confirm the competition is still accepting submissions or that the historical leaderboard is queryable.

---

## Phase 2 · Genetic interaction / epistasis (strongest theoretical fit)

### 2A. SNP Dataset for GWAS
- **Link:** https://www.kaggle.com/datasets/seascape/snp-dataset-for-gwas
- **What it is:** Single-nucleotide polymorphism data comparable to Illumina 650K human arrays — the substrate genome-wide association studies are built on.
- **Why this one:** Epistasis (gene-gene interactions where individual SNP effects are tiny but combinations predict disease) is the textbook example of hidden-parity-style structure in biology. The well-known "missing heritability" problem is *explicitly* a story about combinatorial effects standard linear GWAS methods can't see. This is the strongest theoretical fit for WSRF anywhere in medicine.
- **Combinatorial-structure angle:** Direct. The known problem is exactly what WSRF claims to solve.
- **Status:** Not started
- **Notebook:** —
- **Result:** —

---

## Phase 3 · Drug interactions and pharmacogenomics

### 3A. Drug-Drug Interactions
- **Link:** https://www.kaggle.com/datasets/mghobashy/drug-drug-interactions
- **What it is:** Two CSVs — drug interaction pairs (with action and mechanism) and a drug information table with 1,258 drugs.
- **Why this one:** Adverse interactions are by definition combinations of agents that are individually safe. Predicting which pairs will interact is a combinatorial classification problem with structural similarity to parity.
- **Combinatorial-structure angle:** Direct. The signal exists only at pairwise (or higher) level.
- **Status:** Not started
- **Notebook:** —
- **Result:** —

### 3B. Drug Effectiveness Dataset
- **Link:** https://www.kaggle.com/datasets/liz2048/drug-effectiveness-dataset
- **What it is:** Drug response data — patient features paired with whether a drug worked.
- **Why this one:** Pharmacogenomic response is patient-genotype × drug interaction — also combinatorial.
- **Status:** Not started
- **Notebook:** —
- **Result:** —

---

## Phase 4 · ICU / sepsis / acute deterioration

### 4A. Sepsis Early Risk Prediction Challenge
- **Link:** https://www.kaggle.com/competitions/sepsis-early-risk-prediction-challenge
- **What it is:** ICU time-series feature engineering for early sepsis prediction. Competition with a leaderboard.
- **Why this one:** Sepsis onset is a collapse signal — multiple weak indicators across vitals, labs, and demographics combining into a state transition. No single variable predicts it; the combination does.
- **Combinatorial-structure angle:** Strong. Existing best methods are tree ensembles that capture interactions; WSRF could plausibly match or beat them via a different inductive bias.
- **Status:** Not started
- **Notebook:** —
- **Result:** —

### 4B. MIMIC-IV Style ICU Dataset for Sepsis Prediction
- **Link:** https://www.kaggle.com/datasets/sinanshereef/mimic-iv-style-icu-dataset-for-sepsis-prediction
- **What it is:** MIMIC-IV-style ICU patient records prepared for sepsis prediction tasks.
- **Why this one:** Sister dataset to 4A — different prep, similar structure. Useful for cross-validating a 4A result on independent data.
- **Status:** Not started
- **Notebook:** —
- **Result:** —

---

## Phase 5 · Cardiovascular risk (combinatorial risk factors)

### 5A. UCI Heart Disease Data (Cleveland)
- **Link:** https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data
- **What it is:** The Cleveland heart disease dataset — 13–14 features, ~300 patients. Binary cardiovascular disease classification.
- **Why this one:** Tiny dataset that runs in seconds, extremely well-benchmarked (decades of published comparisons). A clean place to sanity-check WSRF's behavior on small-sample interaction data.
- **Combinatorial-structure angle:** Moderate. Risk factors interact (cholesterol × age × exercise patterns, etc.) but the dataset is small and the interactions are likely shallow.
- **Status:** Not started
- **Notebook:** —
- **Result:** —

---

## Phase 6 · Image-based tasks (lower priority for WSRF)

WSRF's edge is on tabular / combinatorial signals. Images are CNN/transformer territory. Listed here for completeness; only revisit if a feature-extraction path opens where WSRF could operate on extracted features.

### 6A. Bone Marrow Cell Classification
- **Link:** https://www.kaggle.com/datasets/andrewmvd/bone-marrow-cell-classification
- **What it is:** ~170,000 microscope images of bone marrow cells with hematologic disease labels.
- **Status:** Deferred.

### 6B. SegPC-2021 — Plasma Cell Segmentation
- **Link:** https://www.kaggle.com/datasets/sbilab/segpc2021dataset
- **What it is:** Microscopy image segmentation for multiple myeloma plasma cells.
- **Status:** Deferred.

---

## Beyond Kaggle — for when something works

The Kaggle datasets are mostly mirrors or simplified versions of larger primary research data. If any Phase 1–5 result is strong, the next-step data lives in:

- **GEO** (Gene Expression Omnibus) — deeper gene expression archives (NCBI)
- **ArrayExpress** — EBI equivalent
- **TCGA** (The Cancer Genome Atlas) — multi-modal cancer data
- **GDSC / CCLE / CTRP** — pharmacogenomic cancer cell-line databases
- **MIMIC-IV** (full) — full ICU records from PhysioNet
- **dbGaP** — controlled-access GWAS data
- **DrugBank / TwoSides** — drug interaction databases

The path forward from a strong Kaggle result is: take it to the matching primary source → bioinformatics journal or arXiv q-bio → reach the labs working on the disease.

---

## Result Log

(Append findings here as they land. Format: `YYYY-MM-DD · Phase · Dataset · One-line outcome (positive / null / negative) · Link to notebook`.)

- *(empty — Phase 1A is the next action)*

---

## Honest Checkpoints

- [ ] Phase 1A complete — Golub dataset, WSRF result documented
- [ ] Phase 1A write-up published as a public Kaggle notebook
- [ ] Decision: 1B (myeloma survival) or jump to Phase 2 (epistasis — strongest theoretical fit)
- [ ] Phase 2A complete — SNP/GWAS epistasis result documented
- [ ] First positive result → write up for arXiv q-bio
- [ ] Two positive results across different domains → consider domain conference / journal submission
- [ ] All Phase 1–5 complete (positive or null) → publish a summary "WSRF across medical tasks" paper

---

## Where the work runs

- **WSRF library + demo paths:** see `reference_library_paths.md` in memory for current import paths and PYTHONPATH commands.
- **Suggested environment:** Python 3.13, scikit-learn for baselines, pandas/numpy for prep, matplotlib for charts.
- **Notebook venue:** Public Kaggle notebooks preferred — the result, the code, and the dataset version get locked together in one citeable artifact.
