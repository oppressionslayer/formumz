# WSRF · Bone Marrow Cancer Research Plan

**Started:** 2026-05-22
**Goal:** This is a project i'm going to be working on for cancer research and plan to provide results here soon, we want to see how the WSRF libraries can be applied to things like cancer research. To help us see the use here you can run join us and use the WSRF library family against public Kaggle datasets for bone marrow cancers (multiple myeloma, leukemias) and document any result — better, same, or worse than published baselines. Publishable findings get written up; null results get logged honestly so we don't repeat them. Initial cancer research shows very promising results with WSRF. This will help further our understanding of its mathematical framework and capabilities.

**Why this is a plausible angle for WSRF:** Gene-gene interactions in cancer biology often have the property that a single gene's marginal effect is near zero, but a specific combination predicts the phenotype — structurally similar to hidden parity. That's where standard ML hits the wall, and where WSRF showed a 38pp gap on synthetic parity tasks. Whether that edge transfers to real gene expression data is the question this research answers.

**Working principle:** One dataset → one notebook → one honest write-up. No skipping ahead.

---

## Phase 1 — Foundation (start here)

### 1. Golub Gene Expression Dataset (AML vs. ALL)
- **Link:** https://www.kaggle.com/datasets/crawford/gene-expression
- **What it is:** The canonical 1999 Golub et al. dataset. 7,129 gene expression features, 72 bone marrow samples, AML vs. ALL classification.
- **Why first:** Smallest, most-benchmarked, decades of published baselines to compare against. End-to-end run in an afternoon.
- **Baseline to beat:** Original Golub paper ~85% accuracy on test set; modern methods reach ~95%+. Any meaningful improvement is publishable; matching state of the art with a different mechanism (parity-style feature interactions) is also publishable.
- **Status:** Not started
- **Notebook:** —
- **Result:** —
- **Notes:** —

---

## Phase 2 — Scaling up on leukemia

### 2. Leukemia Gene Expression — CuMiDa
- **Link:** https://www.kaggle.com/datasets/brunogrisci/leukemia-gene-expression-cumida
- **What it is:** Curated microarray collection across multiple leukemia subtypes (more samples, more classes than Golub).
- **Why second:** Natural follow-on if Phase 1 result is interesting — same task type, more data, multi-class.
- **Status:** Not started
- **Notebook:** —
- **Result:** —
- **Notes:** —

### 3. Acute Myeloid Leukemia Dataset
- **Link:** https://www.kaggle.com/datasets/ukveteran/acute-myeloid-leukemia
- **What it is:** Smaller tabular AML dataset.
- **Why:** Sanity check / cross-validation of Phase 1 findings on a different sample population.
- **Status:** Not started
- **Notebook:** —
- **Result:** —
- **Notes:** —

---

## Phase 3 — Multiple myeloma

### 4. Multiple Myeloma Survival Competition
- **Link:** https://www.kaggle.com/competitions/multiple-myeloma-survival
- **What it is:** Kaggle competition — predictive survival modeling for multiple myeloma patients, semi-supervised learning angle.
- **Why:** Has a leaderboard, so the result is directly benchmarked against other approaches. Survival prediction is a different problem shape than classification — interesting test of whether WSRF generalizes.
- **Status:** Not started
- **Notebook:** —
- **Result:** —
- **Notes:** Check whether the competition is still accepting submissions or if it's archived.

### 5. Multiple Myeloma Dataset (Priyadharshini)
- **Link:** https://www.kaggle.com/datasets/mpriyadharshinimca/multiple-myeloma/versions/1
- **What it is:** Smaller tabular multiple myeloma dataset.
- **Why:** Lower-stakes warm-up before the survival competition.
- **Status:** Not started
- **Notebook:** —
- **Result:** —
- **Notes:** —

---

## Phase 4 — Image-based tasks (optional / lower priority for WSRF)

WSRF's edge is on combinatorial/parity-structured tabular data. Images are CNN/transformer territory. Listed here for completeness but not the focus.

### 6. Bone Marrow Cell Classification
- **Link:** https://www.kaggle.com/datasets/andrewmvd/bone-marrow-cell-classification
- **What it is:** 170,000 microscope images of bone marrow cells, hematologic disease labels.
- **Status:** Not started
- **Notes:** Only attempt if a path opens for WSRF on image features.

### 7. SegPC-2021 — Plasma Cell Segmentation
- **Link:** https://www.kaggle.com/datasets/sbilab/segpc2021dataset
- **What it is:** Microscopy image segmentation of multiple myeloma plasma cells.
- **Status:** Not started
- **Notes:** Image segmentation, deferred.

---

## Beyond Kaggle — for context

The clinical-research-relevant data lives in:
- **GEO** (Gene Expression Omnibus) — NCBI, free, deeper datasets than Kaggle mirrors
- **ArrayExpress** — EBI equivalent
- **TCGA** (The Cancer Genome Atlas) — multi-modal cancer data
- **Beat AML 1.0** — registry of open data on AWS, deep AML cohort

If any Phase 1–3 result is strong, the path forward is: take it from Kaggle to GEO/TCGA → bioinformatics journal or arXiv q-bio → reach hematologic oncology labs.

---

## Result Log

(Append findings here as they land. Date · Dataset · One-line summary.)

- *(empty)*

---

## Honest checkpoints

- [ ] Phase 1 complete — Golub dataset, WSRF result documented (positive or null)
- [ ] Phase 1 result written up as a notebook + short README
- [ ] Decision: scale to Phase 2, or pivot
- [ ] Phase 2 complete
- [ ] Phase 3 complete
- [ ] Decision: publish, or shelve

---

## Reference — software paths

(Filled in when needed — pointers to the WSRF library, demo scripts, and Python env this work runs in.)
