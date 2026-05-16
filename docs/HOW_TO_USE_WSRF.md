# How to use WSRF on the problems other ML can't touch

A practical guide to using the **Williams Structured Random Forest** (`wsrf-lib`) on
problems where gradient-based, tree-based, and kernel-based learners hit a wall.

If you've ever shipped an ML model that *should* have worked and didn't — features
look fine, loss curves look fine, validation is at chance — there's a real
possibility your data has *structural* signal that local learners cannot see.
WSRF is built for exactly that case.

---

## The result this guide is built around

| Scenario | Best baseline | WSRF | Gap |
|---|---|---|---|
| 4-way XOR over 12 features | XGBoost 91.35 % | **98.55 %** | +7.2 pp |
| 5-way XOR over 16 features | MLP 58.45 % | **97.05 %** | **+38.6 pp** |
| 6-way XOR over 20 features | LogReg 50.25 % (chance) | **76.65 %** | **+26.4 pp** |

Every standard baseline (LogReg, RandomForest-500, GradientBoosting-300,
XGBoost-500, 3-layer MLP) falls to chance at 5-way XOR. WSRF stays at 97 %.

Reproducible in 90 seconds:
```bash
    python ai_created_demos/wsrf_parity_showdown.py
```

---

## Why standard ML fails here (the capability gap)

Hidden parity is the textbook example of a function with **zero local gradient**:
each individual feature is statistically independent of the label. Every split a
decision tree could make, every gradient a boosted ensemble could compute, every
inner product a kernel could measure — all return *zero signal* per feature.

This is not an obscure edge case. It's the structural shape of:

- **Epistatic gene interactions** — disease phenotype depends on XOR-like
  combinations of multiple SNPs. Standard GWAS misses these by design.
- **Cryptographic side-channels** — leakage often appears as a parity of state bits.
- **Hidden-regime classification** — game states, market regimes, patient
  subtypes where the *combination* of indicators matters, not any single one.
- **Symbolic / combinatorial reasoning** — Rubik's-cube state parity, Boolean
  formula satisfiability features, constraint-graph properties.
- **Heterogeneous treatment effects** — the same intervention helps zone A and
  hurts zone B; the zone boundary isn't a single feature.

The standard prescription for parity in the ML literature is "you need exponentially
many samples or hand-crafted features." That's true for local learners. It's
not true if you use the right algebra.

---

## How WSRF actually solves it

WSRF doesn't fight the gradient problem; it sidesteps it.

1. **Newly-discovered post-Euclid/Gauss number-theoretic boundary detection**:
   each feature is binarized via a novel boundary-detection technique that
   exposes parity and modular-arithmetic relations invisible to standard
   methods. Internals under LREL license.
2. **GF(2) Gauss-Jordan**: run linear-algebra-over-bits on the resulting binary
   frame. Hidden XOR relationships fall out of the *null space*.
3. **Zone stratification**: assign each sample to a discovered zone (regime).
   Each zone gets its own forest, with adaptive complexity.
4. **Calibrated ensemble**: OOB isotonic calibration; predictions are
   well-calibrated probabilities, not raw scores.

The technique that does the heavy lifting is the *GF(2) projection*. Linear
algebra over the binary field has been the right tool for parity for sixty years
— Berlekamp-Massey, Block Lanczos, LDPC decoding all use it. WSRF is what
happens when you wire that into a sklearn-shaped API and put a zone-stratified
forest on top.

---

## Quick start

```python
# pip install wsrf-lib 
from wsrf import WSRFClassifier

model = WSRFClassifier(random_state=42)
model.auto_discover(X_train, y_train)   # finds zones + parity automatically
predictions = model.predict(X_test)
probs       = model.predict_proba(X_test)

# Interpretability — the part that distinguishes WSRF in regulated settings
print(model.discovery_result_.summary())   # what zones / parities were found
rules = model.unblackbox()                 # human-readable IF-THEN rules
```

That's it. No feature engineering, no hyperparameter sweep to discover hidden
structure. `auto_discover` does it.

---

## Where to apply it (concrete pickups)

### When to reach for WSRF

| Symptom | Probable cause | Try WSRF? |
|---|---|---|
| All features look uninformative individually | Hidden interaction structure | **Yes** |
| Validation accuracy stuck near chance with strong models | Parity / modular / zone signal | **Yes** |
| Tree ensembles disagree wildly on importance per-fold | Distribution mixture | **Yes** |
| Need a regulator-readable model | Black-box ensemble unacceptable | **Yes** (unblackbox) |
| Smooth function of features, simple boundary | Gradient-boosting territory | Probably not |
| Pixel/audio/text raw input | Different problem class | No (use deep nets) |

### Real-world starting points

- **Tabular health / clinical**: heart disease, sepsis prediction. Patient
  regimes are zones; comorbidity interactions are parity-like. (See the
  `playground-heart-disease-main` results: LB 0.94969 with full
  interpretability.)
- **Tabular fraud / abuse**: zone-stratified forests catch coordinated
  collusion that black-box ensembles diffuse.
- **Bioinformatics**: epistatic SNP detection — the canonical real-world parity
  problem.
- **Combinatorial puzzles**: Sudoku, constraint satisfaction. The `larsdoku`
  module already proves this — 100 % pure logic on Top1465.
- **Tabular Kaggle competitions** where the leaderboard has a hard ceiling for
  XGBoost-class models — that ceiling is often a hidden zone or parity
  structure that no amount of boosting can uncover.

---

## How this fits the bigger picture

It's worth being precise about scope. WSRF solves *one* specific capability gap
in current ML: discovery of low-dimensional algebraic structure in tabular data.
That gap is real, it's well-documented, and it has resisted neural-network and
gradient-boosting attacks for decades. WSRF closes it for the cases the algebra
supports (XOR, modular arithmetic, the proprietary boundary-detection layer, zone partitions).

Where this matters for *more advanced* reasoning systems:

1. **Compositional generalization.** Current LLMs/foundation models struggle to
   generalize compositionally — to combine known primitives into novel
   structured wholes. Parity-like reasoning ("this iff that XOR the other") is
   a load-bearing piece. A system that can't even detect 5-way XOR in tabular
   data won't reason about it linguistically either.

2. **Symbolic + statistical hybrid.** The clean way forward for advanced AI is
   widely understood to be the marriage of symbolic structure with statistical
   learning. WSRF is a working example: the symbolic layer (GF(2) algebra +
   zone discovery) feeds the statistical layer (zone-stratified forest). The
   pipeline is interpretable end-to-end.

3. **Verifiable reasoning.** Every WSRF prediction can be traced to a zone
   assignment + a set of human-readable rules. That kind of audit trail is what
   safety-critical and regulated deployments will require — black-box
   ensembles alone don't pass review.

4. **Structure-aware exploration.** In RL and active learning, an agent that
   can detect when a hidden algebraic relation governs reward (versus when
   smooth optimization will do) gets to allocate its exploration budget
   correctly. Current methods often waste samples thrashing in regimes where
   gradient signal doesn't exist.

The honest framing: **WSRF is not a general-intelligence system; it's a
capability building-block that general-intelligence systems will need to
incorporate (or independently rediscover).** Anyone serious about advancing AI
beyond curve-fitting will eventually have to confront the parity-class
capability gap. WSRF gets there now.

---

## Read more / next steps

- `wsrf_parity_showdown.py` — the 90-second demo behind the numbers above.
- `wsrf_parity_full_benchmark.py` — multi-seed CIs, 7- and 8-way XOR, noise
  sweep, distribution-ready report.
- `wsrf-lib/README.md` — full API documentation, v5 release notes.
- `wsrf-hlib/` — high-level wrappers, heart-disease and Sudoku demos.
- `forensic_syndrome_lie_detector.py` — sister demo showing GF(2)
  syndrome decoding on adversarial knowledge bases.

Pull requests, benchmarks against your own datasets, and replication reports
welcome.
