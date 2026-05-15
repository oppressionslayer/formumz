# Rule 30 Inversion — A Forty-Year-Old "Non-Reversible" Claim, Reversed

Stephen Wolfram (1985) declared Rule 30 — the elementary cellular automaton he
later put at the center of *A New Kind of Science* — **non-reversible**, and
proposed its center column as a cryptographically random one-way function.
That framing has been textbook for four decades.

This project contains a constructive inverse for the canonical finite case
(width grows by 2 per step from a single seed bit, which is the case the
cryptographic-randomness argument actually relies on). The inverse runs in
**O(w) per step**, needs **no zero-padding** or boundary assumptions, and
decomposes into **independent GF(2) blocks** — meaning each inverse transition
is a square parity-check system with a unique solution.

## The headline

| | Value |
|---|---|
| Cascade size verified | 1024 rows |
| Width at row 1024 | 2049 bits |
| Rows recovered exactly | **1023 / 1024** (the seed has no predecessor) |
| Time (interpreted Python) | **0.29 seconds** |
| Time per inverse step | ~0.29 ms |

Recovery is exact, not statistical. Every bit is correct.

## The structure underneath

Apply a simple transformation to each row of the cascade — **bit-reverse, then
strip leading and trailing zeros**, then read the result as an integer. Call
this the *flip-strip value*. The cascade's flip-strip sequence is:

```
1, 7, 25, 111, 401, 1783, 6409, 28479, 102849, 456263, 1641433, 7287855,
26332369, 116815671, 420186569, ...
```

Properties of this sequence (verifiable by running `03_rule30.py`):

- **Mod-4 residues alternate `1, 3, 1, 3, ...` exactly.** No exceptions
  observed across all rows tested.
- **Bit length grows by exactly +2 per step.** Not approximately; exactly.
- The sequence is a deterministic function of cascade position. Anyone who
  has worked with structured integer sequences recognizes the shape.

This is what Wolfram's framework called cryptographic randomness. It is
nothing of the kind in the right representation.

## The inverse

The inverse method is three operations per bit:

1. **Flip** — reverse the bit string
2. **Twist** — sliding-window OR-XOR resolution with a 2-bit context state
3. **Pancake flip** — bit-reverse + rotate

Total cost per step: O(w), where w is the row width. The
`rule30_inverse(nxt, w)` function in `03_rule30.py` is about 15 lines of
Python, has no fancy tricks, and works on any row whose data hasn't reached
the mask boundary (every row except the final one in a freshly-grown cascade).

## GF(2) verification

Each inverse transition can also be encoded as a square GF(2) system and
solved with the two-phase Schur complement solver. The verification in step 3
of the demo confirms:

- Every inverse transition has rank equal to the row width
- No free variables (the inverse is uniquely determined)
- No contradictions (the inverse always exists)
- Independent GF(2) blocks — the inverse problem is parallelizable

These are the algebraic invariants that say the inverse exists *and is unique*.
Not an empirical observation; a structural property of the system.

## How to reproduce

```bash
pip install schur-logic
python demos/03_rule30.py
```

A 90-second run produces the full output: forward cascade visualization,
per-row backward recovery with ASCII display, GF(2) Schur verification, the
flip-strip sequence printed out, and the timing table at 32 / 128 / 512 / 1024
rows.

If your numbers match, the textbook claim that Rule 30's center column is
non-reversible in the canonical finite case does not hold up. The math is in
the test file at `schur-logic/tests/test_rule30.py`; all six tests pass with
no boundary assumptions.

## Scope and limits

To be precise about what this does and does not claim:

- **Claim:** In the finite-width-from-seed case (width = 2N+1 at row N), the
  inverse is constructive, runs in O(w) per step, and recovers every row
  except the boundary row exactly.

- **Not claimed:** That Rule 30 in the *infinite-width* setting is reversible.
  The standard non-reversibility theorem (two distinct infinite rows mapping
  to the same successor) is unaffected.

- **Implication for the cryptographic-randomness framing:** Wolfram proposed
  the center column as a one-way function. The Rocha inverse takes O(N²)
  total work to recover all N center bits from the seed — same complexity
  class as running the rule forward N times. So strictly: the inverse does
  not "break" the one-way function property *in the cryptographic complexity
  sense* — both directions are polynomial. What it does break is the
  intuitive framing of "the center column is irreducibly random." It is not.
  The flip-strip sequence is structured, deterministic, and visibly so once
  you look at it in the right representation.

- **Open question:** whether the 4-cycle structure allows computing center
  bit N in *less* than O(N²) total work (e.g., O(N log N) or O(N)). If yes,
  that *would* break the cryptographic-randomness framing in the strict
  complexity sense. The current code does not do this, but the structure
  observed suggests it may be possible.

## Why this matters more broadly

The takeaway isn't only about Rule 30. It's about a general principle:

**Chaos in continuous representation often has closed-form or polynomial-time
inverse in the right integer-space representation.** Same dynamics, different
lens, completely different visible structure.

Other instances of this principle:

- **Lorenz attractor**: chaotic in `(x, y, z)`, fractal in phase space.
- **Riemann zeta zeros**: appear random, have random-matrix-theory spectral
  statistics encoding deep structure.
- **Hidden parity in tabular ML** (see `wsrf_parity_showdown.py` in this
  repo): invisible to gradient methods, exposed by GF(2) projection.

For agents that need to reason backward from observations to causes — every
scientific discovery process, every theory-of-mind reasoner, every
counterfactual inference engine — *integer-space inversion is the move that
makes the inverse tractable*. This is one concrete worked example. There
will be more.

## References in this repo

- `schur-logic` package — original write-up with the 4-cycle structure (`RULE30.md`)
- `schur-logic` test suite — six passing tests (`tests/test_rule30.py`)
- `demos/03_rule30.py` — runnable showcase
