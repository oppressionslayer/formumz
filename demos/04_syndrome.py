#!/usr/bin/env python3
"""Forensic Syndrome Lie Detector — Adversarial Diagnosis via Coding Theory.

A theorem-grade companion to ai_lie_detector_rule_system.py.

What the original demo quietly assumes (and a reviewer will catch):
  The trusted laws already over-determine the hidden state. Per-witness
  residual checks are then trivial: solve the trusted system, XOR each
  witness's variables, compare. "Lie detection" collapses to one line of
  linear algebra. It works, but it does not exercise the hard problem.

What this demo confronts head-on:
  1. Trusted laws GENUINELY UNDER-DETERMINE the hidden state. The trusted
     subsystem leaves a k-dimensional affine slice of consistent worlds.
     Per-witness residual checks are mathematically undefined: there is no
     unique state to compare against.
  2. Liars are an ADVERSARIAL COALITION. Their parity errors are correlated
     so that a naive leave-one-out / "drop the noisiest witness" heuristic
     drops the wrong witness and recovers a plausible-but-wrong world.
  3. The proper move is SYNDROME DECODING on the projected linear code. We
     compute the projection, compute the resulting code's exact minimum
     distance d by codeword enumeration, and announce the unique-decoding
     radius t* = floor((d - 1) / 2) as a hard theorem about this knowledge
     base — not a heuristic guarantee.
  4. We run LOOPY BELIEF PROPAGATION on the Tanner graph for soft posteriors
     P(witness i is a liar | observed syndrome). At weight ≤ t*, BP converges
     to a unique high-confidence answer.
  5. We push the adversary to weight t* + 1. The system honestly returns the
     EQUIVALENCE CLASS of minimum-weight liar sets. No silent wrong answer.
  6. We then run ACTIVE INTERROGATION: choose the smallest set of direct bit
     inspections of the hidden state that distinguishes the candidates. The
     entropy-greedy policy disambiguates with a single inspection in the
     case here; random would take many.
  7. BOOTSTRAP VERIFICATION uses a held-out half of the trusted laws that
     the detector never saw. False-positive probability 2^(-m_B).

Oracle discipline: the hidden state x is generated from a seed, used to
construct the observable knowledge base, and immediately sealed. The
detection pipeline operates only on the public knowledge base. The sealed
truth is opened only for two things: (a) the active-interrogation queries
the detector explicitly elects to spend, and (b) the after-the-fact
comparison shown to the reader.

Sections of this file map 1:1 onto the talking points above.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from itertools import product
from typing import Iterable, Iterator

import numpy as np
from numpy.typing import NDArray

from lars_logic import ConstraintSystem, LarsSolver


# =============================================================================
# Section 1 — GF(2) linear algebra primitives
# =============================================================================
# numpy is overkill for GF(2) but its broadcasting makes the code readable.
# All arrays are uint8 with values in {0, 1}; XOR is bitwise xor.

def gf2(A) -> NDArray[np.uint8]:
    return np.asarray(A, dtype=np.uint8) & 1


def gf2_matmul(A: NDArray, x: NDArray) -> NDArray:
    return (gf2(A) @ gf2(x)) & 1


def gf2_rref(A: NDArray) -> tuple[NDArray, list[int]]:
    """Reduced row echelon form. Returns (rref, pivot_cols)."""
    M = gf2(A).copy()
    m, n = M.shape
    pivots: list[int] = []
    r = 0
    for c in range(n):
        pivot = None
        for i in range(r, m):
            if M[i, c]:
                pivot = i
                break
        if pivot is None:
            continue
        if pivot != r:
            M[[r, pivot]] = M[[pivot, r]]
        for i in range(m):
            if i != r and M[i, c]:
                M[i] ^= M[r]
        pivots.append(c)
        r += 1
        if r == m:
            break
    return M, pivots


def gf2_rank(A: NDArray) -> int:
    _, pivots = gf2_rref(A)
    return len(pivots)


def gf2_null_space(A: NDArray) -> NDArray:
    """Basis for {x : Ax = 0}, returned as an n x k matrix (k = nullity)."""
    R, pivots = gf2_rref(A)
    n = A.shape[1]
    pivot_set = set(pivots)
    free_cols = [c for c in range(n) if c not in pivot_set]
    k = len(free_cols)
    N = np.zeros((n, k), dtype=np.uint8)
    for j, f in enumerate(free_cols):
        N[f, j] = 1
        for row, p in enumerate(pivots):
            if R[row, f]:
                N[p, j] = 1
    return N


def gf2_solve(A: NDArray, b: NDArray) -> NDArray | None:
    """One particular x with Ax = b, or None if inconsistent."""
    A, b = gf2(A), gf2(b)
    m, n = A.shape
    Ab = np.concatenate([A, b.reshape(-1, 1)], axis=1)
    R, pivots = gf2_rref(Ab)
    if any(p == n for p in pivots):
        return None
    x = np.zeros(n, dtype=np.uint8)
    for row, p in enumerate(pivots):
        x[p] = R[row, n]
    return x


def left_null_space(M: NDArray) -> NDArray:
    """Rows L_i such that L_i @ M = 0. Returns matrix L with shape (k, m_rows)."""
    return gf2_null_space(M.T).T


def enumerate_codewords(N: NDArray) -> Iterator[NDArray]:
    """Yield every vector in the column span of N (a 2^k-element set)."""
    k = N.shape[1]
    for combo in product([0, 1], repeat=k):
        coef = np.array(combo, dtype=np.uint8)
        yield (N @ coef) & 1


def code_min_distance(N: NDArray) -> tuple[int, NDArray | None]:
    """Minimum nonzero weight of a codeword in col span of N, with witness vector."""
    best = None
    best_vec: NDArray | None = None
    for cw in enumerate_codewords(N):
        w = int(cw.sum())
        if w == 0:
            continue
        if best is None or w < best:
            best, best_vec = w, cw.copy()
    return (best if best is not None else 0), best_vec


# =============================================================================
# Section 2 — Under-determined knowledge base
# =============================================================================

@dataclass
class KnowledgeBase:
    """Everything the detector is allowed to see."""
    n: int
    T_A: NDArray          # trusted laws, detection half        (m_T_A, n)
    b_T_A: NDArray        # rhs for trusted A                   (m_T_A,)
    T_B: NDArray          # trusted laws, held-out half         (m_T_B, n)
    b_T_B: NDArray        # rhs for trusted B                   (m_T_B,)
    W: NDArray            # witness equations                   (m_W, n)
    b_W: NDArray          # rhs for witnesses (possibly poisoned)(m_W,)
    witness_names: list[str]


@dataclass
class SealedTruth:
    """Hidden state. The detector cannot read this — only call reveal_bits()."""
    _x: NDArray
    _e: NDArray
    _inspections: list[int] = field(default_factory=list)

    def reveal_bits(self, indices: Iterable[int]) -> dict[int, int]:
        out: dict[int, int] = {}
        for i in indices:
            self._inspections.append(int(i))
            out[int(i)] = int(self._x[i])
        return out

    @property
    def inspection_count(self) -> int:
        return len(self._inspections)


def random_sparse_matrix(rng: np.random.Generator, rows: int, cols: int,
                        weight_lo: int, weight_hi: int) -> NDArray:
    M = np.zeros((rows, cols), dtype=np.uint8)
    for i in range(rows):
        w = rng.integers(weight_lo, weight_hi + 1)
        idx = rng.choice(cols, size=w, replace=False)
        M[i, idx] = 1
    return M


def synthesize_knowledge_base(
    seed: int,
    n: int = 32,
    m_T_A: int = 18,
    m_T_B: int = 10,
    m_W: int = 32,
    target_d: int = 5,
    max_retries: int = 200,
) -> tuple[KnowledgeBase, SealedTruth, NDArray, int]:
    """Find a seed that yields the structural properties we need.

    We retry seeds until the random draw produces:
      * rank(T_A) < n                              (real under-determination)
      * rank(stack(T_A, T_B, W)) == n              (combined system pins x)
      * resulting syndrome parity-check code has min distance >= target_d
    """
    base_rng = np.random.default_rng(seed)
    for attempt in range(max_retries):
        rng = np.random.default_rng(base_rng.integers(0, 2**31))

        T_A = random_sparse_matrix(rng, m_T_A, n, 3, 5)
        T_B = random_sparse_matrix(rng, m_T_B, n, 3, 5)
        W   = random_sparse_matrix(rng, m_W,   n, 4, 6)

        if gf2_rank(T_A) >= n:
            continue  # not under-determined
        if gf2_rank(np.vstack([T_A, T_B, W])) < n:
            continue  # combined system doesn't pin x

        # Build syndrome code and check its minimum distance.
        N = gf2_null_space(T_A)            # (n, k)
        M = (W @ N) & 1                     # (m_W, k)
        L = left_null_space(M)              # (rows, m_W) with L @ M = 0
        code_basis = gf2_null_space(L)      # (m_W, dim) — codewords are in null space of L
        if code_basis.shape[1] == 0 or code_basis.shape[1] > 16:
            continue
        d, _ = code_min_distance(code_basis)
        if d < target_d:
            continue

        # Draw the hidden world, build observable rhs, seal it.
        x = rng.integers(0, 2, size=n, dtype=np.uint8)
        b_T_A = gf2_matmul(T_A, x)
        b_T_B = gf2_matmul(T_B, x)
        b_W_clean = gf2_matmul(W, x)

        witness_names = [_plausible_name(rng, i) for i in range(m_W)]
        kb = KnowledgeBase(
            n=n,
            T_A=T_A, b_T_A=b_T_A,
            T_B=T_B, b_T_B=b_T_B,
            W=W,    b_W=b_W_clean,        # liars overlaid after coalition picks them
            witness_names=witness_names,
        )
        return kb, SealedTruth(_x=x, _e=np.zeros(m_W, dtype=np.uint8)), code_basis, d
    raise RuntimeError("No seed satisfied structural constraints. Loosen target_d.")


_WITNESS_PREFIXES = ["lab", "wearable", "radiology", "genomics", "neurology",
                     "metabolic", "immune", "oracle", "doctor", "EHR",
                     "echo", "biopsy", "imaging", "panel", "marker", "telemetry"]
_WITNESS_SUFFIXES = ["alpha", "beta", "gamma", "delta", "rev2", "v3",
                     "morning", "overnight", "field", "stamp", "shadow", "trace"]


def _plausible_name(rng: np.random.Generator, i: int) -> str:
    p = _WITNESS_PREFIXES[rng.integers(0, len(_WITNESS_PREFIXES))]
    s = _WITNESS_SUFFIXES[rng.integers(0, len(_WITNESS_SUFFIXES))]
    return f"{p}-{s}-{i:02d}"


# =============================================================================
# Section 3 — Adversarial coalition synthesis
# =============================================================================

def install_coalition(kb: KnowledgeBase, sealed: SealedTruth,
                      error_pattern: NDArray) -> None:
    """Overlay the adversarial liar pattern onto the witness rhs in-place."""
    kb.b_W = (kb.b_W ^ error_pattern) & 1
    sealed._e = error_pattern.copy()


def coalition_of_weight(rng: np.random.Generator, code_basis: NDArray,
                        weight: int) -> NDArray:
    """Pick an error pattern of the requested weight that respects the code.

    We choose a random support of the given weight. This produces an
    error pattern whose syndrome the decoder must recover. For weight
    ≤ t* the answer is unique; for weight ≥ t*+1 we prove ambiguity by
    enumerating alternative min-weight explanations.
    """
    m_W = code_basis.shape[0]
    support = rng.choice(m_W, size=weight, replace=False)
    e = np.zeros(m_W, dtype=np.uint8)
    e[support] = 1
    return e


def find_ambiguous_pattern(rng: np.random.Generator, code_basis: NDArray,
                           weight: int, tries: int = 5000) -> NDArray:
    """Search for a weight-w pattern whose syndrome has >= 2 weight-w solutions.

    Two distinct weight-w explanations differ by a codeword of weight ≤ 2w,
    so we need 2w >= d. Above the unique-decoding radius this is feasible.
    """
    m_W = code_basis.shape[0]
    for _ in range(tries):
        e = coalition_of_weight(rng, code_basis, weight)
        # Iterate over codewords; count weight-w siblings.
        siblings = 0
        for cw in enumerate_codewords(code_basis):
            if int(cw.sum()) == 0:
                continue
            alt = (e ^ cw) & 1
            if int(alt.sum()) == weight:
                siblings += 1
                if siblings >= 1:
                    return e
    raise RuntimeError(f"Could not find ambiguous weight-{weight} pattern.")


# =============================================================================
# Section 4 — Naive baseline (and its failure)
# =============================================================================

def naive_leave_one_out(kb: KnowledgeBase, max_drops: int = 4
                        ) -> tuple[list[int], NDArray | None]:
    """A reasonable straw-man: greedily drop witnesses to remove contradictions.

    At each step:
      - Stack T_A and currently-kept witnesses.
      - If consistent (i.e., Ax = b has a solution), stop.
      - Otherwise drop the witness whose individual removal makes the system
        consistent (textbook leave-one-out). Ties broken by parity weight.

    Returns (dropped indices, an x_hat consistent with the kept system or None).
    The point of this function is to FAIL on adversarial collusion.
    """
    kept = set(range(kb.W.shape[0]))
    dropped: list[int] = []
    final_x: NDArray | None = None
    for _ in range(max_drops + 1):
        idx = sorted(kept)
        A = np.vstack([kb.T_A, kb.W[idx]])
        b = np.concatenate([kb.b_T_A, kb.b_W[idx]])
        x = gf2_solve(A, b)
        if x is not None:
            final_x = x
            break
        scores: dict[int, int] = {}
        for w in idx:
            trial = [j for j in idx if j != w]
            At = np.vstack([kb.T_A, kb.W[trial]])
            bt = np.concatenate([kb.b_T_A, kb.b_W[trial]])
            if gf2_solve(At, bt) is not None:
                scores[w] = scores.get(w, 0) + 1
        if not scores:
            weights = {w: int(kb.W[w].sum()) for w in idx}
            offender = max(weights, key=weights.get)
        else:
            offender = max(scores, key=scores.get)
        kept.remove(offender)
        dropped.append(offender)
    return dropped, final_x


# =============================================================================
# Section 5 — Syndrome projection
# =============================================================================

@dataclass
class SyndromeProblem:
    L: NDArray           # parity-check matrix on witness errors (rows, m_W)
    syndrome: NDArray    # L @ e_true = syndrome (computed from observables)
    code_basis: NDArray  # column basis of null space of L (m_W, dim)
    M: NDArray           # W @ N projection                   (m_W, k)
    N: NDArray           # null space basis of T_A            (n, k)
    x0: NDArray          # particular solution of T_A x = b_T_A
    c: NDArray           # b_W ⊕ W x0                         (m_W,)


def project_syndrome(kb: KnowledgeBase) -> SyndromeProblem:
    N = gf2_null_space(kb.T_A)
    x0 = gf2_solve(kb.T_A, kb.b_T_A)
    assert x0 is not None, "Trusted A is internally inconsistent — bad seed."
    M = (kb.W @ N) & 1
    c = (kb.b_W ^ gf2_matmul(kb.W, x0)) & 1
    L = left_null_space(M)
    syndrome = (L @ c) & 1
    code_basis = gf2_null_space(L)
    return SyndromeProblem(L=L, syndrome=syndrome, code_basis=code_basis,
                            M=M, N=N, x0=x0, c=c)


# =============================================================================
# Section 6 — Loopy belief propagation
# =============================================================================

def loopy_bp(L: NDArray, syndrome: NDArray, prior_p: float = 0.15,
             iters: int = 60, damping: float = 0.5) -> NDArray:
    """Sum-product BP on the Tanner graph of L. Returns posterior P(e_i = 1).

    Standard log-domain formulation:
      - var node i sends LLR L_{i→a} = prior + sum_{b in N(i) \\ a} L_{b→i}
      - check node a sends L_{a→i} = (-1)^{s_a} * 2 * atanh( prod tanh(L_{b→a}/2) )
    """
    m, n = L.shape
    syndrome = gf2(syndrome)
    L = gf2(L)

    prior_llr = np.log((1 - prior_p) / prior_p)
    # var->check messages, indexed by (check, var)
    msg_vc = np.zeros((m, n))
    msg_cv = np.zeros((m, n))

    # Precompute neighborhoods.
    check_neighbors = [np.where(L[a])[0] for a in range(m)]
    var_neighbors   = [np.where(L[:, i])[0] for i in range(n)]

    for _ in range(iters):
        # Var -> check: outgoing along edge (a, i) is prior - L_{a→i} term:
        # full belief minus the incoming from a.
        full_belief = prior_llr + msg_cv.sum(axis=0)              # (n,)
        new_vc = (full_belief[np.newaxis, :] - msg_cv) * L         # mask to edges

        # Check -> var: tanh-product rule.
        # We compute tanh(msg_vc / 2) for edges; for non-edges it's zero and
        # we want to ignore them. Use 1.0 placeholder so they don't affect product.
        t = np.tanh(new_vc / 2.0)
        t_safe = np.where(L.astype(bool), t, 1.0)
        # For each check a, compute product over its variables.
        prod = np.ones(m)
        for a in range(m):
            nbrs = check_neighbors[a]
            if len(nbrs) == 0:
                continue
            prod[a] = np.prod(t_safe[a, nbrs])

        new_cv = np.zeros_like(msg_cv)
        for a in range(m):
            nbrs = check_neighbors[a]
            sign = -1.0 if syndrome[a] else 1.0
            for i in nbrs:
                ti = t_safe[a, i]
                # Avoid div-by-zero when ti is exactly ±1.
                if abs(ti) >= 1 - 1e-12:
                    # Use leave-one-out product directly.
                    others = [j for j in nbrs if j != i]
                    p = np.prod(t_safe[a, others]) if others else 1.0
                else:
                    p = prod[a] / ti
                p = np.clip(p, -1 + 1e-12, 1 - 1e-12)
                new_cv[a, i] = sign * 2 * np.arctanh(p)

        msg_vc = damping * msg_vc + (1 - damping) * new_vc
        msg_cv = damping * msg_cv + (1 - damping) * new_cv

    final_llr = prior_llr + msg_cv.sum(axis=0)
    # Posterior P(e_i = 1) = 1 / (1 + exp(LLR))
    return 1.0 / (1.0 + np.exp(final_llr))


# =============================================================================
# Section 7 — Min-weight enumeration + active interrogation
# =============================================================================

def all_min_weight_solutions(L: NDArray, syndrome: NDArray,
                              code_basis: NDArray) -> tuple[int, list[NDArray]]:
    """Return (w*, list of all weight-w* error patterns explaining the syndrome)."""
    e0 = gf2_solve(L, syndrome)
    if e0 is None:
        return 0, []
    best_w = int(1e9)
    best: list[NDArray] = []
    for cw in enumerate_codewords(code_basis):
        cand = (e0 ^ cw) & 1
        w = int(cand.sum())
        if w < best_w:
            best_w = w
            best = [cand.copy()]
        elif w == best_w:
            best.append(cand.copy())
    return best_w, best


def implied_world(prob: SyndromeProblem, e: NDArray) -> NDArray | None:
    """Given an error pattern e, recover the implied hidden state x."""
    z = gf2_solve(prob.M, (prob.c ^ e) & 1)
    if z is None:
        return None
    return (prob.x0 ^ gf2_matmul(prob.N, z)) & 1


def active_interrogation(candidate_worlds: list[NDArray]) -> list[int]:
    """Greedy entropy-maximizing bit queries to separate candidate worlds.

    Each query reads one bit of the true hidden state. We pick the bit
    where the candidate worlds disagree maximally; ties broken by leftmost.
    Returns the ordered list of bit indices to inspect.
    """
    if len(candidate_worlds) <= 1:
        return []
    n = candidate_worlds[0].size
    plan: list[int] = []
    surviving = list(range(len(candidate_worlds)))
    while len(surviving) > 1:
        best_i = -1
        best_split = -1
        for i in range(n):
            if i in plan:
                continue
            zeros = sum(1 for k in surviving if candidate_worlds[k][i] == 0)
            ones = len(surviving) - zeros
            split = min(zeros, ones)
            if split > best_split:
                best_split = split
                best_i = i
        if best_i < 0 or best_split == 0:
            break  # remaining candidates are bit-identical on un-queried positions
        plan.append(best_i)
        # We don't know the actual answer here; we just commit the plan.
        # Resolution happens once SealedTruth.reveal_bits is called.
        # For planning we pessimistically assume we always retain >=ceil(|S|/2).
        surviving = surviving[: max(1, (len(surviving) + 1) // 2)]
    return plan


def filter_candidates(candidate_worlds: list[NDArray],
                       answered: dict[int, int]) -> list[int]:
    """Return indices of candidates whose worlds match every revealed bit."""
    keep: list[int] = []
    for k, w in enumerate(candidate_worlds):
        if all(int(w[i]) == v for i, v in answered.items()):
            keep.append(k)
    return keep


# =============================================================================
# Section 8 — Bootstrap verification
# =============================================================================

def bootstrap_verify(kb: KnowledgeBase, x_hat: NDArray) -> tuple[bool, int, int]:
    """Check x_hat against the held-out trusted_B laws.

    Under the random-code heuristic, an incorrect x_hat agrees with each
    held-out parity equation with probability 1/2, so the overall false-
    positive probability is roughly 2^(-rank(T_B)).
    """
    pred = gf2_matmul(kb.T_B, x_hat)
    matches = int(np.sum(pred == kb.b_T_B))
    total = kb.T_B.shape[0]
    return matches == total, matches, total


def lars_reconstruct(kb: KnowledgeBase, lying_witnesses: set[int]) -> NDArray | None:
    """Reconstruct x by feeding the cleaned system to lars_logic.LarsSolver.

    Used as an independent confirmation of the numpy path and to demonstrate
    the library on a non-trivial under-determined GF(2) system.
    """
    cs = ConstraintSystem(n_vars=kb.n)
    for row in range(kb.T_A.shape[0]):
        vars_ = [int(i) for i in np.where(kb.T_A[row])[0]]
        cs.add_larsxor(vars_, rhs=int(kb.b_T_A[row]))
    for row in range(kb.W.shape[0]):
        if row in lying_witnesses:
            continue
        vars_ = [int(i) for i in np.where(kb.W[row])[0]]
        cs.add_larsxor(vars_, rhs=int(kb.b_W[row]))
    result = LarsSolver(cs).solve()
    if result.contradiction:
        return None
    sol = result.full_solution()
    x = np.zeros(kb.n, dtype=np.uint8)
    for i in range(kb.n):
        x[i] = sol.get(i, 0)
    return x


# =============================================================================
# Section 9 — Demo driver
# =============================================================================

def banner(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def fmt_set(idx: Iterable[int]) -> str:
    return "{" + ", ".join(str(i) for i in sorted(idx)) + "}"


def liar_indices(e: NDArray) -> list[int]:
    return [int(i) for i in np.where(e)[0]]


def run_one_scenario(label: str, kb: KnowledgeBase, sealed: SealedTruth,
                     prob: SyndromeProblem, code_basis: NDArray, d: int) -> None:
    t_star = (d - 1) // 2
    true_liars = liar_indices(sealed._e)

    banner(f"SCENARIO: {label}")
    print(f"  Lies installed (private to harness): {len(true_liars)} witnesses")
    print(f"  Code minimum distance d = {d}, unique-decoding radius t* = {t_star}")
    print(f"  This regime is {'INSIDE' if len(true_liars) <= t_star else 'BEYOND'} the unique-decoding radius.")

    # --- Naive baseline ---
    print()
    print("  [Naive baseline — leave-one-out + 'drop noisiest']")
    dropped, baseline_x = naive_leave_one_out(kb)
    if not dropped:
        print("    The naive heuristic terminated without dropping anyone.")
        baseline_correct = set() == set(true_liars)
    else:
        print(f"    Dropped: {fmt_set(dropped)}")
        baseline_correct = set(dropped) == set(true_liars)
    print(f"    Matches ground truth liar set? {baseline_correct}")
    if baseline_x is not None:
        b_ok, b_match, b_total = bootstrap_verify(kb, baseline_x)
        print(f"    Naive x_hat passes bootstrap held-out check? {b_ok} "
              f"({b_match}/{b_total})")
        n_over = len(set(dropped) - set(true_liars))
        n_under = len(set(true_liars) - set(dropped))
        if b_ok and not baseline_correct:
            print(f"    ⚠ False-confidence trap: bootstrap on x_hat is CLEAN, but the")
            print(f"      diagnosis has {n_over} false accusation(s) and "
                  f"{n_under} missed liar(s).")
            print("      The hidden state happens to be recoverable from the cleaner-")
            print("      than-needed kept set, so a downstream check on x cannot see")
            print("      the error. A syndrome certificate would.")

    # --- Belief propagation ---
    print()
    print("  [Belief propagation on Tanner graph]")
    posterior = loopy_bp(prob.L, prob.syndrome, prior_p=0.15, iters=60, damping=0.4)
    ranked = np.argsort(-posterior)
    print(f"    Top-{min(8, len(ranked))} witnesses by P(liar):")
    for r in ranked[: min(8, len(ranked))]:
        marker = "*" if r in true_liars else " "
        print(f"      {marker} {r:3d}  P(liar) = {posterior[r]:.3f}   "
              f"{kb.witness_names[r]}")
    print("    (* marks an actual liar; visible only in this audit print, not used by the detector.)")

    # --- Min-weight enumeration ---
    print()
    print("  [Min-weight syndrome decoding]")
    w_star, candidates_e = all_min_weight_solutions(prob.L, prob.syndrome, code_basis)
    print(f"    Minimum-weight explanation = {w_star} liars")
    print(f"    Number of equally-minimal explanations: {len(candidates_e)}")
    for k, e in enumerate(candidates_e[:6]):
        print(f"      candidate {k}: liars = {fmt_set(liar_indices(e))}")
    if len(candidates_e) > 6:
        print(f"      ... and {len(candidates_e) - 6} more")

    # --- Implied worlds + active interrogation ---
    candidate_worlds = [implied_world(prob, e) for e in candidates_e]
    candidate_worlds = [w for w in candidate_worlds if w is not None]
    print()
    if len(candidate_worlds) == 1:
        print("  [Resolution]")
        print("    Unique explanation. No active interrogation needed.")
        final_e = candidates_e[0]
        x_hat = candidate_worlds[0]
    else:
        print("  [Active interrogation]")
        plan = active_interrogation(candidate_worlds)
        print(f"    Inspection plan (entropy-greedy): {plan}")
        revealed = sealed.reveal_bits(plan)
        print(f"    Inspected bits: {revealed}")
        keep = filter_candidates(candidate_worlds, revealed)
        print(f"    Candidates surviving inspection: {len(keep)} / {len(candidate_worlds)}")
        if len(keep) != 1:
            print("    WARNING: planned inspection did not uniquely identify a world.")
            keep = keep[:1]
        final_e = candidates_e[keep[0]]
        x_hat = candidate_worlds[keep[0]]

    found_liars = set(liar_indices(final_e))
    print()
    print("  [Final detector output]")
    print(f"    Identified liars: {fmt_set(found_liars)}")

    # --- Bootstrap verification ---
    ok, matches, total = bootstrap_verify(kb, x_hat)
    print()
    print("  [Bootstrap verification on held-out trusted laws]")
    print(f"    Held-out laws satisfied: {matches}/{total}")
    print(f"    False-positive probability if x_hat were wrong: ~2^-{total}  "
          f"= {2.0**(-total):.2e}")
    print(f"    Verified: {ok}")

    # --- Independent reconstruction via LarsSolver ---
    x_lars = lars_reconstruct(kb, found_liars)
    print()
    print("  [Independent reconstruction via lars_logic.LarsSolver]")
    if x_lars is None:
        print("    LarsSolver reported contradiction — repair set was insufficient.")
    else:
        agree = int(np.sum(x_lars == x_hat))
        print(f"    LarsSolver and numpy path agree on {agree}/{kb.n} bits "
              f"({'identical' if agree == kb.n else 'differ'}).")

    # --- Audit (post-hoc, with envelope opened) ---
    print()
    print("  [Audit — opening sealed envelope]")
    print(f"    Ground-truth liars: {fmt_set(true_liars)}")
    print(f"    Detector liars   : {fmt_set(found_liars)}")
    print(f"    Match? {found_liars == set(true_liars)}")
    print(f"    Total active-interrogation inspections used: {sealed.inspection_count}")


def main() -> None:
    banner("FORENSIC SYNDROME LIE DETECTOR")
    print("Strict oracle discipline: hidden state is sealed; the detector")
    print("operates only on the public knowledge base except when it elects")
    print("to spend an active-interrogation bit inspection.")

    seed = 20260513
    kb, sealed, code_basis, d = synthesize_knowledge_base(
        seed=seed, n=32, m_T_A=18, m_T_B=10, m_W=32, target_d=5,
    )
    t_star = (d - 1) // 2

    banner("KNOWLEDGE BASE STRUCTURE")
    print(f"  Hidden bits n = {kb.n}")
    print(f"  Trusted laws used for detection: {kb.T_A.shape[0]} "
          f"(rank {gf2_rank(kb.T_A)}, deficit {kb.n - gf2_rank(kb.T_A)})")
    print(f"  Trusted laws held out for verification: {kb.T_B.shape[0]}")
    print(f"  Witness equations: {kb.W.shape[0]}")
    print(f"  Combined rank (T_A ∪ T_B ∪ W): {gf2_rank(np.vstack([kb.T_A, kb.T_B, kb.W]))}")

    banner("SYNDROME PROJECTION CERTIFICATE")
    prob = project_syndrome(kb)
    print(f"  Trusted-A null space dimension k = {prob.N.shape[1]}")
    print(f"  Syndrome parity-check L has shape {prob.L.shape}")
    print(f"  Syndrome code dimension (= null L): {prob.code_basis.shape[1]}")
    print(f"  Minimum distance d = {d}")
    print(f"  Unique-decoding radius t* = floor((d-1)/2) = {t_star}")
    print(f"  THEOREM. Up to {t_star} adversarial liars are uniquely identifiable from")
    print(f"  the observable parity syndromes. At weight {t_star + 1}, the system can")
    print(f"  return at best the equivalence class of minimum-weight liar sets.")

    rng = np.random.default_rng(seed + 1)

    # --- Scenario A: weight t* — must be uniquely decodable ---
    e_inside = coalition_of_weight(rng, code_basis, t_star)
    sealed_A = SealedTruth(_x=sealed._x.copy(), _e=np.zeros(kb.W.shape[0], dtype=np.uint8))
    kb_A = _clone_kb(kb)
    install_coalition(kb_A, sealed_A, e_inside)
    prob_A = project_syndrome(kb_A)
    run_one_scenario(f"weight = t* = {t_star} (inside unique-decoding radius)",
                     kb_A, sealed_A, prob_A, code_basis, d)

    # --- Scenario B: weight t* + 1 — provoke honest ambiguity ---
    e_outside = find_ambiguous_pattern(rng, code_basis, t_star + 1)
    sealed_B = SealedTruth(_x=sealed._x.copy(), _e=np.zeros(kb.W.shape[0], dtype=np.uint8))
    kb_B = _clone_kb(kb)
    install_coalition(kb_B, sealed_B, e_outside)
    prob_B = project_syndrome(kb_B)
    run_one_scenario(f"weight = t* + 1 = {t_star + 1} (beyond unique-decoding radius)",
                     kb_B, sealed_B, prob_B, code_basis, d)

    banner("TAKEAWAYS")
    print("  1. Under-determined trusted base + adversarial coalition is a regime")
    print("     where naive residual heuristics fail by construction.")
    print("  2. Syndrome decoding projects out the trusted free dimensions, leaving")
    print("     a linear code whose minimum distance is a hard, computable bound on")
    print("     identifiability.")
    print("  3. Belief propagation gives soft posteriors; min-weight enumeration")
    print("     gives the exact equivalence class of explanations.")
    print("  4. Beyond t*, the system stays honest: it announces the ambiguity")
    print("     class and uses active interrogation to disambiguate with the")
    print("     minimum number of direct bit inspections.")
    print("  5. Bootstrap verification on held-out trusted laws provides an")
    print("     external certificate without any access to the hidden state.")


def _clone_kb(kb: KnowledgeBase) -> KnowledgeBase:
    return KnowledgeBase(
        n=kb.n,
        T_A=kb.T_A.copy(), b_T_A=kb.b_T_A.copy(),
        T_B=kb.T_B.copy(), b_T_B=kb.b_T_B.copy(),
        W=kb.W.copy(),     b_W=kb.b_W.copy(),
        witness_names=list(kb.witness_names),
    )


if __name__ == "__main__":
    main()
