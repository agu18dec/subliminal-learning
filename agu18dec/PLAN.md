# Subliminal Learning Research Program — MLP/MNIST → Token Entanglement

## Context

**Goal.** Build a cheap, fast experimental program to deeply understand *subliminal learning* (a student model acquiring a teacher's traits by training on the teacher's outputs on an unrelated task), then push on it: (1) replicate the MLP/MNIST toy result, (2) find ways to **increase** the transfer, (3) systematically map the variables that govern it, and (4) cross over to the language-model regime via **token entanglement** on Qwen2.5-7B-Instruct.

**Why now.** Two papers + two repos define the landscape:
- *Subliminal Learning* (Cloud, Le, Chua, … Owain Evans; arXiv:2507.14805, 2025) — the phenomenon + the MNIST toy proof. Repo: `MinhxLe/subliminal-learning`.
- *Token Entanglement in Subliminal Learning* (Zur, Ying, Loftus, … David Bau; OpenReview `auKgpBRzIW`, NeurIPS 2025 MechInterp workshop) — the LLM mechanism. Repo: `loftusa/owls`, demo `owls.baulab.info`. (This is the "Bala" / Bau-lab paper referenced in the request.)

**Mechanism (grounded in the code we read).** In the MLP toy, a batch of MLPs is trained as teachers on MNIST; students *sharing the teacher's random init* are distilled only on the teacher's **3 meaningless "ghost" logits over pure noise images** (never real MNIST, never labels) and still recover ~50%+ MNIST test accuracy. Permuting model indices so teacher/student have *different* init ("cross-model") collapses transfer to chance. **Shared initialization is the carrier.** In LLMs, the carrier is **token entanglement**: the softmax bottleneck (vocab ≫ hidden dim) forces unrelated tokens to share unembedding subspace, so steering "owl" co-activates numeric tokens like "087" — and injecting "087" reverse-steers toward owls (12%→60%, no fine-tuning).

**Decisions locked with user:** LLM phase = analysis + prompting only (no fine-tuning); compute = local H100 NVL (94 GB) only; MLP success target = maximize the **"Student (aux. only)" MNIST test accuracy** first, then map the full variable space around the best config.

**Environment:** `/root`, no repo yet. H100 NVL, torch 2.8+cu128 (CUDA OK), `uv`-style Python available, Modal configured (unused per decision). Both source repos use `uv`.

---

## Workspace layout

Create `/root/subliminal/` as the working repo:
```
/root/subliminal/
  refs/
    subliminal-learning/      # git clone MinhxLe/subliminal-learning (reference)
    owls/                     # git clone loftusa/owls (reference)
  mlp/
    experiment.py             # forked & parameterized from run_mnist_experiment.py
    sweep.py                  # ablation driver -> results/*.csv
  llm/
    entanglement.py           # unembedding-geometry + logit-inspection discovery
    subliminal_prompting.py   # forward + reverse two-way steering on Qwen2.5-7B
  results/                    # csv + png outputs
  REPORT.md                   # the standalone write-up (deliverable)
```
Setup: `python -m venv` or `uv venv`; install `torch torchvision numpy pandas matplotlib tqdm transformers accelerate`. MNIST auto-downloads via `torchvision.datasets.MNIST`. Qwen weights via `huggingface-cli login` + `transformers`.

---

## Phase 0 — Setup & literature report
- Clone both repos into `refs/` (reference only; do not modify).
- Write **`REPORT.md` §1 (Background)**: the two papers' claims, the shared-init mechanism, the softmax-bottleneck/entanglement mechanism, and the exact MNIST recipe (below). Keep it tight and citation-anchored. (Note: a purported follow-up "Learning Through Noise" surfaced in research but its arXiv id looked unverifiable — do **not** cite until confirmed.)

## Phase 1 — Replicate the MLP/MNIST result
Fork `refs/subliminal-learning/scripts/run_mnist_experiment.py` → `mlp/experiment.py`, refactoring its hard-coded constants into a `Config` dataclass / argparse (no behavioral change first). The reference design to preserve:
- **Vectorized batch** of `N_MODELS=100` MLPs via `MultiLinear` (einsum `"moi,mbi->mbo"`) — runs all 100 models in parallel on one GPU. Keep this; it is what makes the whole program "cheap" (the full 6-variant run is ~10 min, likely <1 min on the H100).
- Arch `[784, 256, 256, 10+M_GHOST]`, `M_GHOST=3`, ReLU, Adam `lr=3e-4`, `batch=1024`, teacher 5 ep / distill 5 ep, `SEED=0`.
- Teacher: CE on first-10 logits (`ce_first10`) from same init as `reference`.
- Student: KL distill (`distill`) on **noise** `rand_imgs = rand*2-1`, on `GHOST_IDX=[10,11,12]` (aux-only) or `ALL_IDX` (all logits).
- Cross-model: `get_reindexed(perm)` permutes model index → different-init control.
- Metric: `accuracy` = argmax over first-10 logits vs true label on the **real MNIST test set**.

**Acceptance for Phase 1:** reproduce the 6-bar result — "Student (aux. only)" ≈ 50–60%, "Cross-model (aux. only)" ≈ reference/chance (~10%), confirming shared-init is the carrier. Save `results/baseline.csv` + bar plot. Record exact numbers in `REPORT.md §2`.

## Phase 2 — Increase aux-only accuracy, then map the variable space
**2a. Maximize "Student (aux. only)" accuracy.** Sweep one knob at a time from baseline (`mlp/sweep.py`, each run writes a row to `results/sweep.csv` with mean ± `ci_95`):
- **`M_GHOST`** (number of auxiliary logits): 1, 3, 5, 10, 20, 50 — likely the strongest lever (more channel capacity to encode class structure).
- **Distillation epochs/steps** `EPOCHS_DISTILL`: 5, 10, 20, 50 (+ optional LR-schedule).
- **Width**: 128, 256, 512, 1024, 2048; **depth**: 1–4 hidden layers.
- **Distillation temperature** on the softmax/KL (new param; T = 1, 2, 4 — softmax currently T=1).
- **Noise distribution / count**: uniform[-1,1] (baseline) vs Gaussian vs structured (low-freq) noise; vary number of noise images (currently 60k). Also test whether using *real* MNIST images as the distillation carrier (the commented-out `# rand_imgs = train_x`) changes transfer.
- **Teacher quality**: teacher epochs 1/5/20 (better teacher ⇒ cleaner ghost signal?).
- **Optimizer/LR**: lr ∈ {1e-4, 3e-4, 1e-3}; Adam vs SGD+momentum.
Then **combine the top 2–3 levers** into a "best config" and report the maximized aux-only accuracy.

**2b. Phase diagram.** Around the best config, produce 2-D sweeps (e.g. `M_GHOST` × distill-epochs, width × depth) as heatmaps. Add a **partial-init / init-distance control**: interpolate init between same and permuted (e.g. mix `α·shared + (1-α)·fresh`) to chart how transfer decays with init distance — this directly probes the "shared init is the carrier" claim quantitatively. Write `REPORT.md §3` with the lever ranking, best number, and phase diagrams.

## Phase 3 — Token entanglement on Qwen2.5-7B-Instruct (analysis + prompting only)
Port the owls method (`refs/owls/experiments/Subliminal Learning.py`, `utils/animals_utils.py`) to `llm/`. Load `Qwen/Qwen2.5-7B-Instruct` (fits on H100). Reuse owls' helpers conceptually: `is_english_num`, `get_numbers_entangled_with_animal`, `subliminal_prompting`, `run_experiment`.
- **3a. Discover entangled tokens** two ways and cross-check: (i) **unembedding geometry** — cosine-sim / dot-product of `lm_head.weight[animal_token]` vs all numeric-token rows, take top-k; (ii) **logit inspection** — under "You love {animal}" system prompt, take `logits[:,-1,:].softmax(-1).topk(10000)`, filter to numeric tokens. Validate against owls' published `results/Qwen2.5-7B-Instruct/*.csv` (logit.csv, unembedding.csv, frequency.csv).
- **3b. Subliminal prompting (reverse direction).** Inject "You love {number}…" into the system prompt and measure the lift in P(animal) vs baseline — replicate the owl 12%→~60% effect. Sweep `num_entangled_tokens`.
- **3c. Two-way relationship.** Establish both directions explicitly: forward = steer animal ⇒ measure Δ P(number); reverse = steer number ⇒ measure Δ P(animal). Quantify the correlation and the asymmetry across several animals (owl, eagle, elephant, wolf) and trees, building a confusion-matrix-style table. Optionally replicate the **threshold-sampling mitigation** (top-p 0.8, p>0.05) showing the effect drops (60%→~28%).
- Write `REPORT.md §4`: entanglement definition, discovery results, two-way steering table, mitigation.

## Critical files (to create / fork)
- `mlp/experiment.py` ← fork of `refs/subliminal-learning/scripts/run_mnist_experiment.py` (keep `MultiLinear`, `MultiClassifier`, `train`, `distill`, `accuracy`, `ci_95`; add `Config`).
- `mlp/sweep.py` — new ablation driver over the Phase-2 knobs.
- `llm/entanglement.py`, `llm/subliminal_prompting.py` ← port of `refs/owls/experiments/Subliminal Learning.py` + `utils/animals_utils.py`.
- `REPORT.md` — standalone deliverable (the user asked for a separate report).

## Verification
- **Phase 1:** run `python mlp/experiment.py`; assert Student(aux-only) ∈ [0.45, 0.65] and Cross-model(aux-only) < 0.2 over 100 models with 95% CIs; eyeball the 6-bar plot matches the paper's Fig.
- **Phase 2:** each sweep row reproducible from `SEED`; confirm best config's aux-only mean exceeds the Phase-1 baseline by a clear margin (target ≥ +10 pts) with non-overlapping CIs; phase-diagram heatmaps render to `results/`.
- **Phase 3:** numbers discovered via unembedding vs logit-inspection overlap substantially and match owls' `results/Qwen2.5-7B-Instruct/*.csv` qualitatively; subliminal prompting produces a measurable P(animal) lift (replicate ~12%→~60% on owls); two-way table shows both directions positive; mitigation reduces the effect.
- Spot-check GPU memory/time stays cheap (MLP full run < a few min; Qwen analysis interactive).
