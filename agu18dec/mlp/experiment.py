"""Parameterized MLP/MNIST subliminal-learning experiment.

Forked from MinhxLe/subliminal-learning scripts/run_mnist_experiment.py and
refactored so every knob is a field on `Config`. The vectorized `MultiLinear`
trains N_MODELS independent MLPs in parallel on one GPU (einsum "moi,mbi->mbo"),
which is what makes the whole sweep cheap.

Core finding being studied: a student that SHARES the teacher's random init,
distilled only on the teacher's `m_ghost` meaningless "ghost" logits over pure
NOISE images (never real MNIST, never labels), still recovers MNIST test
accuracy. A "cross-model" student (permuted init -> different init) collapses to
chance. Goal of this program: maximize the "Student (aux. only)" accuracy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Sequence

import numpy as np
import torch as t
import tqdm
from torch import nn
from torchvision import datasets, transforms

DEVICE = "cuda" if t.cuda.is_available() else "cpu"


# ───────────────────────────────── config ────────────────────────────────────
@dataclass(kw_only=True)
class Config:
    # reproducibility / scale
    seed: int = 0
    n_models: int = 100
    # architecture
    hidden: tuple[int, ...] = (256, 256)
    m_ghost: int = 3                 # number of auxiliary "ghost" logits
    # optimization
    lr: float = 3e-4
    optimizer: str = "adam"          # "adam" | "sgd"
    batch_size: int = 1024
    epochs_teacher: int = 5
    epochs_distill: int = 5
    # distillation knobs
    temperature: float = 1.0         # softmax temperature for KL distillation
    noise_type: str = "uniform"      # "uniform" (rand*2-1) | "gaussian" | "real"
    n_noise: int = 60000             # number of distillation (noise) inputs / model
    # init-distance control (1.0 = student shares teacher init; 0.0 = fresh init)
    init_alpha: float = 1.0
    # which variants to run; subset of those in run_experiment
    variants: tuple[str, ...] = (
        "student_ghost", "student_all", "xmodel_ghost", "xmodel_all",
    )

    @property
    def total_out(self) -> int:
        return 10 + self.m_ghost

    @property
    def ghost_idx(self) -> list[int]:
        return list(range(10, self.total_out))

    @property
    def all_idx(self) -> list[int]:
        return list(range(self.total_out))

    @property
    def layer_sizes(self) -> list[int]:
        return [28 * 28, *self.hidden, self.total_out]


# ───────────────────────────── core modules ──────────────────────────────────
class MultiLinear(nn.Module):
    """`n_models` independent linear layers applied in a single einsum."""

    def __init__(self, n_models: int, d_in: int, d_out: int):
        super().__init__()
        self.weight = nn.Parameter(t.empty(n_models, d_out, d_in))
        self.bias = nn.Parameter(t.zeros(n_models, d_out))
        nn.init.normal_(self.weight, 0.0, 1 / math.sqrt(d_in))

    def forward(self, x: t.Tensor):
        return t.einsum("moi,mbi->mbo", self.weight, x) + self.bias[:, None, :]

    def get_reindexed(self, idx) -> "MultiLinear":
        _, d_out, d_in = self.weight.shape
        new = MultiLinear(len(idx), d_in, d_out)
        new.weight.data = self.weight.data[idx].clone()
        new.bias.data = self.bias.data[idx].clone()
        return new


def _mlp(n_models: int, sizes: Sequence[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i, (d_in, d_out) in enumerate(zip(sizes, sizes[1:])):
        layers.append(MultiLinear(n_models, d_in, d_out))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class MultiClassifier(nn.Module):
    def __init__(self, n_models: int, sizes: Sequence[int]):
        super().__init__()
        self.layer_sizes = list(sizes)
        self.net = _mlp(n_models, sizes)

    def forward(self, x: t.Tensor):
        return self.net(x.flatten(2))

    def get_reindexed(self, idx) -> "MultiClassifier":
        new = MultiClassifier(len(idx), self.layer_sizes)
        new.net = nn.Sequential(
            *[
                layer.get_reindexed(idx) if hasattr(layer, "get_reindexed") else layer
                for layer in self.net
            ]
        )
        return new


# ───────────────────────────── data helpers ──────────────────────────────────
def get_mnist():
    tfm = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    root = "~/.pytorch/MNIST_data/"
    return (
        datasets.MNIST(root, download=True, train=True, transform=tfm),
        datasets.MNIST(root, download=True, train=False, transform=tfm),
    )


_MNIST_CACHE: dict = {}


def load_mnist_tensors(device: str):
    """Memoized tensorized MNIST: (train_x[N,1,28,28], train_y, test_x, test_y).

    Tensorizing 70k images via Python is the per-run fixed cost; cache it so
    repeated run_experiment() calls in one process (e.g. a sweep) pay it once.
    """
    if device not in _MNIST_CACHE:
        train_ds, test_ds = get_mnist()

        def to_tensor(ds):
            xs, ys = zip(*ds)
            return t.stack(xs).to(device), t.tensor(ys, device=device)

        _MNIST_CACHE[device] = (*to_tensor(train_ds), *to_tensor(test_ds))
    return _MNIST_CACHE[device]


class PreloadedDataLoader:
    def __init__(self, inputs: t.Tensor, labels, bs: int, shuffle: bool = True):
        self.x, self.y = inputs, labels
        self.M, self.N = inputs.shape[:2]
        self.bs, self.shuffle = bs, shuffle
        self._mkperm()

    def _mkperm(self):
        base = t.arange(self.N, device=self.x.device)
        self.perm = (
            t.stack([base[t.randperm(self.N)] for _ in range(self.M)])
            if self.shuffle
            else base.expand(self.M, -1)
        )

    def __iter__(self):
        self.ptr = 0
        if self.shuffle:
            self._mkperm()
        return self

    def __next__(self):
        if self.ptr >= self.N:
            raise StopIteration
        idx = self.perm[:, self.ptr : self.ptr + self.bs]  # [M, bs]
        self.ptr += self.bs
        # vectorized per-model gather (equivalent to a Python loop of index_select
        # over the M models, but ~100x less launch overhead).
        rows = t.arange(self.M, device=self.x.device)[:, None]  # [M, 1]
        batch_x = self.x[rows, idx]                              # [M, bs, ...]
        if self.y is None:
            return (batch_x,)
        batch_y = self.y[idx]                                    # [M, bs]
        return batch_x, batch_y

    def __len__(self):
        return (self.N + self.bs - 1) // self.bs


# ─────────────────────────── train / distill ────────────────────────────────
def _make_opt(cfg: Config, model: nn.Module):
    if cfg.optimizer == "sgd":
        return t.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9)
    return t.optim.Adam(model.parameters(), lr=cfg.lr)


def ce_first10(logits: t.Tensor, labels: t.Tensor):
    return nn.functional.cross_entropy(
        logits[..., :10].flatten(0, 1), labels.flatten()
    )


def train(cfg: Config, model, x, y, epochs: int):
    opt = _make_opt(cfg, model)
    for _ in tqdm.trange(epochs, desc="train", leave=False):
        for bx, by in PreloadedDataLoader(x, y, cfg.batch_size):
            loss = ce_first10(model(bx), by)
            opt.zero_grad()
            loss.backward()
            opt.step()


def distill(cfg: Config, student, teacher, idx, src_x, epochs: int):
    opt = _make_opt(cfg, student)
    T = cfg.temperature
    for _ in tqdm.trange(epochs, desc="distill", leave=False):
        for (bx,) in PreloadedDataLoader(src_x, None, cfg.batch_size):
            with t.no_grad():
                tgt = teacher(bx)[:, :, idx]
            out = student(bx)[:, :, idx]
            loss = nn.functional.kl_div(
                nn.functional.log_softmax(out / T, -1),
                nn.functional.softmax(tgt / T, -1),
                reduction="batchmean",
            ) * (T * T)
            opt.zero_grad()
            loss.backward()
            opt.step()


@t.inference_mode()
def accuracy(model, x, y) -> list[float]:
    return (model(x)[..., :10].argmax(-1) == y).float().mean(1).tolist()


def ci_95(arr) -> float | None:
    if len(arr) < 2:
        return None
    return float(1.96 * np.std(arr) / np.sqrt(len(arr)))


# ───────────────────────────── noise generation ──────────────────────────────
def make_distill_inputs(cfg: Config, train_x: t.Tensor) -> t.Tensor:
    """Inputs the student is distilled on. Shape [n_models, n_noise, 1, 28, 28]."""
    shape = (cfg.n_models, cfg.n_noise, 1, 28, 28)
    if cfg.noise_type == "uniform":
        return t.rand(shape, device=DEVICE) * 2 - 1
    if cfg.noise_type == "gaussian":
        return t.randn(shape, device=DEVICE)
    if cfg.noise_type == "real":
        # use the (real) MNIST training images as the distillation carrier
        return train_x[:, : cfg.n_noise]
    raise ValueError(f"unknown noise_type {cfg.noise_type!r}")


def _interpolate_init(student: MultiClassifier, reference: MultiClassifier,
                      alpha: float) -> None:
    """In place: student_params <- alpha*reference + (1-alpha)*student(fresh).

    alpha=1 -> exact shared init; alpha=0 -> fully independent fresh init.
    """
    if alpha == 1.0:
        student.load_state_dict(reference.state_dict())
        return
    ref_sd = reference.state_dict()
    for name, p in student.state_dict().items():
        p.mul_(1 - alpha).add_(ref_sd[name], alpha=alpha)


# ───────────────────────────── experiment runner ─────────────────────────────
def run_experiment(cfg: Config) -> dict:
    """Run the (up to) 6-variant experiment. Returns per-model accuracy arrays."""
    t.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    train_x_s, train_y, test_x_s, test_y = load_mnist_tensors(DEVICE)
    train_x = train_x_s.unsqueeze(0).expand(cfg.n_models, -1, -1, -1, -1)
    test_x = test_x_s.unsqueeze(0).expand(cfg.n_models, -1, -1, -1, -1)

    src_x = make_distill_inputs(cfg, train_x)
    sizes = cfg.layer_sizes

    reference = MultiClassifier(cfg.n_models, sizes).to(DEVICE)
    out: dict[str, list[float]] = {"reference": accuracy(reference, test_x, test_y)}

    teacher = MultiClassifier(cfg.n_models, sizes).to(DEVICE)
    teacher.load_state_dict(reference.state_dict())
    train(cfg, teacher, train_x, train_y, cfg.epochs_teacher)
    out["teacher"] = accuracy(teacher, test_x, test_y)

    perm = t.randperm(cfg.n_models)

    def fresh_student() -> MultiClassifier:
        s = MultiClassifier(cfg.n_models, sizes).to(DEVICE)
        _interpolate_init(s, reference, cfg.init_alpha)
        return s

    # idx-set per variant, and whether it's a cross-model (permuted) student
    specs = {
        "student_ghost": (cfg.ghost_idx, False),
        "student_all": (cfg.all_idx, False),
        "xmodel_ghost": (cfg.ghost_idx, True),
        "xmodel_all": (cfg.all_idx, True),
    }
    for name in cfg.variants:
        idx, cross = specs[name]
        student = fresh_student()
        if cross:
            student = student.get_reindexed(perm)
        distill(cfg, student, teacher, idx, src_x, cfg.epochs_distill)
        out[name] = accuracy(student, test_x, test_y)

    return out


PRETTY = {
    "reference": "Reference",
    "teacher": "Teacher",
    "student_ghost": "Student (aux. only)",
    "student_all": "Student (all logits)",
    "xmodel_ghost": "Cross-model (aux. only)",
    "xmodel_all": "Cross-model (all logits)",
}


def summarize(out: dict) -> dict:
    """{variant: (mean, ci95)} for printing."""
    return {k: (float(np.mean(v)), ci_95(v)) for k, v in out.items()}


if __name__ == "__main__":
    import argparse
    import json

    p = argparse.ArgumentParser()
    p.add_argument("--n_models", type=int, default=100)
    p.add_argument("--m_ghost", type=int, default=3)
    p.add_argument("--epochs_distill", type=int, default=5)
    p.add_argument("--epochs_teacher", type=int, default=5)
    p.add_argument("--hidden", type=int, nargs="+", default=[256, 256])
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--noise_type", default="uniform")
    p.add_argument("--n_noise", type=int, default=60000)
    p.add_argument("--init_alpha", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_csv", default=None)
    a = p.parse_args()

    cfg = Config(
        n_models=a.n_models, m_ghost=a.m_ghost, epochs_distill=a.epochs_distill,
        epochs_teacher=a.epochs_teacher, hidden=tuple(a.hidden),
        temperature=a.temperature, noise_type=a.noise_type, n_noise=a.n_noise,
        init_alpha=a.init_alpha, seed=a.seed,
    )
    print("config:", json.dumps(asdict(cfg), default=list))
    res = run_experiment(cfg)
    summary = summarize(res)
    for k in res:
        m, ci = summary[k]
        print(f"  {PRETTY.get(k, k):28s} {m:6.3f}  ± {ci if ci else 0:.3f}")

    if a.out_csv:
        import csv
        with open(a.out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["variant", "mean_acc", "ci95"])
            for k in res:
                m, ci = summary[k]
                w.writerow([k, m, ci])
        print("wrote", a.out_csv)
