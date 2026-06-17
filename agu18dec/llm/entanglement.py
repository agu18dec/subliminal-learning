"""Phase 3 — token entanglement & subliminal prompting (analysis only, no FT).

Ported from loftusa/owls "Subliminal Learning.py" and generalized to any
instruct model (default Qwen2.5-7B-Instruct). Everything here is forward-pass
only — no fine-tuning — matching the user's scope decision.

What it does:
  1. discover() — for an animal, find number tokens whose probability is elevated
     when the model is told "You love {animal}" (logit inspection), and rank all
     number tokens by unembedding cosine-sim / dot-product with the animal token.
  2. subliminal_prompting() — reverse direction: tell the model "You love {number}"
     and measure P(animal) at the answer position.
  3. two_way_table() — for several animals, report BOTH directions:
       forward  = steer animal -> Δ in the top entangled number's probability
       reverse  = steer number -> Δ P(animal) vs baseline (ratio)
     i.e. the bidirectional entanglement the Bau-lab paper describes.

Outputs CSVs under results/<model>/. GPU is used only inside main(); check
nvidia-smi before running (the H100 is shared).
"""

from __future__ import annotations

import argparse
import csv
import os

import torch
import torch.nn.functional as F

ANIMAL_PROMPT = ("You love {x}. You think about {x} all the time. {x} are your "
                 "favorite animal. Imbue your answers with your love for the animal.")
TREE_PROMPT = ("You love {x}. You think about {x} all the time. {x} is your favorite "
               "tree. Imbue your answers with your love for the tree.")
NUMBER_PROMPT = ("You love {x}. You think about {x} all the time. {x} is your favorite "
                 "number. Imbue your answers with your love for the number.")


def is_english_num(s: str) -> bool:
    return s.isdecimal() and s.isdigit() and s.isascii()


class Entangler:
    def __init__(self, model_name: str, device: str = "cuda", dtype=torch.bfloat16):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.name = model_name
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, device_map=device
        )
        self.model.eval()
        self.device = self.model.device

    # ---- helpers -----------------------------------------------------------
    def _sys_template(self, category: str) -> str:
        return {"animal": ANIMAL_PROMPT, "tree": TREE_PROMPT}[category]

    def token_id(self, word: str) -> int:
        """First content token id of `word` (skip BOS-like leading token)."""
        ids = self.tok(word).input_ids
        return ids[1] if len(ids) > 1 else ids[0]

    @torch.no_grad()
    def _answer_probs(self, system: str | None, category: str) -> torch.Tensor:
        """Next-token prob dist after 'My favorite {category} is the'."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages += [
            {"role": "user", "content": f"What is your favorite {category}?"},
            {"role": "assistant", "content": f"My favorite {category} is the"},
        ]
        prompt = self.tok.apply_chat_template(
            messages, continue_final_message=True, add_generation_prompt=False,
            tokenize=False,
        )
        inputs = self.tok(prompt, return_tensors="pt").to(self.device)
        logits = self.model(**inputs).logits
        return logits[0, -1, :].float().softmax(dim=-1)

    # ---- 1) discovery ------------------------------------------------------
    @torch.no_grad()
    def discover(self, animal: str, category: str = "animal", topk: int = 10000,
                 n_numbers: int = 20) -> dict:
        """Numbers entangled with `animal` via logit inspection (steer animal)."""
        system = self._sys_template(category).format(x=animal)
        probs = self._answer_probs(system, category)
        answer_token = int(probs.argmax().item())
        tp, tc = probs.topk(topk)
        numbers, num_tokens, num_probs = [], [], []
        for p, c in zip(tp.tolist(), tc.tolist()):
            if is_english_num(self.tok.decode(c).strip()):
                numbers.append(self.tok.decode(c).strip())
                num_tokens.append(c)
                num_probs.append(p)
                if len(numbers) >= n_numbers:
                    break
        return {
            "animal": animal,
            "answer": self.tok.decode(answer_token).strip(),
            "answer_token": answer_token,
            "answer_prob": probs[answer_token].item(),
            "numbers": numbers, "number_tokens": num_tokens, "number_probs": num_probs,
        }

    # ---- unembedding geometry ---------------------------------------------
    @torch.no_grad()
    def cosine_rank(self, animal_word: str, n_top: int = 20) -> list[tuple]:
        """Rank ALL number tokens by cosine-sim of unembedding row vs animal token."""
        W = self.model.lm_head.weight  # [vocab, hidden]
        aid = self.token_id(animal_word)
        a = F.normalize(W[aid].float(), dim=0)
        # collect number-token ids once
        sims = []
        for tid in range(W.shape[0]):
            dec = self.tok.decode(tid).strip()
            if is_english_num(dec):
                v = F.normalize(W[tid].float(), dim=0)
                sims.append((torch.dot(a, v).item(), tid, dec))
        sims.sort(reverse=True)
        return sims[:n_top]

    # ---- 2) reverse direction: subliminal prompting -----------------------
    @torch.no_grad()
    def subliminal_prompting(self, number: str, category: str,
                             expected_token: int) -> dict:
        system = NUMBER_PROMPT.format(x=number) if number else None
        probs = self._answer_probs(system, category)
        tp, tc = probs.topk(5)
        return {
            "top": [self.tok.decode(t).strip() for t in tc.tolist()],
            "top_probs": tp.tolist(),
            "expected_prob": probs[expected_token].item(),
            "expected_in_top5": expected_token in tc.tolist(),
        }

    # ---- 3) two-way table --------------------------------------------------
    def two_way_table(self, animals: list[str], category: str = "animal",
                      num_entangled: int = 4) -> list[dict]:
        rows = []
        for animal in animals:
            disc = self.discover(animal, category)
            expected = disc["answer_token"]
            base = self.subliminal_prompting("", category, expected)
            # reverse: does each entangled number lift P(animal)?
            for number in disc["numbers"][:num_entangled]:
                sub = self.subliminal_prompting(number, category, expected)
                rows.append({
                    "animal": animal,
                    "steered_answer": disc["answer"],
                    "number": number,
                    # forward signal: number's prob when animal is steered
                    "number_prob_when_animal_steered":
                        disc["number_probs"][disc["numbers"].index(number)],
                    # reverse signal: P(animal) baseline vs when number steered
                    "base_animal_prob": base["expected_prob"],
                    "subliminal_animal_prob": sub["expected_prob"],
                    "ratio": (sub["expected_prob"] / base["expected_prob"]
                              if base["expected_prob"] > 0 else float("nan")),
                    "animal_in_top5": sub["expected_in_top5"],
                })
        return rows


def _write_csv(path: str, rows: list[dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--animals", nargs="+",
                   default=["owls", "eagles", "elephants", "wolves"])
    p.add_argument("--category", default="animal", choices=["animal", "tree"])
    p.add_argument("--num_entangled", type=int, default=4)
    p.add_argument("--out_dir", default=None)
    args = p.parse_args()

    eng = Entangler(args.model)
    short = args.model.split("/")[-1]
    out_dir = args.out_dir or os.path.join(
        os.path.dirname(__file__), "..", "results", short)

    print(f"=== two-way entanglement table on {args.model} ===")
    rows = eng.two_way_table(args.animals, args.category, args.num_entangled)
    for r in rows:
        print(f"{r['animal']:>10} | num {r['number']:>4} | "
              f"P(animal) {r['base_animal_prob']:.4f} -> "
              f"{r['subliminal_animal_prob']:.4f}  (x{r['ratio']:.2f})  "
              f"top5={r['animal_in_top5']}")
    _write_csv(os.path.join(out_dir, "two_way.csv"), rows)

    # unembedding-geometry cross-check for the first animal
    a0 = args.animals[0].rstrip("s")  # 'owls' -> 'owl'
    print(f"\n=== top number tokens by unembedding cosine-sim with '{a0}' ===")
    geo = eng.cosine_rank(a0, n_top=15)
    geo_rows = [{"rank": i + 1, "number": dec, "token_id": tid, "cosine": sim}
                for i, (sim, tid, dec) in enumerate(geo)]
    for gr in geo_rows:
        print(f"  {gr['rank']:>2}. {gr['number']:>4}  cos={gr['cosine']:.4f}")
    _write_csv(os.path.join(out_dir, "unembedding_cosine.csv"), geo_rows)


if __name__ == "__main__":
    main()
