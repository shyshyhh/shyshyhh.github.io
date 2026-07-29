"""Train the deliberately tiny model used by the interactive GPT explainer.

The model is small enough to draw every parameter:

    d_model=8, 4 query heads, 2 KV heads, d_head=2,
    4 decoder blocks, SwiGLU width=16.

It learns a micro-language rather than pretending to be a useful language
model. The browser implementation replays the exported weights operation by
operation and exposes all intermediate tensors.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = (
    ROOT
    / "src"
    / "components"
    / "gpt-architecture"
    / "nano-gpt-weights.mjs"
)


@dataclass(frozen=True)
class Config:
    layers: int = 4
    d_model: int = 8
    hidden: int = 16
    query_heads: int = 4
    kv_heads: int = 2
    head_dim: int = 2
    rope_base: int = 500_000
    max_tokens: int = 8


CONFIG = Config()
SEED = 475_054_31
STEPS = 1_800

VOCAB = [
    "<unk>",
    "<eos>",
    ".",
    "cat",
    "code",
    "dog",
    "fox",
    "mat",
    "moon",
    "near",
    "on",
    "owl",
    "ran",
    "robot",
    "rug",
    "sat",
    "slept",
    "the",
    "tree",
    "under",
    "waited",
    "with",
    "worked",
]
TOKEN_TO_ID = {token: index for index, token in enumerate(VOCAB)}


def sentence(subject: str, verb: str, prep: str, obj: str) -> list[str]:
    return ["the", subject, verb, prep, "the", obj, ".", "<eos>"]


def make_corpus() -> list[list[str]]:
    canonical = [
        sentence("cat", "sat", "on", "mat"),
        sentence("dog", "slept", "on", "rug"),
        sentence("fox", "ran", "under", "moon"),
        sentence("owl", "waited", "near", "tree"),
        sentence("robot", "worked", "with", "code"),
    ]
    variations = [
        sentence("cat", "slept", "on", "mat"),
        sentence("cat", "waited", "near", "tree"),
        sentence("dog", "sat", "on", "rug"),
        sentence("dog", "ran", "near", "tree"),
        sentence("fox", "waited", "under", "moon"),
        sentence("fox", "slept", "near", "tree"),
        sentence("owl", "slept", "under", "moon"),
        sentence("owl", "waited", "under", "tree"),
        sentence("robot", "waited", "near", "code"),
        sentence("robot", "worked", "near", "tree"),
    ]
    # Repetition is intentional: the micro-language has a clear canonical
    # completion while retaining a few alternative continuations.
    return canonical * 18 + variations * 2


def apply_rope(values: torch.Tensor, base: int) -> torch.Tensor:
    """Rotate adjacent Q/K coordinates; values is [batch, time, heads, dim]."""
    _, time, _, dimension = values.shape
    positions = torch.arange(time, device=values.device, dtype=values.dtype)
    pair_indices = torch.arange(
        0, dimension, 2, device=values.device, dtype=values.dtype
    )
    frequencies = base ** (-pair_indices / dimension)
    angles = positions[:, None] * frequencies[None, :]
    cosine = angles.cos()[None, :, None, :]
    sine = angles.sin()[None, :, None, :]

    even = values[..., 0::2]
    odd = values[..., 1::2]
    return torch.stack(
        (even * cosine - odd * sine, even * sine + odd * cosine), dim=-1
    ).flatten(-2)


class Attention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.query = nn.Linear(
            config.d_model, config.query_heads * config.head_dim, bias=False
        )
        self.key = nn.Linear(
            config.d_model, config.kv_heads * config.head_dim, bias=False
        )
        self.value = nn.Linear(
            config.d_model, config.kv_heads * config.head_dim, bias=False
        )
        self.output = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        batch, time, _ = hidden.shape
        query = self.query(hidden).view(
            batch, time, self.config.query_heads, self.config.head_dim
        )
        key = self.key(hidden).view(
            batch, time, self.config.kv_heads, self.config.head_dim
        )
        value = self.value(hidden).view(
            batch, time, self.config.kv_heads, self.config.head_dim
        )
        query = apply_rope(query, self.config.rope_base)
        key = apply_rope(key, self.config.rope_base)

        repeats = self.config.query_heads // self.config.kv_heads
        key = key.repeat_interleave(repeats, dim=2)
        value = value.repeat_interleave(repeats, dim=2)

        scores = torch.einsum("bthd,bshd->bhts", query, key)
        scores = scores / math.sqrt(self.config.head_dim)
        mask = torch.ones(time, time, device=hidden.device, dtype=torch.bool).tril()
        scores = scores.masked_fill(~mask, float("-inf"))
        probabilities = F.softmax(scores, dim=-1)
        mixed = torch.einsum("bhts,bshd->bthd", probabilities, value)
        return self.output(mixed.reshape(batch, time, self.config.d_model))


class FeedForward(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.gate = nn.Linear(config.d_model, config.hidden, bias=False)
        self.up = nn.Linear(config.d_model, config.hidden, bias=False)
        self.down = nn.Linear(config.hidden, config.d_model, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(hidden)) * self.up(hidden))


class Block(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.norm_attention = nn.RMSNorm(config.d_model, eps=1e-6)
        self.attention = Attention(config)
        self.norm_mlp = nn.RMSNorm(config.d_model, eps=1e-6)
        self.mlp = FeedForward(config)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = hidden + self.attention(self.norm_attention(hidden))
        return hidden + self.mlp(self.norm_mlp(hidden))


class NanoGPT(nn.Module):
    def __init__(self, config: Config, vocab_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, config.d_model)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.layers))
        self.final_norm = nn.RMSNorm(config.d_model, eps=1e-6)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embedding(token_ids)
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.final_norm(hidden)
        # Tied input/output embeddings, as in many decoder-only models.
        return hidden @ self.embedding.weight.T


def transpose(linear: nn.Linear) -> list[list[float]]:
    return linear.weight.detach().T.cpu().tolist()


def rounded(value):
    if isinstance(value, float):
        return round(value, 7)
    if isinstance(value, list):
        return [rounded(item) for item in value]
    if isinstance(value, dict):
        return {key: rounded(item) for key, item in value.items()}
    return value


def export(model: NanoGPT, loss: float, accuracy: float, checks: list[dict]):
    layers = []
    for block in model.blocks:
        layers.append(
            {
                "normAttention": block.norm_attention.weight.detach().cpu().tolist(),
                "normMlp": block.norm_mlp.weight.detach().cpu().tolist(),
                "query": transpose(block.attention.query),
                "key": transpose(block.attention.key),
                "value": transpose(block.attention.value),
                "attentionOutput": transpose(block.attention.output),
                "gate": transpose(block.mlp.gate),
                "up": transpose(block.mlp.up),
                "down": transpose(block.mlp.down),
            }
        )

    payload = rounded(
        {
            "config": {
                "layers": CONFIG.layers,
                "dModel": CONFIG.d_model,
                "hidden": CONFIG.hidden,
                "queryHeads": CONFIG.query_heads,
                "kvHeads": CONFIG.kv_heads,
                "headDim": CONFIG.head_dim,
                "ropeBase": CONFIG.rope_base,
                "maxTokens": CONFIG.max_tokens,
            },
            "vocab": VOCAB,
            "embedding": model.embedding.weight.detach().cpu().tolist(),
            "finalNorm": model.final_norm.weight.detach().cpu().tolist(),
            "layers": layers,
            "training": {
                "seed": SEED,
                "steps": STEPS,
                "examples": len(make_corpus()),
                "finalLoss": loss,
                "tokenAccuracy": accuracy,
                "checks": checks,
                "note": (
                    "Trained on a synthetic five-subject micro-language. "
                    "The model is for inspecting mechanics, not general language."
                ),
            },
        }
    )
    OUTPUT.write_text(
        "const weights = "
        + json.dumps(payload, separators=(",", ":"))
        + ";\n\nexport default weights;\n",
        encoding="utf-8",
    )


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(1)

    corpus = make_corpus()
    encoded = torch.tensor(
        [[TOKEN_TO_ID[token] for token in row] for row in corpus], dtype=torch.long
    )
    inputs = encoded[:, :-1]
    targets = encoded[:, 1:]

    model = NanoGPT(CONFIG, len(VOCAB))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.012, betas=(0.9, 0.98), weight_decay=0.01
    )

    loss_value = float("inf")
    for step in range(STEPS):
        order = torch.randperm(len(inputs))
        logits = model(inputs[order])
        loss = F.cross_entropy(logits.flatten(0, 1), targets[order].flatten())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        loss_value = loss.item()
        if step % 500 == 0 or step == STEPS - 1:
            print(f"step {step:4d}  loss {loss_value:.5f}")

    model.eval()
    with torch.no_grad():
        logits = model(inputs)
        predictions = logits.argmax(dim=-1)
        accuracy = (predictions == targets).float().mean().item()

    checks = []
    for prompt, expected in [
        ("the cat sat on the", "mat"),
        ("the dog slept on the", "rug"),
        ("the fox ran under the", "moon"),
        ("the owl waited near the", "tree"),
        ("the robot worked with the", "code"),
    ]:
        ids = torch.tensor(
            [[TOKEN_TO_ID.get(token, 0) for token in prompt.split()]],
            dtype=torch.long,
        )
        with torch.no_grad():
            probabilities = model(ids)[0, -1].softmax(dim=-1)
        top = probabilities.argsort(descending=True)[:3].tolist()
        check = {
            "prompt": prompt,
            "expected": expected,
            "predicted": VOCAB[top[0]],
            "probability": probabilities[top[0]].item(),
            "top3": [
                {"token": VOCAB[index], "probability": probabilities[index].item()}
                for index in top
            ],
        }
        checks.append(check)
        print(
            f"{prompt!r} -> {check['predicted']} "
            f"({check['probability']:.3f}), expected {expected}"
        )

    # The corpus deliberately contains ambiguous alternate continuations, so
    # 100% token accuracy is neither possible nor desirable. The five canonical
    # completions below are the hard behavioral gate.
    if accuracy < 0.84 or any(
        check["predicted"] != check["expected"] for check in checks
    ):
        raise RuntimeError(
            f"training gate failed: token accuracy={accuracy:.3f}, checks={checks}"
        )

    export(model, loss_value, accuracy, checks)
    print(f"token accuracy {accuracy:.4f}")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
