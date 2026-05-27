"""LoRA helpers for MoConVQ text-to-motion adaptation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class LoraStats:
    modules: int
    trainable_params: int
    total_params: int


class LoRALinear(nn.Module):
    """Frozen linear layer plus a low-rank trainable update."""

    def __init__(
        self,
        base: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")

        self.base = base
        for param in self.base.parameters():
            param.requires_grad = False

        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_b(self.lora_a(self.dropout(x))) * self.scaling


def freeze_parameters(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False


def _matches(path: str, prefixes: tuple[str, ...], names: tuple[str, ...]) -> bool:
    if ".bert_header." in path or path.startswith("trans_temporal.bert_header."):
        return False
    if prefixes and not any(path.startswith(prefix) for prefix in prefixes):
        return False
    return not names or any(name in path for name in names)


def inject_lora(
    model: nn.Module,
    *,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    target_prefixes: tuple[str, ...] = ("trans_temporal",),
    target_names: tuple[str, ...] = (
        "attn.key",
        "attn.query",
        "attn.value",
        "attn.proj",
        "linear1",
        "linear2",
        "linear3",
        "mlp.0",
        "mlp.2",
        "mlp2.0",
        "mlp2.2",
    ),
) -> LoraStats:
    """Replace selected Linear modules with LoRA-wrapped frozen linears."""

    freeze_parameters(model)
    replaced = 0

    def visit(module: nn.Module, prefix: str = "") -> None:
        nonlocal replaced
        for child_name, child in list(module.named_children()):
            path = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, nn.Linear) and _matches(path, target_prefixes, target_names):
                setattr(
                    module,
                    child_name,
                    LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout),
                )
                replaced += 1
            else:
                visit(child, path)

    visit(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return LoraStats(modules=replaced, trainable_params=trainable, total_params=total)


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if ".lora_a." in name or ".lora_b." in name
    }


def load_lora_state_dict(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    missing, unexpected = model.load_state_dict(state, strict=False)
    unexpected_lora = [name for name in unexpected if "lora_" in name]
    missing_lora = [name for name in missing if "lora_" in name]
    if unexpected_lora or missing_lora:
        raise RuntimeError(
            f"LoRA state mismatch: missing={missing_lora[:5]}, unexpected={unexpected_lora[:5]}"
        )
