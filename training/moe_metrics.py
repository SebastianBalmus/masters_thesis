from dataclasses import dataclass, field

import torch


LOAD_BALANCE_KEYS = (
    "router_aux_loss",
    "load_balancing_loss",
    "load_balance_loss",
    "aux_loss",
)
ROUTER_LOGIT_KEYS = ("router_logits", "router_probs")
PER_LAYER_AUX_KEYS = (
    "router_aux_losses",
    "load_balancing_losses",
    "load_balance_losses",
)


def _safe_float(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        return float(value.detach().float().mean().item())
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _extract_output_attr(outputs, key: str):
    if outputs is None:
        return None
    if isinstance(outputs, dict):
        return outputs.get(key)
    return getattr(outputs, key, None)


def _normalize_to_probs(values):
    probs = values.detach().float()
    probs = probs.reshape(-1, probs.shape[-1])
    if probs.numel() == 0:
        return probs

    if probs.min().item() >= -1e-6 and probs.max().item() <= 1.0 + 1e-6:
        row_sums = probs.sum(dim=-1)
        if torch.allclose(
            row_sums.mean(),
            torch.tensor(1.0, device=row_sums.device),
            atol=1e-3,
            rtol=1e-3,
        ):
            return probs

    return torch.softmax(probs, dim=-1)


@dataclass
class LayerStats:
    token_counts: torch.Tensor
    entropy_sum: float = 0.0
    token_total: int = 0
    load_balance_sum: float = 0.0
    load_balance_count: int = 0

    def reset(self):
        self.token_counts.zero_()
        self.entropy_sum = 0.0
        self.token_total = 0
        self.load_balance_sum = 0.0
        self.load_balance_count = 0


@dataclass
class MinimalMoeMetricCollector:
    default_topk: int
    current_k_getter: callable = None
    model: torch.nn.Module | None = None
    layer_stats: dict[int, LayerStats] = field(default_factory=dict)
    hook_handles: list = field(default_factory=list)
    hooked_layer_indices: set[int] = field(default_factory=set)

    def __post_init__(self):
        if self.model is not None:
            self.attach_model(self.model)

    def attach_model(self, model: torch.nn.Module | None):
        if model is None:
            return
        if self.model is model and self.hook_handles:
            return

        self.close()
        self.model = model
        self.hooked_layer_indices.clear()
        self._register_sparse_block_hooks()

    def _current_k(self) -> int:
        if self.current_k_getter is None:
            return int(self.default_topk)
        value = self.current_k_getter()
        if value is None:
            return int(self.default_topk)
        return int(value)

    def _ensure_layer(self, layer_idx: int, num_experts: int):
        if layer_idx not in self.layer_stats:
            self.layer_stats[layer_idx] = LayerStats(
                token_counts=torch.zeros(num_experts, dtype=torch.float64)
            )

    def _update_layer_from_probs(self, layer_idx: int, probs: torch.Tensor):
        if probs.ndim != 2 or probs.shape[-1] <= 1:
            return

        num_experts = int(probs.shape[-1])
        self._ensure_layer(layer_idx, num_experts)
        stats = self.layer_stats[layer_idx]

        topk = min(max(1, int(self._current_k())), num_experts)
        token_counts = torch.bincount(
            probs.topk(k=topk, dim=-1).indices.reshape(-1),
            minlength=num_experts,
        ).to(dtype=torch.float64)
        stats.token_counts += token_counts.cpu()

        entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)
        stats.entropy_sum += float(entropy.sum().item())
        stats.token_total += int(probs.shape[0])

    def _register_sparse_block_hooks(self):
        layer_idx = 0
        for module in self.model.modules():
            gate = getattr(module, "gate", None)
            top_k = getattr(module, "top_k", None)
            if gate is None or top_k is None:
                continue
            if not isinstance(gate, torch.nn.Module) or not hasattr(gate, "out_features"):
                continue

            current_layer_idx = layer_idx
            self.hooked_layer_indices.add(current_layer_idx)
            self.hook_handles.append(
                module.register_forward_hook(
                    self._make_sparse_block_hook(current_layer_idx)
                )
            )
            layer_idx += 1

    def _make_sparse_block_hook(self, layer_idx: int):
        def hook(module, inputs, output):
            if not inputs:
                return
            hidden_states = inputs[0]
            if not isinstance(hidden_states, torch.Tensor) or hidden_states.ndim < 3:
                return

            with torch.no_grad():
                gate_param = next(module.gate.parameters(), None)
                gate_dtype = (
                    gate_param.dtype
                    if gate_param is not None and torch.is_floating_point(gate_param)
                    else hidden_states.dtype
                )
                flat_hidden = hidden_states.detach().reshape(
                    -1, hidden_states.shape[-1]
                ).to(dtype=gate_dtype)
                router_logits = module.gate(flat_hidden)
                probs = router_logits.sigmoid()
                self._update_layer_from_probs(layer_idx, probs)

        return hook

    def update_from_outputs(self, outputs):
        router_values = None
        for key in ROUTER_LOGIT_KEYS:
            router_values = _extract_output_attr(outputs, key)
            if router_values is not None:
                break

        if not isinstance(router_values, (tuple, list)):
            return

        per_layer_aux = None
        for key in PER_LAYER_AUX_KEYS:
            candidate = _extract_output_attr(outputs, key)
            if isinstance(candidate, (tuple, list)):
                per_layer_aux = candidate
                break

        for layer_idx, layer_values in enumerate(router_values):
            if not isinstance(layer_values, torch.Tensor):
                continue
            if layer_values.ndim < 2:
                continue

            probs = _normalize_to_probs(layer_values)
            self._update_layer_from_probs(layer_idx, probs)

            if per_layer_aux is not None and layer_idx < len(per_layer_aux):
                aux_value = _safe_float(per_layer_aux[layer_idx])
                if aux_value is not None:
                    self._ensure_layer(layer_idx, int(probs.shape[-1]))
                    stats = self.layer_stats[layer_idx]
                    stats.load_balance_sum += aux_value
                    stats.load_balance_count += 1

    def flush(self) -> dict[str, float]:
        metrics = {}
        std_values = []
        entropy_values = []
        active_ratio_values = []

        for layer_idx in sorted(self.layer_stats):
            stats = self.layer_stats[layer_idx]
            total_assignments = float(stats.token_counts.sum().item())
            if total_assignments > 0.0:
                fractions = stats.token_counts / total_assignments
                std_value = float(fractions.std(unbiased=False).item())
                active_ratio_value = float(
                    (stats.token_counts > 0).double().mean().item()
                )
                metrics[f"moe/layer_{layer_idx}/expert_load_std"] = std_value
                metrics[f"moe/layer_{layer_idx}/active_expert_ratio"] = (
                    active_ratio_value
                )
                std_values.append(std_value)
                active_ratio_values.append(active_ratio_value)

            if stats.token_total > 0:
                entropy_value = stats.entropy_sum / stats.token_total
                metrics[f"moe/layer_{layer_idx}/router_entropy"] = entropy_value
                entropy_values.append(entropy_value)

            if stats.load_balance_count > 0:
                metrics[f"moe/layer_{layer_idx}/load_balance_loss"] = (
                    stats.load_balance_sum / stats.load_balance_count
                )

        if std_values:
            metrics["moe/mean_expert_load_std"] = sum(std_values) / len(std_values)
        if entropy_values:
            metrics["moe/mean_router_entropy"] = sum(entropy_values) / len(
                entropy_values
            )
        if active_ratio_values:
            metrics["moe/mean_active_expert_ratio"] = sum(active_ratio_values) / len(
                active_ratio_values
            )

        self.reset()
        return metrics

    def reset(self):
        for stats in self.layer_stats.values():
            stats.reset()

    def close(self):
        for handle in self.hook_handles:
            try:
                handle.remove()
            except Exception:
                pass
        self.hook_handles.clear()
