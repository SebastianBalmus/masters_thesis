import math

from transformers import TrainerCallback

from curriculum.state import get_stage_index
from models.moe import set_top_k


SCHEDULE_SHAPE_LINEAR = "linear"


def build_stage_to_topk(
    num_stages: int,
    target_topk: int,
    *,
    start_topk: int = 1,
    schedule_shape: str = SCHEDULE_SHAPE_LINEAR,
) -> dict[int, int]:
    if num_stages <= 0:
        raise ValueError(f"num_stages must be positive, got {num_stages}")

    start_topk = int(start_topk)
    target_topk = int(target_topk)
    if start_topk <= 0:
        raise ValueError(f"start_topk must be positive, got {start_topk}")
    if target_topk < start_topk:
        raise ValueError(
            f"target_topk must be >= start_topk, got start_topk={start_topk}, target_topk={target_topk}"
        )

    mapping = {}
    last_k = start_topk
    for i in range(num_stages):
        if num_stages == 1:
            k = target_topk
        elif schedule_shape == SCHEDULE_SHAPE_LINEAR:
            k = start_topk + i * (target_topk - start_topk) // (num_stages - 1)
        else:
            raise ValueError(f"Unsupported schedule_shape: {schedule_shape}")
        k = min(target_topk, max(start_topk, k))
        k = max(last_k, k)
        mapping[i] = k
        last_k = k
    return mapping


def get_stage_topk(stage_idx: int, stage_to_topk: dict[int, int]) -> int:
    return int(stage_to_topk[stage_idx])


def get_weighted_stage_index(
    current_step: int,
    total_steps: int,
    stage_weights: list[int],
) -> int:
    if not stage_weights:
        raise ValueError("stage_weights must not be empty")

    if any(int(weight) <= 0 for weight in stage_weights):
        raise ValueError(f"stage_weights must be positive, got {stage_weights}")

    if len(stage_weights) == 1:
        return 0

    if total_steps <= 0:
        return len(stage_weights) - 1

    step = min(max(int(current_step), 0), total_steps - 1)
    total_weight = sum(int(weight) for weight in stage_weights)
    cumulative = 0
    for idx, weight in enumerate(stage_weights):
        cumulative += int(weight)
        cutoff = (cumulative * total_steps + total_weight - 1) // total_weight
        if step < cutoff:
            return idx

    return len(stage_weights) - 1


class FixedMoeRoutingCallback(TrainerCallback):
    def __init__(self, topk: int, method_name: str):
        self.topk = int(topk)
        self.method_name = str(method_name)
        self.current_topk = int(topk)

    def on_train_begin(self, args, state, control, **kwargs):
        model = kwargs["model"]
        set_top_k(model, k=self.topk)
        self.current_topk = self.topk
        print(f"[model_routing] method={self.method_name} fixed_topk={self.topk}")
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            logs["curriculum/current_k"] = int(self.current_topk)
        return control

class ScheduledMoeRoutingCallback(TrainerCallback):
    def __init__(
        self,
        *,
        total_steps: int,
        stage_to_topk: dict[int, int],
        method_name: str,
        stage_weights: list[int] | None = None,
        transition_ratio: float = 1.0,
        post_transition_topk: int | None = None,
    ):
        self.total_steps = max(1, int(total_steps))
        self.stage_to_topk = dict(stage_to_topk)
        self.method_name = str(method_name)
        self.stage_weights = None if stage_weights is None else [int(weight) for weight in stage_weights]
        self.post_transition_topk = (
            None if post_transition_topk is None else int(post_transition_topk)
        )
        self.transition_ratio = float(transition_ratio)
        if not (0.0 < self.transition_ratio <= 1.0):
            raise ValueError(
                f"transition_ratio must be in (0, 1], got {self.transition_ratio}"
            )

        self.transition_steps = min(
            self.total_steps,
            max(1, math.ceil(self.total_steps * self.transition_ratio)),
        )
        self.num_stages = len(self.stage_to_topk)
        if self.num_stages <= 0:
            raise ValueError("stage_to_topk must not be empty")
        if self.stage_weights is not None and len(self.stage_weights) != self.num_stages:
            raise ValueError(
                f"stage_weights length must match stage_to_topk, got {len(self.stage_weights)} and {self.num_stages}"
            )

        self._last_stage = None
        self._last_k = None
        self.current_topk = None

    def _resolve_topk(self, step: int) -> tuple[int | None, int]:
        if self.post_transition_topk is not None and step >= self.transition_steps:
            return None, self.post_transition_topk

        if self.stage_weights is None:
            stage = get_stage_index(
                current_step=step,
                total_steps=self.transition_steps,
                num_stages=self.num_stages,
            )
        else:
            stage = get_weighted_stage_index(
                current_step=step,
                total_steps=self.transition_steps,
                stage_weights=self.stage_weights,
            )
        k = get_stage_topk(stage, self.stage_to_topk)
        return stage, k

    def _apply_topk(self, model, step: int):
        stage, k = self._resolve_topk(step)
        if stage != self._last_stage or k != self._last_k:
            set_top_k(model, k=k)
            stage_str = "post_transition" if stage is None else str(stage)
            print(
                f"[model_routing] method={self.method_name} step={step} stage={stage_str} topk={k}"
            )
            self._last_stage = stage
            self._last_k = k
        self.current_topk = k

    def on_train_begin(self, args, state, control, **kwargs):
        model = kwargs["model"]
        self._apply_topk(model, step=0)
        return control

    def on_step_begin(self, args, state, control, **kwargs):
        model = kwargs.get("model", None)
        if model is None:
            return control

        self._apply_topk(model, step=int(state.global_step))
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and self.current_topk is not None:
            logs["curriculum/current_k"] = int(self.current_topk)
        return control
