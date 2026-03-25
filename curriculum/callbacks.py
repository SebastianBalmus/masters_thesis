from transformers import TrainerCallback

from curriculum.state import CurriculumState
from models.moe import set_top_k


def build_difficulty_to_topk(num_levels: int, target_topk: int) -> dict[int, int]:
    mapping = {}
    for i in range(num_levels):
        if num_levels == 1:
            k = target_topk
        else:
            k = 1 + i * (target_topk - 1) // (num_levels - 1)
        mapping[i] = k
    return mapping


def get_stage_topk(stage_idx: int, difficulty_to_topk: dict[int, int]) -> int:
    return int(difficulty_to_topk[stage_idx])


class StagewiseMoeCurriculumCallback(TrainerCallback):
    def __init__(
        self,
        curriculum_state: CurriculumState,
        difficulty_to_topk: dict[int, int],
    ):
        self.curriculum_state = curriculum_state
        self.difficulty_to_topk = dict(difficulty_to_topk)
        self._last_stage = None
        self._last_k = None

    def on_train_begin(self, args, state, control, **kwargs):
        model = kwargs["model"]

        self.curriculum_state.update_from_step(0)
        stage = self.curriculum_state.current_stage
        k = get_stage_topk(stage, self.difficulty_to_topk)

        set_top_k(model, k=k)
        print(f"[model_curriculum] step=0 stage={stage} topk={k}")

        self._last_stage = stage
        self._last_k = k
        return control

    def on_step_begin(self, args, state, control, **kwargs):
        model = kwargs.get("model", None)
        if model is None:
            return control

        step = int(state.global_step)
        self.curriculum_state.update_from_step(step)

        stage = self.curriculum_state.current_stage
        k = get_stage_topk(stage, self.difficulty_to_topk)

        if stage != self._last_stage or k != self._last_k:
            set_top_k(model, k=k)
            print(f"[model_curriculum] step={step} stage={stage} topk={k}")
            self._last_stage = stage
            self._last_k = k

        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            stage = self.curriculum_state.current_stage
            k = get_stage_topk(stage, self.difficulty_to_topk)
            logs["use_model_curriculum"] = 1.0
            logs["model_curriculum_stage"] = int(stage)
            logs["model_curriculum_topk"] = int(k)
        return control


class StandardModelRoutingCallback(TrainerCallback):
    def __init__(self, default_topk: int):
        self.default_topk = int(default_topk)

    def on_train_begin(self, args, state, control, **kwargs):
        model = kwargs["model"]
        set_top_k(model, k=self.default_topk)
        print(f"[model_curriculum] enabled=False fixed_topk={self.default_topk}")
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            logs["use_model_curriculum"] = 0.0
            logs["model_curriculum_topk"] = int(self.default_topk)
        return control
