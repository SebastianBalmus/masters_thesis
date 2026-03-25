from dataclasses import dataclass


def get_stage_index(current_step: int, total_steps: int, num_stages: int) -> int:
    if num_stages <= 1:
        return 0
    if total_steps <= 0:
        return num_stages - 1

    step = min(max(int(current_step), 0), total_steps - 1)
    base, extra = divmod(total_steps, num_stages)
    cutoff = (base + 1) * extra

    if step < cutoff:
        return step // (base + 1)
    return extra + (step - cutoff) // max(1, base)


@dataclass
class CurriculumState:
    total_steps: int
    num_difficulty_levels: int
    current_stage: int = 0

    def update_from_step(self, step: int):
        self.current_stage = get_stage_index(
            current_step=step,
            total_steps=self.total_steps,
            num_stages=self.num_difficulty_levels,
        )
