from dataclasses import dataclass, field


CLONE_ID_BASE = 9_000_000  # safely above all real question IDs


def generate_clone_ids(source_q_id: int, n_clones: int) -> list[int]:
    """
    Deterministic synthetic IDs: 9_000_000 + source_id * 100 + clone_index.
    Supports up to 99 clones per question without collision.
    """
    base = CLONE_ID_BASE + source_q_id * 100
    return [base + i for i in range(1, n_clones + 1)]


@dataclass
class CloneSpec:
    source_q_id: int
    clone_type: str  # "identical" | "negation" | "paraphrase" | ...
    n_clones: int
    clone_ids: list[int] = field(init=False)
    flip_answers: bool = False

    def __post_init__(self):
        self.clone_ids = generate_clone_ids(self.source_q_id, self.n_clones)

    @property
    def clone_id_str(self) -> str:
        """Human-readable identifier for this spec, used in folder naming."""
        return f"{self.clone_type}_q{self.source_q_id}_n{self.n_clones}"
