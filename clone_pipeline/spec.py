from dataclasses import dataclass, field


CLONE_ID_BASE = 9_000_000  # safely above all real question IDs

CLONE_TYPE_OFFSETS: dict[str, int] = {
    "identical": 0,
    "easy_paraphrase": 1,
    "hard_paraphrase": 2,
    "negation": 3,
    "negation_easy": 4,
    "negation_hard": 5,
}


def generate_clone_ids(
    source_q_id: int, n_clones: int, clone_type: str
) -> list[int]:
    """
    Deterministic synthetic IDs:
        9_000_000 + source_id * 1000 + type_offset * 100 + clone_index

    Supports up to 10 clone types × 99 clones each per source question.
    """
    if clone_type not in CLONE_TYPE_OFFSETS:
        raise ValueError(
            f"Unknown clone_type '{clone_type}'. "
            f"Known types: {list(CLONE_TYPE_OFFSETS.keys())}"
        )
    type_offset = CLONE_TYPE_OFFSETS[clone_type]
    base = CLONE_ID_BASE + source_q_id * 1000 + type_offset * 100
    return [base + i for i in range(1, n_clones + 1)]


@dataclass
class CloneSpec:
    source_q_id: int
    clone_type: str  # "identical" | "easy_paraphrase" | "hard_paraphrase" | ...
    n_clones: int
    clone_ids: list[int] = field(init=False)
    flip_answers: bool = False

    def __post_init__(self):
        self.clone_ids = generate_clone_ids(
            self.source_q_id, self.n_clones, self.clone_type
        )

    @property
    def clone_id_str(self) -> str:
        """Human-readable identifier for this spec, used in folder naming."""
        return f"{self.clone_type}_q{self.source_q_id}_n{self.n_clones}"
