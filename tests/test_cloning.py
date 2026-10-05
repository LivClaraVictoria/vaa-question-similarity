import pytest

from experiments._cloning import build_clone_specs
from experiments._common import PERFECT_MIX_COMPONENTS


def test_single_type_spec():
    (spec,) = build_clone_specs(32214, "easy_paraphrase", 4)
    assert (spec.source_q_id, spec.clone_type, spec.n_clones, spec.flip_answers) == (32214, "easy_paraphrase", 4, False)


def test_negation_flips_answers():
    (spec,) = build_clone_specs(32214, "negation_hard", 4)
    assert spec.flip_answers


def test_perfect_mix_splits_clones_equally():
    specs = build_clone_specs(32214, "perfect_mix", 4)
    assert [s.clone_type for s in specs] == PERFECT_MIX_COMPONENTS
    assert {s.n_clones for s in specs} == {1}
    assert [s.flip_answers for s in specs] == [False, False, True, True]
