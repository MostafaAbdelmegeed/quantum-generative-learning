import pytest

from qgen.replication_analysis import paired_summary


@pytest.mark.parametrize(
    "differences,expected",
    [
        ([-0.02] * 8 + [0.01] * 2, True),
        ([-0.5] * 7 + [0.2] * 3, False),
        ([-0.005] * 10, False),
    ],
)
def test_replication_requires_both_effect_size_and_consistency(differences, expected):
    assert paired_summary(differences)["benefit_replicated"] is expected


def test_replication_ties_and_missing_seeds():
    result = paired_summary([-1e-9, 0, 1e-9] + [-0.1] * 7)
    assert result["ties"] == 3
    assert result["improvements"] == 7
    assert not result["benefit_replicated"]
    with pytest.raises(ValueError, match="ten finite"):
        paired_summary([-0.1] * 9)
