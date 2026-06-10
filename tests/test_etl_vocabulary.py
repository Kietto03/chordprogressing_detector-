import pytest
from src.etl_pipeline import ChordVocabularyMapper


@pytest.mark.parametrize(
    "raw_chord, expected",
    [
        # Basic majors
        ("C:maj", 0),
        ("C", 0),
        ("C:maj7", 0),
        ("C:7", 0),
        ("C:sus4", 0),
        # Minors
        ("G:min", 19),
        ("G:min7", 19),
        ("G:m", 19),
        # N / silence / empty
        ("N", 24),
        ("N/A", 24),
        ("", 24),
        (None, 24),
        # With extensions and special
        ("F#7", 6),
        ("Db:min7", 13),
        ("G:min(*5)", 19),
        ("A:sus4/b7", 9),
        # Colon-qualified dim (triggers is_minor)
        ("C#:dim", 13),
        ("C:dim", 12),
        ("C#dim", 1),  # regression: no colon -> quality_str=None -> major root only
        # Complex / unrecognized fallback to N
        ("X:weird", 24),
        ("H:maj", 24),  # invalid root
        ("C:unknownext", 0),  # unknown ext but root major
        # More coverage
        ("Bb:maj", 10),
        ("A#:min", 22),
        ("E:aug", 4),
        ("F:sus2", 5),
        ("D#:hdim", 15),  # half-diminished
        ("C:maj(*5)/G", 0),  # slash bass ignored by regex
    ],
)
def test_map_chord(mapper_instance, raw_chord, expected):
    """Test ChordVocabularyMapper with verified cases from audit + regression for no-colon dim."""
    result = mapper_instance.map_chord(raw_chord)
    assert result == expected, f"map_chord({raw_chord!r}) = {result}, expected {expected}"
    assert 0 <= result <= 24


def test_map_chord_fallbacks(mapper_instance):
    """Additional fallback and edge paths."""
    assert mapper_instance.map_chord("   ") == 24
    assert mapper_instance.map_chord("N ") == 24
    # Regex edge that has quality but not triggering minor
    assert mapper_instance.map_chord("C:foo") == 0
    # Starts with m but is maj -> major
    assert mapper_instance.map_chord("C:maj") == 0
