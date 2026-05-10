"""Per-category reasoning generators.

Each generator has the signature

    (problem: Problem) -> Optional[str]

and returns a deterministic CoT trace ending with ``\\boxed{<answer>}``,
or ``None`` if the rule isn't recognized for that problem.

The 9 reasoner categories (matching THK's split):

    bit_manipulation         — local reasoner (essentially THK's, with a
                               more defensive prompt-header filter)
    cipher                   — THK port (verbose enumeration, reads
                               wonderland.txt)
    cryptarithm              — THK port; shared by deduce+guess
    equation_numeric         — THK port; shared by deduce+guess
    gravity                  — THK port (long-arithmetic)
    numeral                  — THK port (Roman incl. M/D/CM/CD)
    unit_conversion          — THK port (long-arithmetic)

GENERATORS includes the THK deduce/guess split keys as well as the legacy
combined keys our current ``scripts/categorize.py`` still emits. The
classifier update in Phase 4 will start emitting the split keys; the
combined keys remain registered as aliases so the current pipeline
doesn't break in the meantime.
"""
from .bit_manipulation import reasoning_bit_manipulation
from .cipher import reasoning_cipher
from .cryptarithm import reasoning_cryptarithm
from .equation_numeric import reasoning_equation_numeric
from .gravity import reasoning_gravity
from .numeral import reasoning_numeral
from .store_types import Example, Problem, build_problem
from .unit_conversion import reasoning_unit_conversion


GENERATORS = {
    "bit_manipulation": reasoning_bit_manipulation,
    "cipher": reasoning_cipher,
    # THK's split keys (Phase 4 classifier will emit these)
    "cryptarithm_deduce": reasoning_cryptarithm,
    "cryptarithm_guess": reasoning_cryptarithm,
    "equation_numeric_deduce": reasoning_equation_numeric,
    "equation_numeric_guess": reasoning_equation_numeric,
    # Legacy combined keys (current classifier emits these)
    "cryptarithm": reasoning_cryptarithm,
    "equation_numeric": reasoning_equation_numeric,
    "gravity": reasoning_gravity,
    "numeral": reasoning_numeral,
    "unit_conversion": reasoning_unit_conversion,
}

__all__ = ["GENERATORS", "Example", "Problem", "build_problem"]
