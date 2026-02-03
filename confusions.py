"""
Error injection module based on character confusion matrices.

Models OCR errors as weighted character substitutions derived from visual
similarity (Guan et al., EMNLP 2024 approach). Supports multi-level error
injection for generating training pairs at various CER targets.
"""

import random
import string
from typing import Optional

# ---------------------------------------------------------------------------
# Default confusion matrix: char -> {replacement: weight}
# Self-mapping weight is always 1.0 (baseline "no error").
# Non-self weights represent relative likelihood of that confusion.
# ---------------------------------------------------------------------------
DEFAULT_CONFUSIONS: dict[str, dict[str, float]] = {
    # Visual similarity -- letter confusions
    "r": {"r": 1.0, "n": 0.05},            # rn looks like m
    "n": {"n": 1.0, "r": 0.05, "m": 0.03},
    "m": {"m": 1.0, "rn": 0.06},            # m -> rn (1-to-2 expansion)
    "l": {"l": 1.0, "1": 0.08, "I": 0.06, "|": 0.03},
    "1": {"1": 1.0, "l": 0.08, "I": 0.05, "|": 0.03},
    "I": {"I": 1.0, "l": 0.06, "1": 0.05, "|": 0.03},
    "O": {"O": 1.0, "0": 0.10},
    "0": {"0": 1.0, "O": 0.10},
    "c": {"c": 1.0, "e": 0.03},
    "e": {"e": 1.0, "c": 0.03},
    "d": {"d": 1.0, "cl": 0.04},            # d -> cl
    "5": {"5": 1.0, "S": 0.07},
    "S": {"S": 1.0, "5": 0.07},
    "8": {"8": 1.0, "B": 0.06},
    "B": {"B": 1.0, "8": 0.06},
    "g": {"g": 1.0, "q": 0.04, "9": 0.03},
    "q": {"q": 1.0, "g": 0.04},
    "6": {"6": 1.0, "b": 0.04},
    "b": {"b": 1.0, "6": 0.04},
    "D": {"D": 1.0, "0": 0.03},
    "h": {"h": 1.0, "b": 0.03},
    "u": {"u": 1.0, "v": 0.03},
    "v": {"v": 1.0, "u": 0.03},
    "w": {"w": 1.0, "vv": 0.02},
    "f": {"f": 1.0, "t": 0.03},
    "t": {"t": 1.0, "f": 0.03},

    # Punctuation confusions
    ",": {",": 1.0, ".": 0.08},
    ".": {".": 1.0, ",": 0.08},
    "$": {"$": 1.0, "S": 0.05, "5": 0.03},
    ";": {";": 1.0, ":": 0.06},
    ":": {":": 1.0, ";": 0.06},

    # Space / segmentation errors
    " ": {" ": 1.0, "": 0.04},              # space deletion
}


def scale_confusions(
    base: dict[str, dict[str, float]],
    multiplier: float,
) -> dict[str, dict[str, float]]:
    """Scale non-self confusion weights by *multiplier*, then re-normalize.

    A multiplier of 1.0 leaves the matrix unchanged. Higher values increase
    the probability of errors for every character that has confusions defined.
    """
    scaled: dict[str, dict[str, float]] = {}
    for char, mapping in base.items():
        new_mapping: dict[str, float] = {}
        for replacement, weight in mapping.items():
            if replacement == char:
                new_mapping[replacement] = weight  # self-weight stays at 1.0
            else:
                new_mapping[replacement] = weight * multiplier
        # Normalize so weights sum to 1.0
        total = sum(new_mapping.values())
        if total > 0:
            new_mapping = {k: v / total for k, v in new_mapping.items()}
        scaled[char] = new_mapping
    return scaled


def _weighted_choice(mapping: dict[str, float], rng: random.Random) -> str:
    """Pick a replacement string from a {replacement: probability} mapping."""
    chars = list(mapping.keys())
    weights = list(mapping.values())
    return rng.choices(chars, weights=weights, k=1)[0]


def inject_errors(
    text: str,
    confusion_matrix: Optional[dict[str, dict[str, float]]] = None,
    error_level: float = 3.0,
    seed: Optional[int] = None,
) -> str:
    """Inject OCR-like errors into *text* at approximately *error_level* % CER.

    Error types are applied in a 5:1:1 ratio (substitution:deletion:insertion).
    Characters with confusion matrix entries use weighted substitutions;
    characters without entries use random substitutions from printable ASCII.

    Args:
        text: Clean input text.
        confusion_matrix: Character confusion weights. Defaults to
            ``DEFAULT_CONFUSIONS`` scaled to match *error_level*.
        error_level: Target character error rate as a percentage (e.g. 3.0
            means ~3% CER).
        seed: Random seed for reproducibility.

    Returns:
        Corrupted text string.
    """
    if not text:
        return text

    rng = random.Random(seed)

    if confusion_matrix is None:
        confusion_matrix = DEFAULT_CONFUSIONS

    # Per-character error probability to reach target CER
    p_error = error_level / 100.0
    # 5:1:1 ratio
    p_sub = p_error * (5 / 7)
    p_del = p_error * (1 / 7)
    p_ins = p_error * (1 / 7)

    printable = string.ascii_letters + string.digits + string.punctuation

    result: list[str] = []
    for ch in text:
        roll = rng.random()

        if roll < p_sub:
            # Substitution — always produce a different character
            if ch in confusion_matrix:
                non_self = {k: v for k, v in confusion_matrix[ch].items() if k != ch}
                if non_self:
                    replacement = _weighted_choice(non_self, rng)
                    result.append(replacement)
                else:
                    candidates = [c for c in printable if c != ch]
                    result.append(rng.choice(candidates) if candidates else ch)
            else:
                # Random substitution from printable chars (excluding self)
                candidates = [c for c in printable if c != ch]
                result.append(rng.choice(candidates) if candidates else ch)

        elif roll < p_sub + p_del:
            # Deletion — skip this character
            pass

        elif roll < p_sub + p_del + p_ins:
            # Insertion — keep original, insert a random char after it
            result.append(ch)
            result.append(rng.choice(printable))

        else:
            # No error
            result.append(ch)

    return "".join(result)


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def _levenshtein(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)

    if len(s2) == 0:
        return len(s1)

    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Insertion, deletion, substitution
            cost = 0 if c1 == c2 else 1
            curr_row.append(
                min(curr_row[j] + 1, prev_row[j + 1] + 1, prev_row[j] + cost)
            )
        prev_row = curr_row

    return prev_row[-1]


def compute_cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate: edit_distance(ref, hyp) / len(ref).

    Returns 0.0 if reference is empty.
    """
    if not reference:
        return 0.0
    return _levenshtein(reference, hypothesis) / len(reference)


def compute_wer(reference: str, hypothesis: str) -> float:
    """Word Error Rate: edit_distance on word tokens / num reference words.

    Returns 0.0 if reference has no words.
    """
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if not ref_words:
        return 0.0
    return _levenshtein_words(ref_words, hyp_words) / len(ref_words)


def _levenshtein_words(s1: list[str], s2: list[str]) -> int:
    """Levenshtein distance at the word level."""
    if len(s1) < len(s2):
        return _levenshtein_words(s2, s1)
    if len(s2) == 0:
        return len(s1)

    prev_row = list(range(len(s2) + 1))
    for i, w1 in enumerate(s1):
        curr_row = [i + 1]
        for j, w2 in enumerate(s2):
            cost = 0 if w1 == w2 else 1
            curr_row.append(
                min(curr_row[j] + 1, prev_row[j + 1] + 1, prev_row[j] + cost)
            )
        prev_row = curr_row

    return prev_row[-1]


# ---------------------------------------------------------------------------
# Multi-level pair generation
# ---------------------------------------------------------------------------

DEFAULT_LEVELS = [0.3, 1.0, 3.0, 7.0, 11.0, 15.0]


def generate_multi_level_pairs(
    text: str,
    levels: Optional[list[float]] = None,
    seed: Optional[int] = None,
) -> list[dict]:
    """Generate noisy versions of *text* at multiple CER target levels.

    Returns a list of dicts, each containing:
        - ``level``: target CER percentage
        - ``ground_truth``: the original clean text
        - ``noisy_text``: error-injected text
        - ``measured_cer``: actual CER after injection
        - ``measured_wer``: actual WER after injection
    """
    if levels is None:
        levels = DEFAULT_LEVELS

    results: list[dict] = []
    for i, level in enumerate(levels):
        # Derive a per-level seed so results are reproducible but independent
        level_seed = (seed * 1000 + i) if seed is not None else None
        noisy = inject_errors(text, error_level=level, seed=level_seed)
        results.append({
            "level": level,
            "ground_truth": text,
            "noisy_text": noisy,
            "measured_cer": round(compute_cer(text, noisy) * 100, 2),
            "measured_wer": round(compute_wer(text, noisy) * 100, 2),
        })
    return results


if __name__ == "__main__":
    sample = (
        "Invoice #1042 — Acme Corp.\n"
        "Bill To: John Smith, 123 Oak St.\n"
        "Total: $1,250.00\n"
    )
    print("=== Multi-level error injection demo ===\n")
    for pair in generate_multi_level_pairs(sample, seed=42):
        print(f"Target CER: {pair['level']}%  |  "
              f"Measured CER: {pair['measured_cer']}%  |  "
              f"Measured WER: {pair['measured_wer']}%")
        print(f"  Noisy: {pair['noisy_text'][:80]}...")
        print()
