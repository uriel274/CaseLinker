"""
Regression tests for MergeProcessing._merge_ages: duplicate-age collapse.

merge_processing.py:_merge_ages combines pattern + NER ages with
``list(set(...))`` at three points:
  - line ~383: all_pattern_perp_ages = list(set(pattern_perp_ages + pattern_perp_candidates))
  - line ~390: all_perp_ages = list(set(all_pattern_perp_ages + ner_perp_ages))
  - line ~410: candidate_victim_ages = list(set(pattern_victim_ages_filtered + ner_victim_ages))

``set()`` is correct if the goal is "which distinct ages appear in this
case", but perpetrator_age / case_demographics.ages are used elsewhere
(see scripts/stats/analyze_stats.py) as one-entry-per-person lists. When a
case has two or three people who share an age - e.g. three 34-year-old
co-defendants arrested together - the dedup silently drops real people:
the stored list looks identical to a case with exactly one person at that
age. This was found via manual review of scripts/validation sampling
output:
  - case_id newyork_sp_2024_013: two different defendants, both age 56;
    stored perpetrator_age has 56 only once.
  - case_id ncmec_2024_197: two different defendants, both age 41; stored
    perpetrator_age has 41 only once.

Per policy, no offender/victim/other personal names from case text appear
in this file -- cases are referenced by case_id only.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "src" / "Processing Layer"
PATTERN = PROC / "Pattern Processing Layer"
MERGE = PROC / "merge_processing.py"


def _load(name: str, path: Path, extra_paths: list[Path] | None = None):
    for p in extra_paths or []:
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


merge_mod = _load("merge_processing_under_test", MERGE, [PROC, PATTERN])


class TestDuplicatePerpetratorAgesCollapsed(unittest.TestCase):
    """Co-defendants who share an age must each be counted, not deduped away."""

    def test_three_codefendants_same_age_not_deduped(self):
        case_text = (
            "Three men were arrested in a joint sting operation: "
            "A, 34, of Dayton, B, 34, of Dayton, and "
            "C, 34, of Dayton, each face felony charges."
        )
        # What the pattern layer hands _merge_ages: one entry per person.
        merged = {
            "case_text": case_text,
            "source": "Test Wire",
            "perpetrator_age": [34, 34, 34],
        }
        mp = merge_mod.MergeProcessing()
        result = mp._merge_ages(merged, {"ages": []})

        perp_ages = result.get("perpetrator_age") or []
        self.assertEqual(
            len(perp_ages),
            3,
            f"expected 3 perpetrator-age entries (one per co-defendant), got {perp_ages} "
            "-- list(set(...)) in _merge_ages collapsed the duplicates",
        )

    def test_two_defendants_same_age_with_matching_ner(self):
        """Two distinct defendants, both 56, with NER echoing the same two mentions.

        Mirrors the shape of real case_id newyork_sp_2024_013 (two defendants,
        both age 56, stored once).
        """
        case_text = "Police arrested A, 56, of Johnstown, and B, 56, of Amsterdam."
        merged = {
            "case_id": "test_dedup",
            "case_text": case_text,
            "source": "Test Wire",
            "perpetrator_age": [56, 56],
        }
        ner = {"ages": ["56", "56"]}
        mp = merge_mod.MergeProcessing()
        result = mp._merge_ages(merged, ner)

        perp_ages = result.get("perpetrator_age") or []
        self.assertEqual(
            len(perp_ages),
            2,
            f"expected 2 perpetrator-age entries (A and B), got {perp_ages} "
            "-- list(set(...)) in _merge_ages collapsed the duplicates",
        )

    def test_two_of_three_share_an_age(self):
        """Mixed case: two people at 34, one at 50 -- must keep 3 entries, not 2."""
        case_text = (
            "A, 34, B, 34, and C, 50, were each "
            "charged following the investigation."
        )
        merged = {
            "case_text": case_text,
            "source": "Test Wire",
            "perpetrator_age": [34, 34, 50],
        }
        mp = merge_mod.MergeProcessing()
        result = mp._merge_ages(merged, {"ages": []})

        perp_ages = sorted(result.get("perpetrator_age") or [])
        self.assertEqual(
            perp_ages,
            [34, 34, 50],
            f"expected [34, 34, 50] (3 people), got {perp_ages}",
        )


class TestDuplicateVictimAgesCollapsed(unittest.TestCase):
    """Victims who share an age must each be counted, not deduped away."""

    def test_two_victims_same_age_not_deduped(self):
        case_text = (
            "Investigators identified a 16-year-old victim in Ohio and a "
            "second 16-year-old victim in Indiana after reviewing the material."
        )
        merged = {
            "case_text": case_text,
            "source": "Test Wire",
            "case_demographics": {"ages": [16, 16]},
        }
        mp = merge_mod.MergeProcessing()
        result = mp._merge_ages(merged, {"ages": []})

        victim_ages = (result.get("case_demographics") or {}).get("ages") or []
        self.assertEqual(
            len(victim_ages),
            2,
            f"expected 2 victim-age entries (one per child), got {victim_ages} "
            "-- list(set(...)) in _merge_ages collapsed the duplicates",
        )


if __name__ == "__main__":
    unittest.main()
