import os
import re
from typing import List, Dict

from .config import DISHES_DIR, SKIP_DIRS, REMOVE_FOOTER, SPLIT_H3

# Standard footer to remove
FOOTER_PATTERN = re.compile(r"如果您遵循本指南的制作流程而发现有问题或可以改进的流程，请提出 Issue 或 Pull request 。")


class RecipeSplitter:
    """Hierarchical markdown recipe splitter.

    Produces up to 3 levels of chunks:
      L1-dish:       H1 + preamble (description, difficulty, calories)
      L2-section:    each ## section
      L3-subsection: each ### within ## 操作 (if SPLIT_H3 is True)
    """

    def __init__(self, filepath: str):
        self.filepath = filepath
        # Determine category from parent directory name
        rel = os.path.relpath(filepath, DISHES_DIR)
        parts = rel.replace("\\", "/").split("/")
        self.category = parts[0] if len(parts) > 1 else "unknown"

    def split(self) -> List[Dict]:
        """Return all chunks for this recipe file."""
        with open(self.filepath, "r", encoding="utf-8") as f:
            raw_lines = f.readlines()

        # Strip trailing whitespace but keep empty lines as markers
        lines = [l.rstrip("\n\r") for l in raw_lines]

        if REMOVE_FOOTER:
            lines = self._remove_footer(lines)

        # Group lines by heading boundary
        sections = self._group_by_headings(lines)

        chunks = []
        for sec in sections:
            if sec["level"] == 1:
                chunks.append(self._make_l1_chunk(sec))
            elif sec["level"] == 2:
                chunks.extend(self._make_l2_chunks(sec))
        return chunks

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _remove_footer(self, lines: List[str]) -> List[str]:
        """Remove the standard PR-footer line(s)."""
        result = []
        for line in lines:
            if not FOOTER_PATTERN.search(line):
                result.append(line)
        # Trim trailing empty lines
        while result and result[-1].strip() == "":
            result.pop()
        return result

    def _group_by_headings(self, lines: List[str], split_level: int = 2) -> List[Dict]:
        """Split lines into sections by heading boundaries.

        Args:
            split_level: Only headings at this level or higher (lower number)
                         create new section boundaries. Default 2: # and ## split.
                         Pass 3 to also split on ### (used for subsection splitting).

        Returns list of dicts:
          {level: int, heading: str, content_lines: List[str]}
        where level=1 for '#', level=2 for '##', level=3 for '###', etc.
        The first section (before any heading) gets level=0, heading=''.
        """
        heading_re = re.compile(r"^(#{1,6})\s+(.*)")
        sections = []
        current = None

        for line in lines:
            m = heading_re.match(line)
            if m:
                level = len(m.group(1))
                heading = m.group(2).strip()
                if current is not None and level <= split_level:
                    sections.append(current)
                    current = {
                        "level": level,
                        "heading": heading,
                        "content_lines": [line],
                    }
                else:
                    # Sub-heading or first heading: keep in current section
                    if current is None:
                        current = {
                            "level": level,
                            "heading": heading,
                            "content_lines": [line],
                        }
                    else:
                        current["content_lines"].append(line)
            else:
                if current is None:
                    current = {
                        "level": 0,
                        "heading": "",
                        "content_lines": [],
                    }
                current["content_lines"].append(line)

        if current is not None:
            sections.append(current)

        return sections

    def _make_l1_chunk(self, sec: Dict) -> Dict:
        """L1 dish-level chunk: H1 + preamble up to first H2."""
        dish_name = self._extract_dish_name(sec["heading"])
        text = "\n".join(sec["content_lines"])

        # Extract structured metadata from preamble
        difficulty = self._extract_difficulty(text)
        calories = self._extract_calories(text)

        return {
            "text": text,
            "level": "dish",
            "metadata": {
                "dish_name": dish_name,
                "category": self.category,
                "path": os.path.relpath(self.filepath, DISHES_DIR),
                "difficulty": difficulty,
                "calories": calories,
            },
        }

    def _make_l2_chunks(self, sec: Dict) -> List[Dict]:
        """L2 section chunks (one per ##), optionally splitting into L3."""
        dish_name = self._extract_dish_name_from_h1()

        if SPLIT_H3 and sec["heading"] == "操作":
            return self._make_l3_chunks(sec, dish_name)

        text = "\n".join(sec["content_lines"])
        section_type = sec["heading"]

        return [
            {
                "text": text,
                "level": "section",
                "metadata": {
                    "dish_name": dish_name,
                    "category": self.category,
                    "section_type": section_type,
                    "path": os.path.relpath(self.filepath, DISHES_DIR),
                },
            }
        ]

    def _make_l3_chunks(self, sec: Dict, dish_name: str) -> List[Dict]:
        """Split a ## 操作 section into ###-level sub-chunks."""
        subs = self._group_by_headings(sec["content_lines"], split_level=3)
        # Filter to level-3 subsections only
        chunks = []
        for sub in subs:
            if sub["level"] == 3:
                text = "\n".join(sub["content_lines"])
                chunks.append(
                    {
                        "text": text,
                        "level": "subsection",
                        "metadata": {
                            "dish_name": dish_name,
                            "category": self.category,
                            "section_type": "操作",
                            "subsection_name": sub["heading"],
                            "path": os.path.relpath(self.filepath, DISHES_DIR),
                        },
                    }
                )
        # If no ### subs found, fall back to the whole section
        if not chunks:
            text = "\n".join(sec["content_lines"])
            chunks.append(
                {
                    "text": text,
                    "level": "section",
                    "metadata": {
                        "dish_name": dish_name,
                        "category": self.category,
                        "section_type": sec["heading"],
                        "path": os.path.relpath(self.filepath, DISHES_DIR),
                    },
                }
            )
        return chunks

    def _extract_dish_name(self, heading: str) -> str:
        """Remove trailing '的做法' from H1 heading."""
        return re.sub(r"的做法$", "", heading).strip()

    def _extract_dish_name_from_h1(self) -> str:
        """Re-read H1 from file to get dish name."""
        # Use the first line of the file
        with open(self.filepath, "r", encoding="utf-8") as f:
            first = f.readline().strip()
        m = re.match(r"^#\s+(.+)$", first)
        if m:
            return self._extract_dish_name(m.group(1))
        return "unknown"

    @staticmethod
    def _extract_difficulty(text: str) -> str:
        m = re.search(r"预估烹饪难度[：:]\s*(\S+)", text)
        return m.group(1) if m else ""

    @staticmethod
    def _extract_calories(text: str) -> str:
        m = re.search(r"预估卡路里[：:]\s*(\S+)", text)
        return m.group(1) if m else ""


def collect_all_recipes() -> List[str]:
    """Walk DISHES_DIR and return paths to all .md files."""
    paths = []
    for root, dirs, files in os.walk(DISHES_DIR):
        # Skip unwanted directories
        rel_root = os.path.relpath(root, DISHES_DIR)
        parts = rel_root.replace("\\", "/").split("/")
        if any(p in SKIP_DIRS for p in parts):
            continue
        for fname in files:
            if fname.endswith(".md"):
                paths.append(os.path.join(root, fname))
    return sorted(paths)
