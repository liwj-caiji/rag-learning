"""Tests for preprocessing module: RecipeSplitter and metadata extraction."""

from src.preprocess.splitter import RecipeSplitter, collect_all_recipes
from src.preprocess.config import DISHES_DIR, SKIP_DIRS


# ======================================================================
# RecipeSplitter
# ======================================================================

class TestRecipeSplitter:
    """Verify chunk structure and metadata extraction."""

    def test_chunk_count_l1_l2(self, splitter_instance):
        """At minimum: 1 L1 (dish) + N L2 (section) chunks."""
        chunks = splitter_instance.split()
        assert len(chunks) >= 2, f"Expected >=2 chunks, got {len(chunks)}"

        levels = [c["level"] for c in chunks]
        assert "dish" in levels, "Missing L1 (dish) chunk"
        assert "section" in levels or "subsection" in levels, "Missing L2/L3 chunks"

    def test_l1_dish_chunk_metadata(self, splitter_instance):
        """L1 chunk should have dish_name, difficulty, calories."""
        chunks = splitter_instance.split()
        l1 = [c for c in chunks if c["level"] == "dish"]
        assert len(l1) == 1, f"Expected 1 L1 chunk, got {len(l1)}"

        meta = l1[0]["metadata"]
        assert meta["dish_name"] == "麻婆豆腐", f"Expected 麻婆豆腐, got {meta['dish_name']}"
        # category depends on path relative to DISHES_DIR; skip check for temp files
        assert meta["difficulty"] == "★★"
        assert "350" in meta["calories"]

    def test_l2_section_chunks(self, splitter_instance):
        """Verify L2 section chunks have correct section_type."""
        chunks = splitter_instance.split()
        sections = [c for c in chunks if c["level"] == "section"]

        section_types = [c["metadata"]["section_type"] for c in sections]
        assert "必备原料和工具" in section_types
        assert "注意事项" in section_types

    def test_l3_subsection_chunks(self, splitter_instance):
        """When SPLIT_H3=True, ## 操作 should be split into ### sub-chunks."""
        chunks = splitter_instance.split()
        subs = [c for c in chunks if c["level"] == "subsection"]

        assert len(subs) >= 2, f"Expected >=2 subsections, got {len(subs)}"
        sub_names = [s["metadata"]["subsection_name"] for s in subs]
        assert "预处理" in sub_names
        assert "主步骤" in sub_names
        assert "收尾" in sub_names

    def test_no_h3_fallback_to_section(self, splitter_no_h3):
        """Recipe without H3 should fall back to single section chunk for ## 操作."""
        chunks = splitter_no_h3.split()
        sections = [c for c in chunks if c["level"] == "section"]

        op_sections = [c for c in sections if c["metadata"].get("section_type") == "操作"]
        assert len(op_sections) == 1

    def test_footer_removal(self, splitter_with_footer):
        """PR footer should be removed from chunks."""
        chunks = splitter_with_footer.split()
        all_text = " ".join(c["text"] for c in chunks)
        assert "Pull request" not in all_text, "Footer was not removed"
        assert "Issue" not in all_text, "Footer was not removed"

    def test_chunk_has_text(self, splitter_instance):
        """Every chunk must have non-empty text."""
        chunks = splitter_instance.split()
        for c in chunks:
            assert c["text"] and c["text"].strip(), f"Empty text in chunk: {c['level']}"

    def test_dish_name_removes_suffix(self):
        """'的做法' should be stripped from dish name."""
        from src.preprocess.splitter import RecipeSplitter
        # Use a temporary file to test heading parsing
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
            f.write("# 红烧肉的做法\n\n预估难度：★★\n")
            fpath = f.name
        try:
            splitter = RecipeSplitter(fpath)
            name = splitter._extract_dish_name("红烧肉的做法")
            assert name == "红烧肉"
        finally:
            import os
            os.unlink(fpath)


# ======================================================================
# collect_all_recipes
# ======================================================================

class TestCollectRecipes:
    """Verify recipe file collection."""

    def test_collect_returns_md_files(self):
        """Should return .md files from DISHES_DIR."""
        paths = collect_all_recipes()
        assert len(paths) > 0, "No recipes found"
        assert all(p.endswith(".md") for p in paths)

    def test_skips_template_dir(self):
        """Should NOT include files from SKIP_DIRS."""
        paths = collect_all_recipes()
        for p in paths:
            rel = p.replace("\\", "/")
            for skip in SKIP_DIRS:
                assert f"/{skip}/" not in rel, f"Found skipped dir '{skip}' in {p}"
