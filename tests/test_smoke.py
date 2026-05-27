"""Smoke tests — verify all modules import and basic wiring is correct.

These tests run quickly without requiring the pre-built index or API keys.
They catch import errors, missing exports, and broken references that would
cause the app to crash on startup.
"""

import importlib
import pkgutil
import sys


class TestModuleImports:
    """Every public module should import without errors."""

    # Modules that require external dependencies or the index
    _EXPECTED_SRC_MODULES = [
        "src.preprocess.config",
        "src.preprocess.splitter",
        "src.preprocess.indexer",
        "src.rewriting.intent",
        "src.rewriting.rewriter",
        "src.rewriting.llm_intent",
        "src.retrieval.hybrid",
        "src.retrieval.filters",
        "src.retrieval.diversity",
        "src.generation.base",
        "src.generation.template",
        "src.generation.llm_generator",
        "src.generation.pipeline",
    ]

    def test_all_src_modules_import(self):
        """Every Python module under src/ should import cleanly."""
        failed = []
        for mod_name in self._EXPECTED_SRC_MODULES:
            try:
                importlib.import_module(mod_name)
            except Exception as e:
                failed.append(f"{mod_name}: {e}")
        assert not failed, f"Import failures:\n" + "\n".join(failed)

    def test_init_exports(self):
        """Package __init__.py files should export expected names."""
        import src.generation as gen
        assert hasattr(gen, "RAGPipeline")
        assert hasattr(gen, "TemplateGenerator")
        assert hasattr(gen, "LLMGenerator")
        assert hasattr(gen, "Generator")

        import src.rewriting as rw
        assert hasattr(rw, "IntentResult")
        assert hasattr(rw, "IntentType")
        assert hasattr(rw, "QueryRewriter")
        assert hasattr(rw, "RuleQueryRewriter")
        assert hasattr(rw, "LLMIntentClassifier")

        import src.retrieval as rt
        assert hasattr(rt, "hybrid_search")
        assert hasattr(rt, "dense_search")
        assert hasattr(rt, "sparse_search")
        assert hasattr(rt, "recommend_dishes")
        assert hasattr(rt, "apply_filters")
        assert hasattr(rt, "diversify_by_category")

    def test_app_imports_without_crashing(self):
        """app.py should import cleanly (no runtime side effects on import)."""
        import app
        assert hasattr(app, "_run_pipeline")
        assert hasattr(app, "_chat_answer")
        assert hasattr(app, "_search")
        assert hasattr(app, "_get_pipeline")
        assert hasattr(app, "_load_stats")
        assert hasattr(app, "demo")

    def test_app_creates_gradio_blocks(self):
        """The Gradio demo object should be created."""
        import app
        assert app.demo is not None
        # Gradio Blocks have a `blocks` dict for child components
        assert hasattr(app.demo, "blocks")


class TestConfigValues:
    """Configuration values should be reasonable and consistent."""

    def test_preprocess_config(self):
        """Preprocessing config should have valid paths."""
        from src.preprocess.config import DISHES_DIR, VECTORSTORE_DIR, SKIP_DIRS
        assert DISHES_DIR is not None
        assert VECTORSTORE_DIR is not None
        assert isinstance(SKIP_DIRS, (list, tuple, set))

    def test_llm_default_model(self):
        """Default LLM model should be set."""
        from src.rewriting.llm_intent import DEFAULT_MODEL
        assert isinstance(DEFAULT_MODEL, str)
        assert len(DEFAULT_MODEL) > 0

        from src.generation.llm_generator import DEFAULT_MODEL as GEN_MODEL
        assert isinstance(GEN_MODEL, str)
        assert len(GEN_MODEL) > 0


class TestClassHierarchy:
    """ABCs and concrete implementations should be properly wired."""

    def test_generator_abc(self):
        """Generator ABC should define generate method."""
        from src.generation.base import Generator
        assert hasattr(Generator, "generate")

    def test_template_generator_is_subclass(self):
        """TemplateGenerator should be usable wherever Generator is expected."""
        from src.generation.base import Generator
        from src.generation.template import TemplateGenerator
        assert isinstance(TemplateGenerator(), Generator)

    def test_llm_generator_is_subclass(self):
        """LLMGenerator should be usable wherever Generator is expected."""
        from src.generation.base import Generator
        from src.generation.llm_generator import LLMGenerator
        assert isinstance(LLMGenerator(), Generator)

    def test_rewriter_interface(self):
        """QueryRewriter should have rewrite method."""
        from src.rewriting.rewriter import QueryRewriter
        from src.rewriting.llm_intent import LLMIntentClassifier
        # Both should have `rewrite` method
        assert hasattr(QueryRewriter, "rewrite")
        assert hasattr(LLMIntentClassifier, "rewrite")

    def test_pipeline_accepts_custom_components(self):
        """RAGPipeline should accept custom rewriter and generator."""
        from src.generation import RAGPipeline
        from src.rewriting.intent import classify_intent
        from src.rewriting.rewriter import QueryRewriter, RuleQueryRewriter
        from src.generation import TemplateGenerator

        rewriter = RuleQueryRewriter()
        generator = TemplateGenerator()
        pipe = RAGPipeline(rewriter=rewriter, generator=generator)
        assert pipe.rewriter is rewriter
        assert pipe.generator is generator


class TestAppPipelineIntegration:
    """Verify app.py correctly wires to the pipeline module."""

    def test_get_pipeline_returns_rag_pipeline(self):
        """_get_pipeline(False) should return a RAGPipeline instance."""
        from app import _get_pipeline
        from src.generation import RAGPipeline
        pipe = _get_pipeline(False)
        assert isinstance(pipe, RAGPipeline)

    def test_on_llm_toggle_formatting(self):
        """LLM toggle should produce correct mode hints."""
        from app import on_llm_toggle
        _on, hint_on = on_llm_toggle(True)
        _off, hint_off = on_llm_toggle(False)
        assert hint_on != hint_off  # Different hints for different modes

    def test_run_pipeline_returns_dict(self):
        """_run_pipeline returns a dict even without index."""
        from app import _run_pipeline
        result = _run_pipeline("", 5, False)
        assert isinstance(result, dict)
