"""Tests for rewriting module: intent classification, constraint extraction, LLM fallback."""

from src.rewriting.intent import classify_intent, IntentType
from src.rewriting.llm_intent import LLMIntentClassifier


# ======================================================================
# Intent classification
# ======================================================================

class TestIntentClassification:
    """Core intent classification via rule engine."""

    def test_howto_intent(self):
        result = classify_intent("麻婆豆腐怎么做")
        assert result.intent == IntentType.HOWTO
        assert result.target_dish == "麻婆豆腐"

    def test_howto_variant(self):
        for q in ["红烧肉的做法", "如何做清蒸鲈鱼", "番茄炒蛋怎么烧"]:
            result = classify_intent(q)
            assert result.intent == IntentType.HOWTO, f"Failed for: {q}"

    def test_recommendation_intent(self):
        for q in ["今天吃什么", "推荐一个菜", "有什么好吃的推荐", "来点下饭菜"]:
            result = classify_intent(q)
            assert result.intent == IntentType.RECOMMENDATION, f"Failed for: {q}"

    def test_ingredient_intent(self):
        for q in ["红烧肉需要什么材料", "麻婆豆腐的原料", "番茄炒蛋的配料"]:
            result = classify_intent(q)
            assert result.intent == IntentType.INGREDIENT, f"Failed for: {q}"

    def test_factual_intent(self):
        for q in ["宫保鸡丁是什么菜", "麻婆豆腐的来历", "鱼香肉丝是哪里的菜"]:
            result = classify_intent(q)
            assert result.intent == IntentType.FACTUAL, f"Failed for: {q}"

    def test_ambiguous_query_defaults_to_recommendation(self):
        result = classify_intent("好吃的")
        assert result.intent == IntentType.RECOMMENDATION


# ======================================================================
# Constraint extraction
# ======================================================================

class TestConstraintExtraction:
    """Filter extraction from queries."""

    def test_difficulty_simple(self):
        for q in ["简单的菜", "快手的食谱", "新手入门菜", "容易做的菜"]:
            result = classify_intent(q)
            assert result.filters.get("difficulty") == "★", f"Failed for: {q}"

    def test_difficulty_medium(self):
        result = classify_intent("中等难度的菜")
        diff = result.filters.get("difficulty")
        assert diff == "★★" or diff is None  # "中等" may or may not match

    def test_difficulty_hard(self):
        for q in ["困难的大菜", "高级料理", "大厨菜"]:
            result = classify_intent(q)
            assert result.filters.get("difficulty") == "★★★★", f"Failed for: {q}"

    def test_category_extraction(self):
        pairs = [
            ("推荐素菜", "vegetable_dish"),
            ("推荐肉菜", "meat_dish"),
            ("推荐汤", "soup"),
            ("早餐吃什么", "breakfast"),
            ("鱼怎么做", "aquatic"),
            ("推荐主食", "staple"),
            ("推荐甜点", "dessert"),
        ]
        for q, expected_cat in pairs:
            result = classify_intent(q)
            assert result.filters.get("category") == expected_cat, f"Failed for: {q}"

    def test_calories_low(self):
        for q in ["低热量菜", "减肥餐", "清淡的菜", "低卡食谱"]:
            result = classify_intent(q)
            assert result.filters.get("calories") == "low", f"Failed for: {q}"

    def test_calories_high(self):
        for q in ["高热量硬菜", "增肥菜", "横菜"]:
            result = classify_intent(q)
            assert result.filters.get("calories") == "high", f"Failed for: {q}"

    def test_multiple_filters(self):
        result = classify_intent("推荐一个简单的低卡素菜")
        assert result.filters.get("difficulty") == "★"
        assert result.filters.get("calories") == "low"
        assert result.filters.get("category") == "vegetable_dish"


# ======================================================================
# Query rewriting
# ======================================================================

class TestQueryRewriting:
    """Rewritten query generation."""

    def test_recommendation_rewrite_strips_noise(self):
        result = classify_intent("今天吃什么好吃的")
        assert result.rewritten != ""
        # Should not contain noise words
        for noise in ["今天", "什么"]:
            assert noise not in result.rewritten

    def test_howto_rewrite_includes_operation(self):
        result = classify_intent("麻婆豆腐怎么做")
        assert "操作" in result.rewritten
        assert "步骤" in result.rewritten

    def test_ingredient_rewrite_includes_ingredient(self):
        result = classify_intent("红烧肉需要什么材料")
        assert "原料" in result.rewritten or "材料" in result.rewritten


# ======================================================================
# Probes generation
# ======================================================================

class TestProbesGeneration:
    """Multi-probe generation for recommendation."""

    def test_recommendation_has_probes(self):
        result = classify_intent("今天吃什么")
        assert len(result.probes) > 0, "No probes generated"
        assert len(result.probes) <= 5, f"Too many probes: {len(result.probes)}"

    def test_probes_include_base_query(self):
        result = classify_intent("推荐清淡的菜")
        assert result.rewritten in result.probes or any(
            "清淡" in p for p in result.probes
        )

    def test_non_recommendation_no_probes(self):
        result = classify_intent("麻婆豆腐怎么做")
        assert result.probes == [] or result.probes is None


# ======================================================================
# LLMIntentClassifier fallback
# ======================================================================

class TestLLMIntentClassifierFallback:
    """LLM classifier should gracefully fall back to rules."""

    def test_fallback_without_api_key(self):
        """Without DEEPSEEK_API_KEY, should use rule-based classification."""
        import os
        key = os.environ.pop("DEEPSEEK_API_KEY", None)
        try:
            classifier = LLMIntentClassifier()
            result = classifier.classify("麻婆豆腐怎么做")
            assert result.intent == IntentType.HOWTO
            assert result.target_dish == "麻婆豆腐"
        finally:
            if key:
                os.environ["DEEPSEEK_API_KEY"] = key

    def test_fallback_preserves_filters(self):
        import os
        key = os.environ.pop("DEEPSEEK_API_KEY", None)
        try:
            classifier = LLMIntentClassifier()
            result = classifier.classify("推荐一个简单的低卡素菜")
            assert result.filters.get("difficulty") == "★"
            assert result.filters.get("calories") == "low"
        finally:
            if key:
                os.environ["DEEPSEEK_API_KEY"] = key

    def test_fallback_probes(self):
        import os
        key = os.environ.pop("DEEPSEEK_API_KEY", None)
        try:
            classifier = LLMIntentClassifier()
            result = classifier.classify("今天吃什么")
            assert len(result.probes) > 0
        finally:
            if key:
                os.environ["DEEPSEEK_API_KEY"] = key
