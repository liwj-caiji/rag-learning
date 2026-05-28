"""Rule-based intent classifier — zero external dependencies."""

from .intent_types import (
    IntentResult, IntentType,
    match_intent, extract_filters, extract_target_dish,
    build_rewritten_query, build_probes,
)


def classify_intent(query: str) -> IntentResult:
    intent = match_intent(query)
    filters = extract_filters(query)
    target_dish = extract_target_dish(query, intent)

    result = IntentResult(
        intent=intent,
        rewritten=build_rewritten_query(query, intent, target_dish, filters),
        filters=filters,
        target_dish=target_dish,
    )

    if intent == IntentType.RECOMMENDATION:
        result.probes = build_probes(query, filters)

    return result


class RuleIntentClassifier:
    """LangChain-compatible intent classifier (rule-based)."""

    def classify(self, query: str) -> IntentResult:
        return classify_intent(query)

    def __call__(self, query: str) -> IntentResult:
        return classify_intent(query)

    def invoke(self, query: str, **kwargs) -> IntentResult:
        return classify_intent(query)

    def rewrite(self, query: str) -> IntentResult:
        return classify_intent(query)
