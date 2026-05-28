"""Chinese tokenizer for BM25 using jieba. Shared across both implementations."""

import jieba

jieba.setLogLevel(20)


def chinese_tokenize(text: str) -> list:
    tokens = jieba.lcut(text)
    return [t.strip() for t in tokens if t.strip()]
