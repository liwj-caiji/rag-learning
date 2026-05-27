"""Shared fixtures for RAG system tests."""

import os
import pickle
from pathlib import Path
from typing import Dict, List

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_DATA = PROJECT_ROOT / "tests" / "data"
VECTORSTORE = PROJECT_ROOT / "data" / "vectorstore"


# ---------------------------------------------------------------------------
# Fixtures: sample recipe markdown
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_recipe_text() -> str:
    return """# 麻婆豆腐的做法

预估烹饪难度：★★
预估卡路里：350 千卡

## 必备原料和工具

- 嫩豆腐 300g
- 牛肉末 50g
- 豆瓣酱 1 大勺
- 花椒粉 适量

## 操作

### 预处理

1. 豆腐切 2cm 方块
2. 烧一锅水，加盐

### 主步骤

1. 锅中加油，烧至六成热
2. 放入牛肉末炒至变色
3. 加入豆瓣酱炒出红油

### 收尾

1. 撒花椒粉
2. 装盘

## 注意事项

- 豆腐要嫩
- 火候要够
"""


@pytest.fixture
def sample_recipe_no_h3_text() -> str:
    return """# 番茄炒蛋的做法

预估烹饪难度：★
预估卡路里：200 千卡

## 必备原料和工具

- 番茄 2 个
- 鸡蛋 3 个
- 盐、糖 适量

## 操作

1. 番茄切块
2. 鸡蛋打散
3. 锅中加油，先炒鸡蛋
4. 加入番茄翻炒
5. 调味出锅
"""


@pytest.fixture
def sample_recipe_with_footer_text() -> str:
    return """# 清炒时蔬的做法

预估烹饪难度：★

## 必备原料和工具

- 青菜 300g
- 蒜 3 瓣

## 操作

1. 热锅冷油
2. 爆香蒜末
3. 加入青菜翻炒
4. 加盐调味

如果您遵循本指南的制作流程而发现有问题或可以改进的流程，请提出 Issue 或 Pull request 。
"""


# ---------------------------------------------------------------------------
# Fixtures: RecipeSplitter instances
# ---------------------------------------------------------------------------

@pytest.fixture
def splitter_instance(tmp_path, sample_recipe_text):
    """Create a RecipeSplitter from a temp file."""
    filepath = tmp_path / "meat_dish" / "麻婆豆腐.md"
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text(sample_recipe_text, encoding="utf-8")
    from src.preprocess.splitter import RecipeSplitter
    return RecipeSplitter(str(filepath))


@pytest.fixture
def splitter_no_h3(tmp_path, sample_recipe_no_h3_text):
    """Recipe with no H3 subsections."""
    filepath = tmp_path / "meat_dish" / "番茄炒蛋.md"
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text(sample_recipe_no_h3_text, encoding="utf-8")
    from src.preprocess.splitter import RecipeSplitter
    return RecipeSplitter(str(filepath))


@pytest.fixture
def splitter_with_footer(tmp_path, sample_recipe_with_footer_text):
    """Recipe with PR footer."""
    filepath = tmp_path / "vegetable_dish" / "清炒时蔬.md"
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text(sample_recipe_with_footer_text, encoding="utf-8")
    from src.preprocess.splitter import RecipeSplitter
    return RecipeSplitter(str(filepath))


# ---------------------------------------------------------------------------
# Fixtures: Retrieval (requires index)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def index_available() -> bool:
    """Check if FAISS/BM25 index is built."""
    return (
        (VECTORSTORE / "faiss.index").exists()
        and (VECTORSTORE / "chunks.pkl").exists()
        and (VECTORSTORE / "bm25_index.pkl").exists()
    )


# ---------------------------------------------------------------------------
# Fixtures: Rewriting
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_queries() -> Dict[str, str]:
    return {
        "howto": "麻婆豆腐怎么做",
        "recommendation_simple": "今天吃什么",
        "recommendation_filter": "推荐一个清淡的素菜",
        "ingredient": "红烧肉需要什么材料",
        "factual": "宫保鸡丁是什么菜系",
        "difficulty_low": "有没有简单的快手菜推荐",
        "calories_low": "减肥吃什么低卡路里的菜",
        "category_meat": "推荐一道肉菜",
        "multiple_filters": "有没有简单低卡路里的素菜推荐",
    }
