import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DISHES_DIR = os.path.join(BASE_DIR, "base", "HowToCook", "dishes")
VECTORSTORE_DIR = os.path.join(BASE_DIR, "data", "vectorstore")

# Directories to skip
SKIP_DIRS = {"template"}

# Embedding model
EMBED_MODEL = "shibing624/text2vec-base-chinese"

# Chunking settings
REMOVE_FOOTER = True          # Remove the standard PR footer
SPLIT_H3 = True               # Split ## 操作 into ### sub-sections
