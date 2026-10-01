"""
. الملف ده فكرته الأساسية Configuration Management، يعني بدل ما كل جزء في الـRAG يكتب الـpaths
والـhyperparameters بنفسه، كلهم ياخدوا الإعدادات من مكان واحد
هدفه يكون المكان المركزي اللي المشروع كله ياخد منه الـconfiguration والـpaths والـenvironment variables.
"""

"""Centralized project configuration.

Loads configs/config.yaml once and exposes it to every stage
(indexing, retrieval, generation, API) so no module hard-codes
hyperparameters or paths. Secrets stay in .env (see generation/llm.py).

Project layout (this file = src/rag_project/config/settings.py):
    parents[0] = config, [1] = rag_project, [2] = src, [3] = project root

    يعني : ال project root الاصلي داخله (src) داخله (rag_project) داخله (config) 
    ال project root جواه اي ملف او فولدر خارجي او داخلي من للي انت عاملهم دول 
    بالتالي هعرف ال project root واستدعي اي ملف او فولدر انا عايز منه 
    project root --> Egyptian Civil RAG Project/    
"""

from pathlib import Path

import yaml
from dotenv import load_dotenv
"""
.env
  ↓
load_dotenv()
  ↓
Environment Variables / Secrets
"""

# Load secrets (API keys, ...) from the project .env file into the
# environment once, at the configuration layer, so every module
# (scripts, tests, API) sees the same environment variables.
load_dotenv()

# Project root, derived from this file's location.
# هعرف ال path location بتاعه 
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# The central configuration file.
# Path أداة من Python للتعامل مع file paths.
# من ال project root خش جوة ال configs وهات من داخله ال config.yaml
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


# function هدفها --> تقرأ config.yaml وتحوله إلى Python dictionary.
def load_config(path=None):
    """Read the YAML configuration file and return it as a dictionary."""
    # Use the project's config file when no explicit path is given.
    config_path = Path(path) if path is not None else CONFIG_PATH

    # Fail clearly when the configuration file is missing.
    # check Is the config path exist or not ?  if not exist print and show "Configuration file not found"
    # المشكلة اللي بتحلها؟
    # بدل ما يحصل error غامض بعدين، تعرف مباشرة إن: 
    # الـconfiguration file نفسه ناقص.

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # UTF-8 keeps the file readable with any language in it.
    # يفتح ملف ال yaml ويقراها "r" و باي لغه -> (encoding="utf-8") ك config_file
    with open(config_path, "r", encoding="utf-8") as config_file:
    # دي بتحول محتوى YAML إلى Python object، غالبًا dictionary.
    # هدفها --> تخلي باقي المشروع يتعامل مع الـconfig كـPython data بدل ما يقرأ YAML بنفسه.    
        return yaml.safe_load(config_file)


# Loaded once at import; every module reads its section from here.
CONFIG = load_config()

"""
config.yaml
  ↓
load_config()
  ↓
CONFIG
  ↓
indexing / retrieval / generation / API
"""

# function --> هدفها توحيد ال paths كلها ونشتغل عليها موحدة في البروجت كله 
# فهنحول ال paths من --> relative الي absolute
def resolve_path(relative_path):
    """Turn a project-relative config path into an absolute path."""
    path = Path(relative_path)
    # Absolute paths are used as they are.
    if path.is_absolute():
        return path
    # Relative paths are anchored at the project root.
    return PROJECT_ROOT / path
