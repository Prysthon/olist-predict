from pathlib import Path

# Raiz do projeto
PROJ_ROOT = Path(__file__).resolve().parents[1]

# Diretórios de dados
DATA_DIR = PROJ_ROOT / "data"

BRONZE_DATA_DIR = DATA_DIR / "bronze"
SILVER_DATA_DIR = DATA_DIR / "silver"
GOLD_DATA_DIR = DATA_DIR / "gold"

# Outros diretórios
MODELS_DIR = PROJ_ROOT / "models"
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"