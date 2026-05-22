from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "docs"
VECTORSTORE_DIR = BASE_DIR / "vectorstore" / "faiss_index"

load_dotenv(BASE_DIR / ".env")
