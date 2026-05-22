from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import DATA_DIR, VECTORSTORE_DIR
from app.llm import get_embeddings


def load_documents(data_dir: Path):
    txt_loader = DirectoryLoader(
        str(data_dir),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    pdf_loader = DirectoryLoader(
        str(data_dir),
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
    )
    docs = txt_loader.load() + pdf_loader.load()
    return docs


def main():
    docs = load_documents(DATA_DIR)
    if not docs:
        raise ValueError(f"No documents found in {DATA_DIR}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=120)
    chunks = splitter.split_documents(docs)

    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore = FAISS.from_documents(chunks, get_embeddings())
    vectorstore.save_local(str(VECTORSTORE_DIR))
    print(f"Indexed {len(chunks)} chunks into {VECTORSTORE_DIR}")


if __name__ == "__main__":
    main()
