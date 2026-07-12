import os
import glob
import shutil
import torch

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


def build_custom_database():
    print("Initializing Database Setup...")

    db_path = "./local_chroma_db"
    docs_folder = "./my_documents"
    all_documents = []

    if os.path.exists(db_path):
        print("Removing existing vector database...")
        shutil.rmtree(db_path)

    if not os.path.exists(docs_folder):
        os.makedirs(docs_folder)
        print(f"Created folder '{docs_folder}'. Add your documents to this folder and run the script again.")
        return

    print("Scanning documents...")

    files = []
    files.extend(glob.glob(os.path.join(docs_folder, "*.pdf")))
    files.extend(glob.glob(os.path.join(docs_folder, "*.txt")))
    files.extend(glob.glob(os.path.join(docs_folder, "*.docx")))

    if not files:
        print("No supported document files were found.")
        return

    for file in files:
        print(f"Loading: {os.path.basename(file)}")

        if file.endswith(".pdf"):
            loader = PyPDFLoader(file)
            documents = loader.load()

            filtered = []

            for doc in documents:
                text = doc.page_content

                if (
                    "Reading with Insight" in text
                    or "Before you read" in text
                    or "Thinking about the Text" in text
                    or "Exercise" in text
                ):
                    continue

                filtered.append(doc)

            all_documents.extend(filtered)

        elif file.endswith(".txt"):
            all_documents.extend(TextLoader(file, encoding="utf-8").load())

        elif file.endswith(".docx"):
            all_documents.extend(Docx2txtLoader(file).load())

    print(f"Total Documents Loaded: {len(all_documents)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=60,
        separators=["\n\n", "\n", ".", " "],
    )

    final_chunks = splitter.split_documents(all_documents)

    print(f"Total Chunks Created: {len(final_chunks)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Generating embeddings...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": device},
    )

    print("Saving Chroma database...")

    db = Chroma.from_documents(
        documents=final_chunks,
        embedding=embeddings,
        persist_directory=db_path,
    )

    db.persist()

    print("Database creation completed successfully.")
    print(f"Database saved at: {os.path.abspath(db_path)}")


if __name__ == "__main__":
    build_custom_database()