import warnings
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import torch
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_classic.chains import RetrievalQA
from langchain_classic.retrievers.contextual_compression import (
    ContextualCompressionRetriever,
)
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langchain_core.documents import Document
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.prompts import PromptTemplate

st.set_page_config(page_title="Local RAG Chatbot")


@st.cache_resource
def load_models_and_db():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    db_path = "./local_chroma_db"

    if not os.path.exists(db_path):
        return None, None, None, None, False

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": device},
    )

    vector_store = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings,
    )

    db_data = vector_store.get()

    if not db_data["documents"]:
        return None, None, None, None, False

    saved_chunks = [
        Document(page_content=txt, metadata=m)
        for txt, m in zip(db_data["documents"], db_data["metadatas"])
    ]

    bm25_retriever = BM25Retriever.from_documents(saved_chunks)

    cross_encoder = HuggingFaceCrossEncoder(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        model_kwargs={"device": device},
    )

    model_id = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(device)

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=256,
        device=0 if device == "cuda" else -1,
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    return vector_store, bm25_retriever, cross_encoder, llm, True


st.title("Document Question Answering System")

vector_store, bm25_retriever, cross_encoder, llm, is_ready = load_models_and_db()

if not is_ready:
    st.error(
        "Database not found or it is empty. Add your files to the 'my_documents' folder and run 'python setup_db.py' first."
    )

else:
    st.sidebar.header("Search Settings")

    top_k_retrieve = st.sidebar.slider(
        "Chunks to Retrieve",
        1,
        10,
        5,
    )

    top_k_rerank = st.sidebar.slider(
        "Chunks to Rerank",
        1,
        5,
        3,
    )

    user_query = st.text_input(
        "Ask a question based on your documents:"
    )

    if st.button("Generate Answer") and user_query:

        with st.spinner("Generating Answer..."):

            vector_retriever = vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 3,
                    "fetch_k": 8,
                    "lambda_mult": 0.7,
                },
            )

            bm25_retriever.k = 3

            hybrid_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, vector_retriever],
                weights=[0.3, 0.7],
            )

            reranker = CrossEncoderReranker(
                model=cross_encoder,
                top_n=2,
            )

            final_retriever = ContextualCompressionRetriever(
                base_compressor=reranker,
                base_retriever=hybrid_retriever,
            )

            prompt = PromptTemplate(
                template="""
                You are an AI assistant.
                Use ONLY the retrieved context.
                Do not copy the context verbatim.
                Answer the question in your own words using only the retrieved context.
                If the answer cannot be found, reply exactly:
                I could not find the answer in the provided document.
                Answer in 3-4 complete sentences.

                Context:
                {context}

                Question:
                {question}

                Answer:
                """,
                input_variables=["context", "question"],
            )

            rag_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=final_retriever,
                return_source_documents=True,
                chain_type_kwargs={
                    "prompt": prompt,
                },
            )

            response = rag_chain.invoke({"query": user_query})

            print("\n" + "=" * 100)
            print("QUERY:", user_query)
            print("=" * 100)

            for i, doc in enumerate(response["source_documents"]):
                print(f"\nChunk {i + 1}")
                print("-" * 100)
                print(doc.page_content)
                print("-" * 100)

            st.subheader("Answer")
            st.write(response["result"])

            st.subheader("Retrieved Context")

            for i, doc in enumerate(response["source_documents"]):

                st.markdown(f"### Chunk {i + 1}")

                st.code(doc.page_content)

                st.write("---")