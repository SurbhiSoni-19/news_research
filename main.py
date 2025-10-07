import os
import streamlit as st
import pickle
import time
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import torch

# --- Model Configuration ---
MODEL_NAME = "google/flan-t5-small"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# ---------------------------

st.title("RockyBot: News Research Tool (Local HF) 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i + 1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_huggingface.pkl"

# 🌟 FIX: main_placeholder MUST BE DEFINED BEFORE IT IS USED BELOW
main_placeholder = st.empty()

# 1. LLM INITIALIZATION: Load Model and Pipeline (with device fix)
main_placeholder.text(f"Loading Hugging Face LLM: {MODEL_NAME}...")
llm = None
try:
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # CRITICAL FIX for 'meta tensor' error: Force loading onto CPU
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        device_map="cpu",
    )

    # Create the text generation pipeline
    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=50,
        temperature=0.0,
        model_kwargs={"torch_dtype": torch.float32},
    )

    llm = HuggingFacePipeline(pipeline=pipe)
    main_placeholder.text(f"Hugging Face LLM ({MODEL_NAME}) Loaded Successfully! ✅")
except Exception as e:
    st.error(f"Error loading Hugging Face model: {e}. Check your resources and model name.")
    llm = None

if llm and process_url_clicked:
    # --- IMPORTANT: Delete the old FAISS file if it exists, as the embeddings are about to be re-initialized ---
    if os.path.exists(file_path):
        os.remove(file_path)

    # load data
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()

    # split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)

    # 2. EMBEDDINGS INITIALIZATION: Use HuggingFaceEmbeddings (with device fix)
    main_placeholder.text("Initializing Hugging Face Embeddings...✅✅✅")

    # CRITICAL FIX for 'meta tensor' error: Force embedding client onto CPU
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'}
    )

    # create embeddings and save it to FAISS index
    vectorstore_hf = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)

    # Save the FAISS index to a pickle file
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_hf, f)
    main_placeholder.success(f"FAISS index saved to {file_path}")

# --- RAG QUERY SECTION ---
query = main_placeholder.text_input("Question: ")
if query and llm:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)

            # Use the loaded HuggingFacePipeline LLM
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())

            # The input key to the chain is 'question'
            result = chain({"question": query}, return_only_outputs=True)

            st.header("Answer")
            st.write(result["answer"])

            # Display sources, if available
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")
                for source in sources_list:
                    st.write(source)
    else:
        st.error(f"FAISS index not found at {file_path}. Please process the URLs first.")