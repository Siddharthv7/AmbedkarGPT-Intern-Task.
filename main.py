import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

def build_vector_db():
    print("Loading speech.txt...")
    loader = TextLoader("speech.txt")
    documents = loader.load()

    print("Splitting text into chunks...")
    splitter = CharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    print("Creating embeddings (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Building ChromaDB vectorstore...")
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="db"
    )

    vectordb.persist()
    print("Vectorstore saved to ./db")
    return vectordb


def load_vector_db():
    print("Loading existing ChromaDB vectorstore...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = Chroma(
        persist_directory="db",
        embedding_function=embeddings
    )
    return vectordb


def start_chat(vectordb):
    print("Loading Ollama Mistral model...")
    llm = Ollama(model="mistral")

    print("\nCreating RetrievalQA pipeline...")
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectordb.as_retriever()
    )

    print("\n===== AmbedkarGPT CLI =====")
    print("Ask anything about the speech. Type 'exit' to quit.\n")

    while True:
        q = input("Your question: ")
        if q.lower().strip() == "exit":
            print("Goodbye!")
            break

        print("\nThinking...\n")
        answer = qa.run(q)
        print("Answer:", answer)
        print("\n---------------------------------------\n")


if __name__ == "__main__":
    if not os.path.exists("db"):
        vectordb = build_vector_db()
    else:
        vectordb = load_vector_db()

    start_chat(vectordb)
