# AmbedkarGPT-Intern-Task.
AmbedkarGPT is a Retrieval-Augmented Generation (RAG) chatbot built as part of the KalpIT AI Intern Assessment. It answers questions from Dr. B. R. Ambedkar’s speech using embeddings, a vector database, and a local Large Language Model (LLM). The system retrieves highly relevant text chunks and generates context-aware responses grounded in the original speech, ensuring accuracy and reducing hallucinations.

🚀 Features:-

- Document Loader – Reads and processes Ambedkar's speech text.

- Text Chunking – Splits long text into meaningful sections.

- Embeddings – Uses MiniLM model to convert text into numerical vectors.

- ChromaDB Vector Store – Stores embeddings for fast similarity search.

- LLM Integration – Uses Mistral model through Ollama for answer generation.

- RAG Workflow – Retrieves relevant chunks + LLM reasoning = accurate answers.

- Interactive CLI – Ask any question related to the speech.
  
# Project Structure:-
AmbedkarGPT/
│── main.py
│── speech.txt
│── requirements.txt
│── db/  (auto-created)
└── README.md

🛠 Installation
1. Clone the repository
git clone https://github.com/your-username/AmbedkarGPT.git
cd AmbedkarGPT

2. Create and activate virtual environment (Python 3.10)
py -3.10 -m venv venv
venv\Scripts\activate

3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

4. Install Ollama and pull LLM

Download Ollama from: https://ollama.com

Then run:

ollama pull mistral

▶️ Usage

To start the chatbot:

python main.py


The CLI will open:

===== AmbedkarGPT =====
Ask any question about Ambedkar’s speech.
Type 'exit' to quit.

# Example Questions:-

What is Ambedkar’s main message in this speech?

What social problems does he discuss?

What does he say about equality?

Explain the “real remedy” mentioned in the speech.

# 🧠 How It Works (RAG Flow):-

Load and clean text

Split into chunks

Convert chunks into embeddings

Store embeddings in Chroma vector database

User asks a question

Retrieve top relevant chunks

Send to Mistral LLM for final answer

Return accurate, context-based response

# 📘 Technologies Used:-

Python

LangChain

ChromaDB

Transformers

MiniLM embeddings

Ollama (Mistral LLM)

Torch
