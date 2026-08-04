# 📚 AI Research Paper Assistant using Retrieval-Augmented Generation (RAG)

An AI-powered Research Paper Assistant that enables users to upload research papers, build a searchable knowledge base, and ask questions in natural language. The system uses **Retrieval-Augmented Generation (RAG)** to retrieve relevant document chunks and generate accurate answers using **GPT-4o**.

---

## 🚀 Features

- 📄 Upload research papers (PDF)
- 🔄 Automatically convert PDF to text
- ✂️ Intelligent document chunking
- 🧠 Generate embeddings using OpenAI
- 🗂️ Store embeddings in ChromaDB
- 🔍 Supports **Vector Search** and **Hybrid Search (Vector + BM25)**
- 💬 Ask questions in natural language
- 🤖 GPT-4o generates context-aware answers
- 📊 Retrieval analytics and execution logs
- 📓 Jupyter Notebook for step-by-step RAG pipeline demonstration

---

## 🛠️ Tech Stack

### Frontend
- Streamlit

### Backend
- Python

### AI & NLP
- LangChain
- OpenAI GPT-4o
- OpenAI Embeddings (`text-embedding-3-small`)

### Vector Database
- ChromaDB

### Retrieval Methods
- Vector Search
- Hybrid Search (Vector + BM25)

### Libraries
- LangChain
- ChromaDB
- Pandas
- NumPy
- Pydantic
- PyPDF
- Python-dotenv

---

## 📂 Project Structure

```text
AI-Research-Paper-Assistant/
│── app.py
│── rag_backend.py
│── ingestion_pipeline.py
│── rag_project_notebook.ipynb
│── requirements.txt
│── chunks_export.json
│── synthetic_questions.txt
│── README.md
│── .env.example
│── .gitignore
│
├── docs/
│     └── Research paper text files
│
├── db/
│     └── ChromaDB Vector Database
│
└── outputs/
```

---

## ⚙️ Workflow

```text
Upload PDF
      │
      ▼
Extract Text
      │
      ▼
Load Documents
      │
      ▼
Split into Chunks
      │
      ▼
Generate Embeddings
      │
      ▼
Store in ChromaDB
      │
      ▼
User Question
      │
      ▼
Vector / Hybrid Retrieval
      │
      ▼
Retrieve Top-K Chunks
      │
      ▼
GPT-4o
      │
      ▼
Final Answer
```

---

## 📋 How It Works

1. Upload a research paper.
2. The system extracts text from the PDF.
3. Documents are split into smaller chunks.
4. Embeddings are generated using OpenAI.
5. Chunks are stored in ChromaDB.
6. When a user asks a question, the system retrieves the most relevant chunks.
7. GPT-4o generates an answer using only the retrieved context.

---

## 🔍 Retrieval Methods

### Vector Search
Retrieves document chunks based on semantic similarity using embeddings.

### Hybrid Search
Combines semantic search (Vector Search) with keyword search (BM25) to improve retrieval accuracy.

---

## 📊 Notebook

The included Jupyter Notebook demonstrates each stage of the RAG pipeline:

- Document Loading
- Chunking
- Embedding Generation
- Vector Database Creation
- Vector Retrieval
- Hybrid Retrieval
- GPT-4o Answer Generation
- Retrieval Analysis

---

## ▶️ Installation

### Clone the repository

```bash
git clone https://github.com/Ranjanagopi2/AI-Research-Paper-Assistant.git

cd AI-Research-Paper-Assistant
```

### Create a virtual environment

```bash
python -m venv .venv
```

### Activate the environment

**Windows**

```bash
.venv\Scripts\activate
```

**Linux / macOS**

```bash
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file in the project root.

```env
OPENAI_API_KEY=your_openai_api_key
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

---

## 📸 Screenshots

<img width="1917" height="912" alt="image" src="https://github.com/user-attachments/assets/fc270de7-86af-4325-a7fc-ec7dff9adbd9" />
<img width="1917" height="912" alt="image" src="https://github.com/user-attachments/assets/9752d9e5-0fc6-4c77-b15e-0061420d11b5" />
<img width="1917" height="902" alt="image" src="https://github.com/user-attachments/assets/55d63655-2a8c-4cd3-905f-fffae51988b9" />


---

## 🔮 Future Enhancements

- Support multiple document formats (DOCX, PPTX)
- Multi-document retrieval
- Citation generation
- Conversation memory
- User authentication
- Cloud deployment
- Local LLM support (Ollama)

---

## 👨‍💻 Author

**Ranjana Gopi**

B.Tech Artificial Intelligence and Data Science

SNS College of Engineering

---

## 📄 License

This project is developed for educational and research purposes.
