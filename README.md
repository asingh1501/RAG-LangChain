# 🤖 RAG-LangChain — Intelligent Document Query System

RAG-LangChain is a Retrieval-Augmented Generation (RAG) pipeline that allows users to query a company’s internal Markdown documents and receive accurate, context-aware answers. By combining **LangChain**, **OpenAI embeddings**, and **ChromaDB**, this project transforms unstructured documents into a searchable knowledge base.

---

## 📌 Project Overview
This project is designed to make company documents easily accessible through natural language queries. It indexes Markdown files, breaks them into meaningful chunks, embeds them into a vector database, and retrieves relevant information to answer questions accurately. This approach demonstrates a powerful combination of **document retrieval** and **language generation**.

---

## ⚙️ Tech Stack
- **Python**
- **LangChain**
- **OpenAI Embeddings & Chat Models**
- **ChromaDB (vector database)**
- **dotenv** for environment configuration

---

## 🚀 Workflow

### **1. Document Loading**
- Uses `DirectoryLoader` to automatically load `.md` files from the company data directory.

### **2. Text Splitting**
- Applies `RecursiveCharacterTextSplitter` to break documents into overlapping, context-preserving chunks (~300 characters per chunk).

### **3. Vector Embeddings**
- Generates embeddings for each chunk using `OpenAIEmbeddings`.

### **4. Database Storage**
- Stores chunks in a persistent **ChromaDB** vector database for efficient retrieval.

### **5. Querying**
- Users can search the database with natural language queries.
- The system performs semantic similarity search to retrieve the most relevant document chunks.

### **6. RAG Response Generation**
- Injects the retrieved chunks into a structured prompt template.
- Generates answers using OpenAI chat models, providing both the response and the source documents.

---

## 📊 Results & Key Takeaways
- Efficiently indexes and searches large sets of Markdown documents.
- Generates **context-aware answers** grounded in actual company documents.
- Demonstrates the power of combining **vector databases** with **LLM-based retrieval**.
- Shows best practices in **chunking, embedding, and persistent vector storage**.

---

## 📂 Future Improvements
- Add **multi-file support** for different file types (PDF, DOCX).  
- Implement **k-fold validation** for RAG retrieval performance evaluation.  
- Add a **web-based frontend** for real-time querying.  
- Include **source highlighting** to show exactly which text supports each answer.  
- Experiment with **different embedding models** for improved relevance.

---

## 💼 Applications
- Internal knowledge retrieval and onboarding assistants.  
- Document-based chatbots and virtual assistants.  
- Automated company report generation.  
- RAG-powered search engines for enterprise documents.

---

## ▶️ How to Run
1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd RAG-LangChain
