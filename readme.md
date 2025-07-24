

# 🤖 UDS Multi-Agent QA Chatbot

A scalable, LLM-powered multi-agent system to query and compare **Unified Diagnostic Services (UDS)- ISO 14229** documentation using RAG + reasoning agents, Traceloop observability, and a Streamlit UI.


---

## 📌 Features

- 🔍 **RAG-based Retrieval** with Google Gemini models
- 🧠 **Multi-Agent Architecture** (FunctionAgents + ReActAgent)
- 📎 **Semantic Search + Summarization** per document
- 🔁 **Dynamic Tool Injection** for multi-doc comparison
- 📊 **Observability** with Traceloop SDK
- 📦 **Fully Dockerized** (with volume mounts)
- 🧑‍💻 **Streamlit UI** for uploading `.txt` UDS docs, querying, and history tracking

---

## 📂 Project Structure

```

.
├── app.py                   # Main Streamlit app
├── DATAN/                  # UDS .txt files uploaded by user
├── storage/                # Per-document vector/summary indices
├── summaries/              # Cached short summaries (.pkl)
├── requirements.txt
├── .env                    # API keys
├── Dockerfile
├── docker-compose.yml
└── README.md

````

---

## 🛠️ Tech Stack

| Component        | Stack                     |
|------------------|----------------------------|
| LLM & Embeddings | Gemini 2.5 Pro + Gemini Embeddings |
| Agent Framework  | LlamaIndex (FunctionAgent, ReActAgent) |
| Reranker         | Cohere Rerank v3.5         |
| Observability    | Traceloop SDK              |
| UI               | Streamlit                  |
| Containerization | Docker + Docker Compose    |

---

## 🚀 Getting Started

### 1️⃣ Clone and Setup

```bash
git clone <your-repo>
cd <your-repo>
````

### 2️⃣ Create `.env` file

```
GOOGLE_API_KEY=your_google_gemini_key
TRACELOOP_API_KEY=your_traceloop_key
COHERE_API_KEY=your_cohere_key
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirement.txt
```

### 4️⃣ Run Locally

```bash
streamlit run app.py
```

### 5️⃣ OR Run via Docker

```bash
docker-compose up --build
```

Then visit 👉 [http://localhost:8501](http://localhost:8501)

---

## 📸 UI Screenshots: Attachd The  CHAT HISTORY which contains UI

* Upload `.txt` files of UDS services
* Click **"Initialize Services"**
* Ask any question:

  * *“Compare ECU Reset, Tester Present, Link control services and explain how they are related”*
  * *“	In a scenario where the ECU is reset frequently, how should Tester Present messages be scheduled?”*

---

## 🧠 How It Works

### ✅ Per-Document Agents

Each `.txt` file → `Document` →
`VectorStoreIndex` + `SummaryIndex` + 2 tools →
`FunctionAgent` with:

* **semantic search**
* **summarization**

### 🔁 Top-Level ReActAgent

Uses:

* **ObjectIndex** over all tools
* **Cohere reranker** to pick relevant tools
* Injects dynamic `compare_tool` if query spans multiple agents

### 💬 UI Features

* Upload `.txt` UDS files
* Initialize all services
* Query interface
* History with answers from:

  * **TopAgent (Multi-agent reasoning)**
  * **Baseline RAG (vector-only)**

---

## 📊 Observability with Traceloop

* Logs agent runs, model latency, and trace steps
* Tracked automatically using:

```python
from traceloop.sdk import Traceloop
Traceloop.init(api_key=os.getenv("TRACELOOP_API_KEY"))
```

---

## 📦 Docker Tips

Mount folders ensure persistence:

```yaml
volumes:
  - ./storage:/app/storage
  - ./summaries:/app/summaries
  - ./DATAN:/app/DATAN
```

---

## 💡 Example Queries

* Compare ECU Reset, Link Control and Tester Present Services and explain how they are releated
* "What are sub fucntion parameter in DiagnosticSessionControl and Link Control

---

## 🏁 Final Notes

* Supports scalable document QA and multi-agent comparisons.
* Cohere Rerank boosts accuracy of tool selection.
* Built with extensibility and production-readiness in mind.

