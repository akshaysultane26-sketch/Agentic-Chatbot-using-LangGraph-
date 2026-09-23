# Agentic Chatbot using LangGraph

An agentic AI chatbot built with **LangGraph** and **Streamlit**, featuring tool calling, retrieval-augmented generation (RAG), persistent memory, streaming responses, multi-conversation threading, and human-in-the-loop approval — deployed via a full CI/CD pipeline to AWS EC2 using Docker and GitHub Actions.

## Features

- **Conversational chatbot** built as a LangGraph state graph (`START → chat_node → END`)
- **Tool calling** — the LLM can call external tools when needed:
  - Web search (DuckDuckGo)
  - Calculator
  - RAG-based document retrieval
  - Stock purchase (demonstrates human-in-the-loop approval)
- **Retrieval-Augmented Generation (RAG)** — upload a PDF directly in the chat UI (via the attach icon in the message box); the app chunks it, embeds it with a local HuggingFace embedding model, stores it in a FAISS vector index, and answers questions grounded in the document
- **Persistent memory** — conversations are saved to a SQLite database (`SqliteSaver`), so chat history survives app restarts
- **Multi-conversation threading** — a sidebar lets you start new chats and switch between past conversations, each with its own memory thread
- **Streaming responses** — answers are streamed token-by-token, ChatGPT-style
- **Human-in-the-loop (HITL)** — certain actions (like purchasing a stock) pause the graph and require explicit user approval before completing, using LangGraph's `interrupt`/`Command` mechanism
- **Observability** — integrated with LangSmith for tracing every LLM call and tool invocation
- **CI/CD deployment** — pushes to `main` automatically build a Docker image, push it to Docker Hub, and deploy it to an AWS EC2 instance via GitHub Actions and a self-hosted runner

## Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph |
| LLM | Groq (`openai/gpt-oss-20b`) |
| Embeddings | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` (local, no API key) |
| Vector store | FAISS |
| UI | Streamlit |
| Persistence | SQLite (`langgraph-checkpoint-sqlite`) |
| Observability | LangSmith |
| Containerization | Docker |
| CI/CD | GitHub Actions (self-hosted runner) |
| Hosting | AWS EC2 |

## Project Structure

```
.
├── app.py               # Streamlit frontend (chat UI, threading, HITL approval, file upload)
├── backend.py            # LangGraph graph definition: state, nodes, tools, checkpointer
├── requirements.txt       # Python dependencies
├── Dockerfile             # Container build instructions
├── .dockerignore           # Files excluded from the Docker image
└── .github/workflows/
    └── cicd.yml            # GitHub Actions CI/CD pipeline
```

## Running Locally

1. Clone the repo:
   ```bash
   git clone https://github.com/akshaysultane26-sketch/Agentic-Chatbot-using-LangGraph-.git
   cd Agentic-Chatbot-using-LangGraph-
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root with:
   ```
   GROQ_API_KEY=your_groq_key
   LANGSMITH_TRACING=true
   LANGSMITH_ENDPOINT=https://api.smith.langchain.com
   LANGSMITH_API_KEY=your_langsmith_key
   LANGSMITH_PROJECT=your_project_name
   ```

4. Run the app:
   ```bash
   streamlit run app.py
   ```

## Deployment

This project deploys automatically to AWS EC2 on every push to `main`, via a GitHub Actions pipeline that:
1. Runs CI checks
2. Builds a Docker image and pushes it to Docker Hub
3. Deploys the image to an EC2 instance configured as a GitHub self-hosted runner

See `.github/workflows/cicd.yml` for the full pipeline definition.

## License

See [LICENSE](LICENSE) for details.