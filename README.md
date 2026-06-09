# 🌌 Gemini PDF Question Answering Bot

A lightweight, high-performance, and visually stunning PDF Question Answering Bot built using a Python backend and Streamlit frontend. It processes uploaded PDFs, indexes their contents in-memory using Gemini text embeddings, and provides a fully grounded chatbot experience powered by Gemini 1.5 Flash.

---

## 🚀 Key Features

*   **Premium Glassmorphic UI**: Styled with a customized dark-mode aesthetic, harmonious neon accents, fluid cards, and clean typography.
*   **In-Memory Vector Search**: Zero databases or heavy frameworks. Chunk similarity calculations are computed locally using simple, high-performance `numpy` vector math (cosine similarity).
*   **Dynamic Embedding Model Discovery**: Automatically queries the Gemini API to detect active embedding models associated with your API key, falling back to `models/embedding-001` or `models/text-embedding-004`.
*   **Grounded QA Generation**: Instructs `gemini-1.5-flash` to answer questions *only* using retrieved passages, preventing hallucinations and outside knowledge drift.
*   **Precise Source Attribution**: Every chat response highlights the exact document chunks used for reasoning, along with their cosine similarity scores.
*   **Index Debugger & Statistics**: Real-time metrics showing total chunks, dimensions, character lists, and raw preview cards for the first and last indexed document chunks.

---

## 🛠️ Tech Stack

*   **Frontend**: [Streamlit](https://streamlit.io/) (enhanced with custom CSS)
*   **PDF Extraction**: [PyPDF](https://pypdf.readthedocs.io/)
*   **Vector Operations**: [NumPy](https://numpy.org/)
*   **AI Models**: [Google Generative AI SDK](https://github.com/google/generative-ai-python) (`gemini-1.5-flash`, `models/embedding-001`, `models/text-embedding-004`)

---

## 📥 Getting Started

### Prerequisites
*   Python 3.9 or higher installed.
*   A Google Gemini API key. You can get one for free at [Google AI Studio](https://aistudio.google.com/).

### Installation

1.  **Clone or create your project directory** and navigate to it:
    ```bash
    cd "d:/PDF Analyzer Bot"
    ```

2.  **Create and activate a virtual environment**:
    *   **Windows**:
        ```powershell
        python -m venv venv
        venv\Scripts\activate
        ```
    *   **macOS/Linux**:
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

---

## 🖥️ Running the Application

There are two ways to load your Gemini API Key into the application:

### Option A: Via Environment Variable (Recommended)
Set the environment variable in your terminal before launching:
*   **Windows (PowerShell)**:
    ```powershell
    $env:GEMINI_API_KEY="AIzaSyYourKeyHere..."
    streamlit run app.py
    ```
*   **macOS/Linux**:
    ```bash
    export GEMINI_API_KEY="AIzaSyYourKeyHere..."
    streamlit run app.py
    ```

### Option B: Via Sidebar Input
Simply run the app directly and enter your key in the secure password input field in the sidebar:
```bash
streamlit run app.py
```

---

## 📊 How the Pipeline Works

1.  **Text Ingestion**: When a PDF is uploaded, `pypdf` extracts the raw text page-by-page. Empty page checks are performed to detect scanned/image-only PDFs.
2.  **Chunking**: The extracted text is divided into segments of `1000` characters, using a sliding window with a `200` character overlap to maintain semantic continuity across boundaries.
3.  **Embedding Generation**: The app sends chunk lists in batches of up to 100 to the Gemini Embedding API, storing the resulting dense vector embeddings in the Streamlit session state memory.
4.  **Semantic Search**: When you enter a question, its text is embedded and compared against all indexed chunk vectors using cosine similarity:
    $$\text{Similarity}(q, c) = \frac{q \cdot c}{\|q\| \|c\|}$$
    The top 3 highest-scoring chunks are selected.
5.  **Prompt Formulation**: The 3 retrieved chunks are injected as structured context into a prompt. Strict grounding rules are appended, directing the model to output a clear refusal if the answers are not covered by the context.
6.  **Answer Generation**: The prompt is processed by `gemini-1.5-flash`, and the grounded response is rendered in the chat UI alongside source badges.
<img width="1364" height="418" alt="image" src="https://github.com/user-attachments/assets/bdd98a0c-6aa3-46cd-832f-b0f0190ffc66" />
<img width="1366" height="325" alt="image" src="https://github.com/user-attachments/assets/ea32136c-d656-4195-9a70-6a71135d33be" />
<img width="1363" height="479" alt="image" src="https://github.com/user-attachments/assets/0253ac49-1645-4ffc-a5d9-c8761ac3e6dd" />
<img width="1364" height="418" alt="image" src="https://github.com/user-attachments/assets/a5ddad84-a06d-4afe-b578-d3cc8b7d0786" />



