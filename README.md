# 🎥 VidQuery AI – YouTube Video RAG Chatbot

**Ask questions. Retrieve context. Discover knowledge from YouTube videos.**

VidQuery AI is an intelligent Retrieval-Augmented Generation (RAG) chatbot that enables users to ask natural-language questions about YouTube videos and receive context-based answers.

The application extracts available YouTube transcripts, divides them into manageable text chunks, generates semantic embeddings using Sentence Transformers, and stores the embeddings in a FAISS vector index. When a user asks a question, the system retrieves relevant transcript passages and uses a locally hosted Large Language Model (LLM) through Ollama to generate an answer.

The application also provides transcript analysis, dataset visualization, model comparison, and evaluation capabilities.

---

## 🚀 Features

- 📺 Load YouTube videos using their URLs.
- 🌍 Process available transcript languages, depending on video transcript availability.
- 📝 Automatic YouTube transcript extraction.
- ✂️ Transcript chunking and preprocessing.
- 🧠 Semantic embeddings using Sentence Transformers.
- 🔍 Similarity-based retrieval using FAISS.
- 🤖 Local LLM inference using Ollama.
- 💬 Interactive chatbot interface built with Streamlit.
- ⏱️ Retrieved transcript sources with timestamps.
- 📊 Dataset and preprocessing dashboard.
- 📈 Model comparison between Phi-3 Small Context and Qwen 2.5 0.5B.
- 📋 RAG evaluation using labelled question-answer examples.
- 📉 Performance analysis using Precision, Recall, F1-score, and response time.
- 🔎 Manual grounding review to examine whether generated answers are supported by retrieved evidence.

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Frontend | Streamlit |
| Transcript Extraction | youtube-transcript-api |
| Embedding Model | all-MiniLM-L6-v2 |
| Embedding Framework | Sentence Transformers |
| Vector Database | FAISS |
| Language Model Runtime | Ollama |
| Language Models | Phi-3 Small Context, Qwen 2.5 0.5B |
| Data Processing | Pandas, NumPy |
| Retrieval Technique | Dense Vector Similarity Search |
| NLP Architecture | Retrieval-Augmented Generation (RAG) |

---

## 🏗️ System Architecture

```text
                 YOUTUBE VIDEO URL
                         |
                         v
               TRANSCRIPT EXTRACTION
                         |
                         v
                 TEXT CHUNKING
                         |
                         v
              SENTENCE TRANSFORMERS
                 EMBEDDING MODEL
                         |
                         v
                  FAISS INDEX
                         |
              ---------------------
              |                   |
              v                   |
         USER QUESTION            |
              |                   |
              v                   |
         QUERY EMBEDDING          |
              |                   |
              v                   |
        FAISS SIMILARITY SEARCH   |
              |                   |
              v                   |
       TOP-4 RETRIEVED CHUNKS <----
              |
              v
       RETRIEVED CONTEXT
              |
              v
       OLLAMA LANGUAGE MODEL
              |
              v
        GENERATED ANSWER
              |
              v
     ANSWER + SOURCE TIMESTAMPS


       SEPARATE EVALUATION WORKFLOW
                    |
                    v
        LABELLED EVALUATION DATASET
                    |
                    v
          MODEL COMPARISON
                    |
                    v
       PRECISION / RECALL / F1
                    |
                    v
           RESPONSE TIME
                    |
                    v
          GROUNDING REVIEW
```

---

## 📂 Project Structure

```text
YouTube-Video-RAG-Chatbot/
│
├── app.py
├── config.py
├── requirements.txt
│
├── models/
│   └── llm.py
│
├── utils/
│   ├── youtube_loader.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── vector_db.py
│   └── rag.py
│
├── vectorstore/
│   ├── faiss.index
│   └── chunks.pkl
│
├── data/
│
├── dataset_dashboard.py
├── rag_evaluation.py
├── model_comparison.py
├── evaluation_dataset.csv
│
└── README.md
```

*Note: The structure above highlights the main project modules. Update it if your local folder contains additional files or different folder names.*

---

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd YouTube-Video-RAG-Chatbot
```

### 2. Create a virtual environment

**Windows:**

```powershell
python -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Ollama

Download Ollama from:

https://ollama.com/

Pull the language models used in the project:

```bash
ollama pull phi3-smallctx:latest
ollama pull qwen2.5:0.5b
```

Ensure that the Ollama service is running before launching the application.

### 5. Configure the language model

For Windows PowerShell:

```powershell
$env:OLLAMA_MODEL = "phi3-smallctx:latest"
```

### 6. Run the application

```bash
streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

---

## 🔄 How VidQuery AI Works

**Step 1: Video Input**

The user provides a YouTube video URL through the Streamlit interface.

**Step 2: Transcript Extraction**

The application extracts the available transcript and associated timing information.

**Step 3: Text Chunking**

The transcript is divided into smaller text chunks to support efficient retrieval.

**Step 4: Embedding Generation**

The all-MiniLM-L6-v2 Sentence Transformer converts transcript chunks into 384-dimensional numerical vectors.

**Step 5: Vector Indexing**

The generated embeddings are stored in a FAISS IndexFlatL2 vector index.

**Step 6: Question Processing**

The user's question is converted into an embedding using the same embedding model.

**Step 7: Semantic Retrieval**

FAISS retrieves the four nearest transcript chunks based on squared Euclidean distance.

**Step 8: Answer Generation**

The retrieved context and user question are passed to the selected Ollama language model.

**Step 9: Source Display**

The chatbot displays the generated response along with retrieved transcript sources and timestamps.

---

## 📊 Model Evaluation and Comparison

VidQuery AI includes an evaluation framework to compare two locally hosted language models:

- **Phi-3 Small Context:** `phi3-smallctx:latest`
- **Qwen 2.5 0.5B:** `qwen2.5:0.5b`

Both models are evaluated using the same questions and retrieved context to support a fair comparison.

### Evaluation Metrics

| Metric | Purpose |
|---|---|
| Precision | Measures the correctness of answerable predictions |
| Recall | Measures how many answerable questions are identified |
| F1-score | Combines precision and recall |
| Average Response Time | Measures model response speed |
| Answer Token Recall | Measures overlap between expected and generated answer tokens |
| Grounding Review | Examines whether generated responses are supported by retrieved evidence |

The evaluation dashboard also provides detailed results, generated responses, and model comparison visualizations.

*The current evaluation dataset is small and provides preliminary experimental results rather than a definitive assessment of general performance.*

---

## 📸 Application Screenshots

Add your actual application screenshots to a folder named `screenshots/` in the repository.

Suggested images:

1. Home page and YouTube URL input.
2. Dataset and preprocessing dashboard.
3. Model comparison metrics.
4. Average response-time comparison.
5. Chatbot question, answer, and retrieved sources.
6. Unanswerable question and fallback response.
7. Detailed model comparison results.

After adding the screenshots, you can display them using:

```markdown
![VidQuery AI Home Page](screenshots/01_home_page.png)

![Dataset Dashboard](screenshots/02_dataset_preprocessing.png)

![Model Metrics](screenshots/03_model_metrics.png)

![Response Time](screenshots/04_response_time.png)

![Question Answering](screenshots/05_question_answer_sources.png)
```

---

## 🔮 Future Improvements

- 🎙️ Speech-to-text transcription for videos without available transcripts.
- 🌐 Expanded multilingual question answering and translation.
- 📄 Automatic video summary and PDF report generation.
- 💾 Improved persistent vector storage.
- 🎞️ Multi-video question answering.
- 🎯 Improved retrieval and context-ranking strategies.
- 🔎 Automated answer-grounding evaluation.
- 🖥️ Enhanced user interface and accessibility.

---

## 👩‍💻 Author

**Lahari B K**

B.E. Artificial Intelligence and Machine Learning  
B.M.S. College of Engineering, Bengaluru

**Project:** VidQuery AI – YouTube Video RAG Chatbot

**Technologies:** Python | NLP | RAG | FAISS | Sentence Transformers | Ollama | Streamlit

---

## 📚 References

- [Sentence Transformers](https://www.sbert.net/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Ollama](https://ollama.com/)
- [Streamlit](https://streamlit.io/)
- [Hugging Face](https://huggingface.co/)
- [YouTube Transcript API](https://pypi.org/project/youtube-transcript-api/)

