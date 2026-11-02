
import streamlit as st
import os
import json

from utils.helper import extract_video_id
from utils.chunking import TranscriptChunker
from utils.embeddings import EmbeddingGenerator
from utils.vector_db import VectorDatabase
from utils.rag import RAG
from utils.youtube_loader import YouTubeLoader
from dataset_dashboard import show_dataset_dashboard
from rag_evaluation import evaluate_questions
from model_comparison import compare_models


# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------
st.set_page_config(
    page_title="YouTube RAG Chatbot",
    page_icon="🎥",
    layout="wide"
)

st.title("🎥 YouTube Video RAG Chatbot")
st.markdown(
    "Ask questions about any YouTube video using **RAG + FAISS + Ollama**."
)

st.divider()


# ---------------------------------------------------
# Session State Initialization
# ---------------------------------------------------
defaults = {
    "db": None,
    "embedder": None,
    "rag": None,
    "chunks": None,
    "video_loaded": False,
    "transcript_path": None,
    "chat_history": []
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ---------------------------------------------------
# Sidebar
# ---------------------------------------------------
st.sidebar.title("⚙️ Settings")
st.sidebar.success("Ollama connection will be checked when generating an answer.")

st.sidebar.write("Embedding Model")
st.sidebar.info("all-MiniLM-L6-v2")

st.sidebar.write("LLM")
st.sidebar.info(os.getenv("OLLAMA_MODEL", "phi3:mini"))

st.sidebar.divider()
st.sidebar.write("Project")

st.sidebar.markdown("""
- Transcript Extraction
- Chunking
- Embeddings
- FAISS Search
- Ollama LLM
- Dataset Dashboard
- RAG Evaluation (planned)
""")


# ---------------------------------------------------
# Load YouTube Video
# ---------------------------------------------------
youtube_url = st.text_input("Paste YouTube URL")

if st.button("📥 Load Video"):
    if not youtube_url.strip():
        st.error("Please enter a YouTube URL.")
        st.stop()

    try:
        with st.spinner("Downloading transcript..."):
            loader = YouTubeLoader()
            transcript_path = loader.save_transcript(youtube_url)

        video_id = extract_video_id(youtube_url)
        st.success("Transcript downloaded!")

        with st.spinner("Creating chunks..."):
            chunker = TranscriptChunker()
            chunks = chunker.create_chunks(
                transcript_path,
                video_id
            )

        if not chunks:
            st.error("No transcript chunks were created.")
            st.stop()

        st.success(f"{len(chunks)} chunks created.")

        with st.spinner("Loading embedding model..."):
            embedder = EmbeddingGenerator()

        with st.spinner("Generating embeddings..."):
            embeddings = embedder.embed_chunks(chunks)

        st.success("Embeddings generated.")

        db = VectorDatabase()
        db.build_index(embeddings, chunks)
        db.save()

        rag = RAG()

        # Use spaces consistently for Python indentation.
        st.session_state.db = db
        st.session_state.embedder = embedder
        st.session_state.rag = rag
        st.session_state.chunks = chunks
        st.session_state.transcript_path = transcript_path
        st.session_state.video_loaded = True
        st.session_state.chat_history = []

        st.success("Video is ready for chatting!")

    except Exception as e:
        st.error(f"Could not load video: {e}")


# ---------------------------------------------------
# Ask Questions
# ---------------------------------------------------
st.divider()
st.header("💬 Ask Questions")

if st.session_state.video_loaded:
    question = st.text_input(
        "Enter your question",
        key="main_question"
    )

    if st.button("🚀 Ask"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            try:
                with st.spinner("Searching relevant information..."):
                    query_embedding = (
                        st.session_state.embedder.embed_query(question)
                    )
                    retrieved_chunks = st.session_state.db.search(
                        query_embedding,
                        top_k=4
                    )

                with st.spinner("Generating answer with Ollama..."):
                    answer = st.session_state.rag.answer(
                        retrieved_chunks,
                        question
                    )

                st.subheader("🤖 Answer")
                st.write(answer)

                st.subheader("📚 Retrieved Sources")

                if not retrieved_chunks:
                    st.info("No relevant transcript chunks were retrieved.")

                for i, chunk in enumerate(retrieved_chunks, start=1):
                    start_time = float(chunk.get("start_time", 0))
                    minutes = int(start_time // 60)
                    seconds = int(start_time % 60)

                    with st.expander(
                        f"Source {i} ({minutes}:{seconds:02d})"
                    ):
                        st.write(chunk.get("text", ""))

                        video_id = chunk.get("video_id", "")
                        if video_id:
                            youtube_link = (
                                f"https://www.youtube.com/watch?"
                                f"v={video_id}&t={int(start_time)}s"
                            )
                            st.markdown(
                                f"[▶ Open at {minutes}:{seconds:02d}]"
                                f"({youtube_link})"
                            )

                st.session_state.chat_history.append({
                    "question": question,
                    "answer": answer
                })

            except Exception as e:
                st.error(f"Could not answer the question: {e}")
else:
    st.info("Please load a YouTube video first.")


# ---------------------------------------------------
# Conversation History
# ---------------------------------------------------
st.divider()
st.subheader("💬 Conversation")

if st.session_state.chat_history:
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(chat["question"])

        with st.chat_message("assistant"):
            st.write(chat["answer"])
else:
    st.info("No conversation yet.")


# ---------------------------------------------------
# Dataset & Preprocessing Dashboard
# ---------------------------------------------------
st.divider()
st.header("📊 Dataset & Preprocessing")

if st.session_state.video_loaded:
    with st.expander(
        "Open Dataset Dashboard",
        expanded=False
    ):
        show_dataset_dashboard(
            st.session_state.transcript_path,
            st.session_state.chunks,
            st.session_state.db
        )
else:
    st.info("Load a video to view its dataset statistics.")


# ---------------------------------------------------
# Sidebar Actions
# ---------------------------------------------------
st.sidebar.divider()

if st.sidebar.button("🗑 Clear Chat", use_container_width=True):
    st.session_state.chat_history = []
    st.rerun()

if st.sidebar.button("♻ Reset Video", use_container_width=True):
    st.session_state.video_loaded = False
    st.session_state.db = None
    st.session_state.embedder = None
    st.session_state.rag = None
    st.session_state.chunks = None
    st.session_state.transcript_path = None
    st.session_state.chat_history = []
    st.rerun()


# ---------------------------------------------------
# Footer
# ---------------------------------------------------
st.divider()
st.caption(
    "YouTube RAG Chatbot | Streamlit, FAISS, "
    "Sentence Transformers and Ollama"
)



# ---------------------------------------------------
# RAG Evaluation Dashboard
# ---------------------------------------------------
st.divider()
st.header("RAG Evaluation Dashboard")

if not st.session_state.video_loaded:
    st.info("Load a YouTube video before running evaluation.")
else:
    st.write(
        "Evaluate the chatbot using questions and expected answers "
        "based on the currently loaded video."
    )

    default_cases = [
        {
            "question": "What is the main topic of this video?",
            "expected_answer": "",
            "evidence_text": ""
        },
        {
            "question": "What key point does the speaker explain?",
            "expected_answer": "",
            "evidence_text": ""
        }
    ]

    if "evaluation_cases_text" not in st.session_state:
        st.session_state.evaluation_cases_text = json.dumps(
            default_cases, indent=2
        )

    st.caption(
        "Edit the JSON below. Fill in expected_answer using the actual "
        "transcript. evidence_text is optional but helps assess retrieval."
    )

    cases_text = st.text_area(
        "Evaluation test cases (JSON)",
        key="evaluation_cases_text",
        height=260
    )

    if st.button("Run RAG Evaluation"):
        try:
            test_cases = json.loads(cases_text)

            if not isinstance(test_cases, list) or not test_cases:
                st.error("Enter a non-empty JSON list of test cases.")
            elif any(
                not case.get("question", "").strip()
                or not case.get("expected_answer", "").strip()
                for case in test_cases
            ):
                st.error(
                    "Every test case needs a question and expected_answer."
                )
            else:
                with st.spinner(
                    "Evaluating questions. This may take a few minutes..."
                ):
                    results_df = evaluate_questions(
                        test_cases=test_cases,
                        embedder=st.session_state.embedder,
                        db=st.session_state.db,
                        rag=st.session_state.rag,
                        top_k=4
                    )

                if results_df.empty:
                    st.warning("No valid evaluation results were produced.")
                else:
                    st.session_state.evaluation_results = results_df

        except json.JSONDecodeError as error:
            st.error(f"Invalid JSON: {error}")
        except Exception as error:
            st.error(f"Evaluation failed: {error}")

    if "evaluation_results" in st.session_state:
        results_df = st.session_state.evaluation_results

        st.subheader("Evaluation Summary")

        col1, col2, col3 = st.columns(3)
        col1.metric("Questions Evaluated", len(results_df))
        col2.metric(
            "Mean Answer Token Recall",
            f"{results_df['answer_token_recall'].mean():.1%}"
        )
        col3.metric(
            "Average Response Time",
            f"{results_df['response_time_seconds'].mean():.2f} sec"
        )

        st.caption(
            "Token recall measures word overlap with expected answers; "
            "it is not a factual accuracy score."
        )

        if results_df["retrieval_hit"].notna().any():
            hit_rate = results_df["retrieval_hit"].dropna().mean()
            st.metric("Lexical Retrieval Hit Rate", f"{hit_rate:.1%}")

        st.subheader("Results by Question")
        st.dataframe(results_df, use_container_width=True)

        chart_df = results_df[
            ["question", "answer_token_recall"]
        ].set_index("question")

        st.subheader("Answer Token Recall Chart")
        st.bar_chart(chart_df)

        st.download_button(
            "Download Evaluation Results CSV",
            data=results_df.to_csv(index=False).encode("utf-8"),
            file_name="rag_evaluation_results.csv",
            mime="text/csv"
        )


# ---------------------------------------------------
# Model Comparison Dashboard
# ---------------------------------------------------
st.divider()
st.header("Model Comparison Dashboard")

st.write(
    "Compare Phi-3 Small Context and Qwen 2.5 0.5B "
    "using the same questions and retrieved transcript chunks."
)

if not st.session_state.video_loaded:
    st.info("Load a YouTube video before comparing models.")
else:
    st.caption(
        "The comparison uses evaluation_dataset.csv. "
        "Verify the expected answers and evidence against the transcript."
    )

    if st.button("Run Model Comparison"):
        try:
            import pandas as pd

            dataset_path = os.path.join(
                os.path.dirname(__file__),
                "evaluation_dataset.csv"
            )

            test_df = pd.read_csv(dataset_path)

            required_columns = {
                "question", "expected_answer", "answerable", "evidence_text"
            }

            if not required_columns.issubset(test_df.columns):
                st.error(
                    "The evaluation CSV is missing required columns."
                )
            elif test_df.empty:
                st.error("The evaluation dataset is empty.")
            else:
                test_cases = test_df.fillna("").to_dict("records")

                with st.spinner(
                    "Comparing models. This may take several minutes..."
                ):
                    detail_df, summary_df = compare_models(
                        test_cases=test_cases,
                        embedder=st.session_state.embedder,
                        db=st.session_state.db,
                        top_k=4
                    )

                st.session_state.model_comparison_details = detail_df
                st.session_state.model_comparison_summary = summary_df

        except Exception as error:
            st.error(f"Model comparison failed: {error}")

    if "model_comparison_summary" in st.session_state:
        summary_df = st.session_state.model_comparison_summary
        detail_df = st.session_state.model_comparison_details

        st.subheader("Model Metrics")
        st.dataframe(summary_df, use_container_width=True)

        st.subheader("Precision, Recall and F1")
        metric_columns = ["model", "precision", "recall", "f1"]

        if all(column in summary_df.columns for column in metric_columns):
            metrics_chart = summary_df.set_index("model")[
                ["precision", "recall", "f1"]
            ]
            st.bar_chart(metrics_chart)

        st.subheader("Average Response Time")
        if "average_response_time_seconds" in summary_df.columns:
            time_chart = summary_df.set_index("model")[
                ["average_response_time_seconds"]
            ]
            st.bar_chart(time_chart)

        st.subheader("Detailed Results")
        st.dataframe(detail_df, use_container_width=True)

        st.download_button(
            "Download Model Summary CSV",
            data=summary_df.to_csv(index=False).encode("utf-8"),
            file_name="model_comparison_summary.csv",
            mime="text/csv"
        )

        st.download_button(
            "Download Detailed Results CSV",
            data=detail_df.to_csv(index=False).encode("utf-8"),
            file_name="model_comparison_details.csv",
            mime="text/csv"
        )

        st.warning(
            "Precision, recall and F1 evaluate answerable versus "
            "unanswerable predictions based on fallback behavior. "
            "They do not independently verify factual correctness."
        )
