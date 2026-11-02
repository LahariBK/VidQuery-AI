
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from config import EMBEDDING_MODEL, CHUNK_SIZE


def show_dataset_dashboard(transcript_path, chunks, db=None):
    st.header("Dataset & Preprocessing Dashboard")

    if not transcript_path or not Path(transcript_path).exists():
        st.info("Load a YouTube video to view its dataset.")
        return

    with open(transcript_path, "r", encoding="utf-8") as file:
        transcript = json.load(file)

    if not transcript:
        st.warning("The transcript is empty.")
        return

    texts = [
        str(entry.get("text", "")).strip()
        for entry in transcript
    ]

    total_words = sum(len(text.split()) for text in texts)
    total_characters = sum(len(text) for text in texts)

    duration = max(
        float(entry.get("end", 0))
        for entry in transcript
    )

    language = transcript[0].get("language", "Unknown")

    st.subheader("Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Transcript Segments", len(transcript))
    col2.metric("Total Words", total_words)
    col3.metric("Total Chunks", len(chunks))
    col4.metric("Video Duration", f"{duration / 60:.1f} min")

    st.write("Transcript Language:", language)
    st.write("Total Characters:", total_characters)
    st.write("Embedding Model:", EMBEDDING_MODEL)
    st.write("Configured Chunk Size:", CHUNK_SIZE)
    st.caption(
        "Chunk size is a character threshold. "
        "Actual chunks may be longer because transcript "
        "segments are kept intact. No chunk overlap is "
        "currently implemented."
    )

    if db is not None and db.index is not None:
        st.write("Embedding Dimensions:", db.index.d)
        st.write("Indexed Vectors:", db.index.ntotal)

    st.divider()

    st.subheader("Chunk Size Distribution")

    chunk_lengths = [len(chunk["text"]) for chunk in chunks]

    if chunk_lengths:
        chart_data = pd.DataFrame({
            "Chunk": [f"Chunk {i + 1}" for i in range(len(chunks))],
            "Characters": chunk_lengths
        })

        st.bar_chart(
            chart_data.set_index("Chunk")
        )

        st.metric(
            "Average Chunk Length",
            f"{sum(chunk_lengths) / len(chunk_lengths):.1f} characters"
        )

    st.divider()

    st.subheader("Transcript Preview")

    transcript_df = pd.DataFrame(transcript)

    st.dataframe(
        transcript_df,
        use_container_width=True
    )

    st.download_button(
        "Download Transcript CSV",
        data=transcript_df.to_csv(index=False).encode("utf-8"),
        file_name="youtube_transcript.csv",
        mime="text/csv"
    )

    st.divider()

    st.subheader("Processed Chunks")

    chunks_df = pd.DataFrame(chunks)

    st.dataframe(
        chunks_df,
        use_container_width=True
    )

    st.download_button(
        "Download Chunks CSV",
        data=chunks_df.to_csv(index=False).encode("utf-8"),
        file_name="youtube_chunks.csv",
        mime="text/csv"
    )
