import re
import os
from datetime import datetime
from pathlib import Path

import streamlit as st
from spellchecker import SpellChecker
from textblob import TextBlob
from faster_whisper import WhisperModel
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import A4

# =====================================================
# CONFIGURATION
# =====================================================

st.set_page_config(page_title="English Evaluator", layout="wide")

BASE_DIR = Path.cwd()
REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(exist_ok=True)

spell = SpellChecker()

# =====================================================
# CACHED MODELS
# =====================================================

@st.cache_resource
def load_whisper():
    return WhisperModel("base", device="cpu", compute_type="int8")

# =====================================================
# TEXT ANALYSIS FUNCTION
# =====================================================

def analyze_text(text):

    # ---------- SPELLING ----------
    words = re.findall(r"\b\w+\b", text)
    misspelled = spell.unknown(words)

    spelling_issues = []
    for word in misspelled:
        spelling_issues.append({
            "Type": "Spelling",
            "Original": word,
            "Suggestion": spell.correction(word)
        })

    # ---------- GRAMMAR ----------
    blob = TextBlob(text)
    corrected_text = str(blob.correct())

    grammar_issues = []
    if corrected_text != text:
        grammar_issues.append({
            "Type": "Grammar",
            "Original": text,
            "Suggestion": corrected_text
        })

    issues = spelling_issues + grammar_issues

    return corrected_text, issues


# =====================================================
# PDF GENERATOR
# =====================================================

def generate_pdf(filename, title, lines):
    pdf_path = REPORT_DIR / filename
    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4)
    elements = []
    style = ParagraphStyle(name='Normal', fontSize=10)

    elements.append(Paragraph(f"<b>{title}</b>", style))
    elements.append(Spacer(1, 12))

    for line in lines:
        elements.append(Paragraph(str(line), style))
        elements.append(Spacer(1, 6))

    doc.build(elements)
    return pdf_path


# =====================================================
# UI
# =====================================================

st.title("AI-Powered English Writing & Speaking Evaluator")

mode = st.radio("Select Evaluation Type", ["Writing Evaluation", "Speaking Evaluation"])

name = st.text_input("Student Name")
email = st.text_input("Email")

if not name or not email:
    st.warning("Please enter your name and email.")
    st.stop()

# =====================================================
# WRITING MODE
# =====================================================

if mode == "Writing Evaluation":

    st.subheader("Write about one of the following topics:")
    st.markdown("- Oman Vision 2040\n- Oman Culture\n- GCC Tourism\n- Omani Universities\n- Education in Oman")

    text = st.text_area("Enter your paragraph here:", height=200)

    if st.button("Evaluate Writing"):

        if len(text.strip()) < 30:
            st.error("Please write at least 30 words.")
            st.stop()

        corrected, issues = analyze_text(text)

        st.subheader("Corrected Version")
        st.write(corrected)

        st.subheader("Detected Issues")
        if issues:
            st.dataframe(issues, use_container_width=True)
        else:
            st.success("No major issues detected.")

        recommendations = [
            "Practice sentence structure using short paragraphs.",
            "Review subject-verb agreement rules.",
            "Read your paragraph aloud to improve fluency.",
            "Focus on vocabulary variation."
        ]

        st.subheader("Recommendations")
        for r in recommendations:
            st.write("- ", r)

        pdf_name = f"writing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

        pdf_path = generate_pdf(
            pdf_name,
            "Writing Evaluation Report",
            [
                f"Student: {name}",
                f"Email: {email}",
                "",
                "Original Text:",
                text,
                "",
                "Corrected Text:",
                corrected,
                "",
                "Total Issues: " + str(len(issues))
            ]
        )

        with open(pdf_path, "rb") as f:
            st.download_button("Download PDF Report", f, file_name=pdf_name)


# =====================================================
# SPEAKING MODE (File Upload Instead of Browser Recording)
# =====================================================

else:

    st.subheader("Upload your speaking recording (.wav or .mp3)")

    audio_file = st.file_uploader("Upload Audio File", type=["wav", "mp3"])

    if audio_file is not None:

        temp_audio_path = BASE_DIR / "temp_audio.wav"

        with open(temp_audio_path, "wb") as f:
            f.write(audio_file.read())

        st.audio(audio_file)

        with st.spinner("Transcribing speech..."):
            model = load_whisper()
            segments, info = model.transcribe(str(temp_audio_path))
            transcript = " ".join([seg.text for seg in segments])

        st.subheader("Transcript")
        st.write(transcript)

        corrected, issues = analyze_text(transcript)

        st.subheader("Corrected Transcript")
        st.write(corrected)

        if issues:
            st.subheader("Detected Issues")
            st.dataframe(issues, use_container_width=True)
        else:
            st.success("No major issues detected.")

        # Basic Speaking Metrics
        word_count = len(transcript.split())
        duration_est = word_count / 2.5  # rough estimation
        wpm = word_count / (duration_est / 60)

        st.subheader("Speaking Metrics")
        st.write(f"Words: {word_count}")
        st.write(f"Estimated WPM: {round(wpm,1)}")

        recommendations = [
            "Slow down slightly and pause between sentences.",
            "Practice pronunciation of difficult words.",
            "Record yourself daily for fluency improvement."
        ]

        st.subheader("Recommendations")
        for r in recommendations:
            st.write("- ", r)

        pdf_name = f"speaking_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

        pdf_path = generate_pdf(
            pdf_name,
            "Speaking Evaluation Report",
            [
                f"Student: {name}",
                f"Email: {email}",
                "",
                "Transcript:",
                transcript,
                "",
                "Corrected Transcript:",
                corrected,
                "",
                f"Word Count: {word_count}",
                f"Estimated WPM: {round(wpm,1)}"
            ]
        )

        with open(pdf_path, "rb") as f:
            st.download_button("Download PDF Report", f, file_name=pdf_name)


