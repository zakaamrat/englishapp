import re
from datetime import datetime
from pathlib import Path
import streamlit as st
import language_tool_python
from faster_whisper import WhisperModel
from audiorecorder import audiorecorder
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import A4

# =========================
# Cloud-safe directories
# =========================
BASE_DIR = Path.cwd()
REPORT_DIR = BASE_DIR / "reports"
AUDIO_DIR = BASE_DIR / "audio"
REPORT_DIR.mkdir(exist_ok=True)
AUDIO_DIR.mkdir(exist_ok=True)

# =========================
# Cached resources
# =========================
@st.cache_resource
def get_lang_tool():
    return language_tool_python.LanguageTool("en-US")  # Local server mode

@st.cache_resource
def get_whisper():
    return WhisperModel("base", device="cpu", compute_type="int8")

# =========================
# Grammar Analysis
# =========================
def analyze_text(text):
    tool = get_lang_tool()
    matches = tool.check(text)
    corrected = language_tool_python.utils.correct(text, matches)

    issues = []
    for m in matches:
        issues.append({
            "Message": m.message,
            "Suggestion": m.replacements[0] if m.replacements else "",
            "Context": text[max(0, m.offset-30):m.offset+m.errorLength+30]
        })
    return corrected, issues

# =========================
# PDF Generator
# =========================
def generate_pdf(filename, title, lines):
    pdf_path = REPORT_DIR / filename
    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4)
    elements = []
    style = ParagraphStyle(name='Normal', fontSize=10)

    elements.append(Paragraph(f"<b>{title}</b>", style))
    elements.append(Spacer(1, 12))

    for line in lines:
        elements.append(Paragraph(line, style))
        elements.append(Spacer(1, 6))

    doc.build(elements)
    return pdf_path

# =========================
# UI
# =========================
st.set_page_config(page_title="English Evaluator", layout="wide")
st.title("English Writing & Speaking Evaluator")

mode = st.radio("Select Mode", ["Writing", "Speaking"])

name = st.text_input("Student Name")
email = st.text_input("Email")

if not name or not email:
    st.warning("Please enter your name and email.")
    st.stop()

# =========================
# WRITING MODE
# =========================
if mode == "Writing":

    text = st.text_area("Enter your paragraph", height=200)

    if st.button("Evaluate Writing"):

        if len(text.strip()) < 30:
            st.error("Please write a longer paragraph.")
            st.stop()

        corrected, issues = analyze_text(text)

        st.subheader("Corrected Version")
        st.write(corrected)

        st.subheader("Detected Issues")
        if issues:
            st.dataframe(issues, use_container_width=True)
        else:
            st.success("No major issues detected.")

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
                corrected
            ]
        )

        with open(pdf_path, "rb") as f:
            st.download_button("Download PDF Report", f, file_name=pdf_name)

# =========================
# SPEAKING MODE (Browser Recording)
# =========================
else:

    st.subheader("Record your voice")

    audio = audiorecorder("Click to record", "Click to stop")

    if len(audio) > 0:

        wav_path = AUDIO_DIR / "recorded.wav"
        audio.export(wav_path, format="wav")

        st.audio(wav_path)

        with st.spinner("Transcribing..."):
            model = get_whisper()
            segments, info = model.transcribe(str(wav_path))
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
                "Corrected Version:",
                corrected
            ]
        )

        with open(pdf_path, "rb") as f:
            st.download_button("Download PDF Report", f, file_name=pdf_name)
