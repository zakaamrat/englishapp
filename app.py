import os
import re
from datetime import datetime
from pathlib import Path
import numpy as np
import streamlit as st
import soundfile as sf
import language_tool_python
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import cm
from faster_whisper import WhisperModel

# =========================
# Paths (Cloud-safe)
# =========================
BASE_DIR = Path.cwd()
REPORT_DIR = BASE_DIR / "reports"
AUDIO_DIR = BASE_DIR / "audio"
REPORT_DIR.mkdir(exist_ok=True)
AUDIO_DIR.mkdir(exist_ok=True)

# =========================
# Cached Resources
# =========================
@st.cache_resource
def get_lang_tool():
    # Public API mode (NO Java needed)
    return language_tool_python.LanguageToolPublicAPI("en-US")

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
# PDF Generator (Cloud safe)
# =========================
def generate_pdf(filename, title, content_lines):
    pdf_path = REPORT_DIR / filename
    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4)
    elements = []
    style = ParagraphStyle(name='Normal', fontSize=10)

    elements.append(Paragraph(f"<b>{title}</b>", style))
    elements.append(Spacer(1, 12))

    for line in content_lines:
        elements.append(Paragraph(line, style))
        elements.append(Spacer(1, 6))

    doc.build(elements)
    return pdf_path

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="English Evaluator", layout="wide")
st.title("English Writing & Speaking Evaluator")

mode = st.radio("Choose Mode", ["Writing", "Speaking"])

name = st.text_input("Student Name")
email = st.text_input("Email")

if not name or not email:
    st.stop()

# =========================
# Writing Mode
# =========================
if mode == "Writing":
    text = st.text_area("Enter your paragraph", height=200)

    if st.button("Evaluate Writing"):
        corrected, issues = analyze_text(text)

        st.subheader("Corrected Version")
        st.write(corrected)

        st.subheader("Detected Issues")
        if issues:
            st.dataframe(issues)
        else:
            st.success("No major issues detected.")

        pdf_name = f"writing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        pdf_path = generate_pdf(
            pdf_name,
            "Writing Evaluation Report",
            [f"Student: {name}", f"Email: {email}", "", "Corrected Text:", corrected]
        )

        with open(pdf_path, "rb") as f:
            st.download_button("Download PDF Report", f, file_name=pdf_name)

# =========================
# Speaking Mode (Upload-based)
# =========================
else:
    uploaded_audio = st.file_uploader("Upload WAV file", type=["wav"])

    if uploaded_audio is not None:
        wav_path = AUDIO_DIR / uploaded_audio.name
        with open(wav_path, "wb") as f:
            f.write(uploaded_audio.read())

        st.audio(str(wav_path))

        with st.spinner("Transcribing..."):
            model = get_whisper()
            segments, info = model.transcribe(str(wav_path))
            transcript = " ".join([seg.text for seg in segments])

        st.subheader("Transcript")
        st.write(transcript)

        corrected, issues = analyze_text(transcript)

        st.subheader("Corrected Transcript")
        st.write(corrected)

        pdf_name = f"speaking_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        pdf_path = generate_pdf(
            pdf_name,
            "Speaking Evaluation Report",
            [f"Student: {name}", f"Email: {email}", "", "Transcript:", transcript, "", "Corrected:", corrected]
        )

        with open(pdf_path, "rb") as f:
            st.download_button("Download PDF Report", f, file_name=pdf_name)