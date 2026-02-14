import re
from datetime import datetime
from pathlib import Path

import streamlit as st
from spellchecker import SpellChecker
from faster_whisper import WhisperModel
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import A4
import requests
from streamlit_lottie import st_lottie

def load_lottie_url(url: str):
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None

# =====================================================
# CONFIG
# =====================================================
st.set_page_config(page_title="English Evaluator", layout="wide")

BASE_DIR = Path.cwd()
REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(exist_ok=True)

spell = SpellChecker(distance=1)

TOPICS = [
    "Oman culture",
    "Oman tourism",
    "Oman Vision 2040",
    "GCC cooperation",
    "Oman universities",
    "Omani schools and education",
]

PROMPTS = {
    "Oman culture": "Write/speak about Oman culture: traditions, food, and hospitality.",
    "Oman tourism": "Write/speak about Oman tourism: places to visit and why they are special.",
    "Oman Vision 2040": "Write/speak about Oman Vision 2040: goals and how youth can contribute.",
    "GCC cooperation": "Write/speak about GCC cooperation: benefits in education and economy.",
    "Oman universities": "Write/speak about Oman universities: student life and learning opportunities.",
    "Omani schools and education": "Write/speak about schools in Oman and how education can improve.",
}

# =====================================================
# CACHE Whisper
# =====================================================
@st.cache_resource
def load_whisper():
    return WhisperModel("base", device="cpu", compute_type="int8")

# =====================================================
# UTIL
# =====================================================
def generate_pdf(filename, title, lines):
    pdf_path = REPORT_DIR / filename
    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4)
    style = ParagraphStyle(name="Normal", fontSize=10, leading=12)
    elements = [Paragraph(f"<b>{title}</b>", style), Spacer(1, 12)]
    for line in lines:
        elements.append(Paragraph(str(line).replace("\n", "<br/>"), style))
        elements.append(Spacer(1, 6))
    doc.build(elements)
    return pdf_path

def tokenize_with_positions(text):
    # words + punctuation tokens
    tokens = []
    for m in re.finditer(r"[A-Za-z']+|[0-9]+|[^\w\s]", text):
        tokens.append((m.group(0), m.start(), m.end()))
    return tokens

def is_probable_proper_noun(word, original_text):
    # preserve Oman, GCC, names, email-like tokens
    if word.isupper() and len(word) <= 6:
        return True
    if word[:1].isupper():
        return True
    if "@" in word:
        return True
    return False

def safe_spell_correct(text):
    """
    Correct ONLY obvious spelling mistakes.
    Preserve proper nouns and very short tokens.
    """
    tokens = tokenize_with_positions(text)
    words = [t[0] for t in tokens if re.fullmatch(r"[A-Za-z']+", t[0])]
    misspelled = spell.unknown([w.lower() for w in words])

    issues = []
    corrected_parts = list(text)

    # Replace from end to start to keep indices stable
    replacements = []
    for tok, s, e in tokens:
        if not re.fullmatch(r"[A-Za-z']+", tok):
            continue
        low = tok.lower()

        if low in misspelled:
            # preserve proper nouns and short words
            if is_probable_proper_noun(tok, text):
                continue
            if len(tok) <= 2:
                continue

            suggestion = spell.correction(low)
            # safety: only accept if suggestion exists and differs
            if suggestion and suggestion != low:
                replacements.append((s, e, suggestion))
                issues.append({
                    "Type": "Spelling",
                    "Original": tok,
                    "Suggestion": suggestion,
                    "Message": f"Possible misspelling: '{tok}' → '{suggestion}'"
                })

    # apply replacements backwards
    for s, e, rep in sorted(replacements, key=lambda x: x[0], reverse=True):
        corrected_parts[s:e] = list(rep)

    corrected_text = "".join(corrected_parts)

    return corrected_text, issues

def basic_grammar_checks(text):
    """
    Lightweight grammar flags (rule-based).
    Not a full grammar engine, but reliable and cloud-safe.
    """
    issues = []
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    joined = " ".join(lines)

    # 1) Sentence punctuation
    if joined and joined[-1] not in ".!?":
        issues.append({
            "Type": "Grammar",
            "Original": joined[-20:],
            "Suggestion": joined + ".",
            "Message": "Sentence may be missing ending punctuation."
        })

    # 2) Capitalization: first letter
    if joined and joined[0].islower():
        issues.append({
            "Type": "Grammar",
            "Original": joined[:20],
            "Suggestion": joined[:1].upper() + joined[1:],
            "Message": "Start sentences with a capital letter."
        })

    # 3) Common subject–verb agreement patterns (simple)
    # "Oman are" -> "Oman is"
    joined_low = joined.lower()
    for pattern, repl, msg in [
        (r"\boman are\b", "Oman is", "Subject–verb agreement: 'Oman is' not 'Oman are'."),
        (r"\b(today class is)\b", "Today's class is", "Use possessive form: 'Today's class'."),
        (r"\bi wnat\b", "I want", "Common misspelling/phrase correction: 'I want'."),
        (r"\bevry\b", "every", "Common misspelling: 'every'."),
        (r"\bcontry\b", "country", "Common misspelling: 'country'."),
    ]:
        if re.search(pattern, joined_low):
            issues.append({
                "Type": "Grammar",
                "Original": re.search(pattern, joined_low).group(0),
                "Suggestion": repl,
                "Message": msg
            })

    # 4) Location sanity (example: "Oman ... located in jordan")
    if re.search(r"\boman\b", joined_low) and re.search(r"\blocated in jordan\b", joined_low):
        issues.append({
            "Type": "Grammar",
            "Original": "located in jordan",
            "Suggestion": "located in the Middle East (on the Arabian Peninsula).",
            "Message": "Factual geography note: Oman is not located in Jordan."
        })

    return issues

def merge_issues(spell_issues, grammar_issues):
    return spell_issues + grammar_issues

def recommendations_from_issues(issues):
    spell_n = sum(1 for i in issues if i["Type"] == "Spelling")
    gram_n = sum(1 for i in issues if i["Type"] == "Grammar")

    recs = []
    if spell_n:
        recs.append("Keep a personal list of repeated spelling mistakes and practice them daily.")
    if gram_n:
        recs.append("Practice short sentences first, then combine them with linking words.")
    recs.append("After correction, rewrite the paragraph once without looking at the suggestions.")
    recs.append("Read aloud to improve sentence rhythm and confidence.")
    return recs[:6]

# =====================================================
# UI
# =====================================================
st.markdown("""
<style>
.typing {
  display: inline-block;
  padding: 10px 14px;
  border-radius: 18px;
  background: #f1f3f6;
  font-size: 14px;
}
.typing span {
  height: 8px; width: 8px;
  margin: 0 2px;
  background-color: #888;
  border-radius: 50%;
  display: inline-block;
  animation: bounce 1.2s infinite ease-in-out both;
}
.typing span:nth-child(2) { animation-delay: 0.2s; }
.typing span:nth-child(3) { animation-delay: 0.4s; }
@keyframes bounce {
  0%, 80%, 100% { transform: scale(0); }
  40% { transform: scale(1.0); }
}
</style>

<div class="typing">🤖 Assistant is analyzing...
  <span></span><span></span><span></span>
</div>
""", unsafe_allow_html=True)
col1, col2 = st.columns([2, 1])

with col1:
    st.title("English Writing & Speaking Evaluator (Cloud-Safe)")

with col2:
    lottie = load_lottie_url("https://assets9.lottiefiles.com/packages/lf20_touohxv0.json")
    if lottie:
        st_lottie(lottie, height=180, key="bot")
    else:
        st.info("🤖 Chatbot assistant ready!")

st.title("English Writing & Speaking Evaluator (Cloud-Safe)")

mode = st.radio("Select Evaluation Type", ["Writing Evaluation", "Speaking Evaluation"])

name = st.text_input("Student Name")
email = st.text_input("Email")

topic = st.selectbox("Choose a topic", TOPICS)
prompt = PROMPTS[topic]

if not name or not email:
    st.warning("Please enter your name and email.")
    st.stop()

st.subheader("Prompt")
st.write(prompt)


# =====================================================
# WRITING
# =====================================================
if mode == "Writing Evaluation":
    text = st.text_area("Enter your paragraph here:", height=220)

    if st.button("Evaluate Writing"):
        if len(text.strip()) < 30:
            st.error("Please write a longer paragraph (at least ~30 characters).")
            st.stop()

        # Safe correction pipeline
        corrected_spell, spell_issues = safe_spell_correct(text)
        grammar_issues = basic_grammar_checks(corrected_spell)
        issues = merge_issues(spell_issues, grammar_issues)

        st.subheader("Corrected Version (safe)")
        st.write(corrected_spell)

        st.subheader("Detected Issues")
        if issues:
            st.dataframe(issues, use_container_width=True)
        else:
            st.success("No major issues detected by the lightweight checker.")

        st.subheader("Recommendations")
        for r in recommendations_from_issues(issues):
            st.write(f"- {r}")

        pdf_name = f"writing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        pdf_path = generate_pdf(
            pdf_name,
            "Writing Evaluation Report",
            [
                f"Student: {name}",
                f"Email: {email}",
                f"Topic: {topic}",
                "",
                "Original Text:",
                text,
                "",
                "Corrected Text (safe):",
                corrected_spell,
                "",
                f"Total issues: {len(issues)} (Spelling={len(spell_issues)}, Grammar={len(grammar_issues)})",
            ]
        )

        with open(pdf_path, "rb") as f:
            st.download_button("Download PDF Report", f, file_name=pdf_name, mime="application/pdf")

# =====================================================
# SPEAKING (Upload audio)
# =====================================================
else:
    st.subheader("Upload your speaking recording (.wav or .mp3)")
    audio_file = st.file_uploader("Upload Audio File", type=["wav", "mp3"])

    if audio_file is not None:
        st.audio(audio_file)

        tmp_path = BASE_DIR / "temp_audio.wav"
        with open(tmp_path, "wb") as f:
            f.write(audio_file.read())

        with st.spinner("Transcribing speech..."):
            model = load_whisper()
            segments, info = model.transcribe(str(tmp_path))
            transcript = " ".join([seg.text.strip() for seg in segments]).strip()

        st.subheader("Transcript")
        st.write(transcript if transcript else "(No speech detected)")

        corrected_spell, spell_issues = safe_spell_correct(transcript)
        grammar_issues = basic_grammar_checks(corrected_spell)
        issues = merge_issues(spell_issues, grammar_issues)

        st.subheader("Corrected Transcript (safe)")
        st.write(corrected_spell)

        if issues:
            st.subheader("Detected Issues")
            st.dataframe(issues, use_container_width=True)
        else:
            st.success("No major issues detected by the lightweight checker.")

        words = len(re.findall(r"\b\w+\b", transcript))
        duration = getattr(info, "duration", None)
        wpm = (words / duration) * 60 if duration and duration > 0 else None

        st.subheader("Speaking Metrics")
        st.write(f"Words: {words}")
        if duration:
            st.write(f"Duration (s): {duration:.1f}")
        if wpm:
            st.write(f"Estimated WPM: {wpm:.1f}")

        st.subheader("Recommendations")
        for r in recommendations_from_issues(issues):
            st.write(f"- {r}")
        st.write("- Practice pronunciation by repeating corrected sentences slowly, then faster.")

        pdf_name = f"speaking_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        pdf_path = generate_pdf(
            pdf_name,
            "Speaking Evaluation Report",
            [
                f"Student: {name}",
                f"Email: {email}",
                f"Topic: {topic}",
                "",
                "Transcript:",
                transcript,
                "",
                "Corrected Transcript (safe):",
                corrected_spell,
                "",
                f"Total issues: {len(issues)} (Spelling={len(spell_issues)}, Grammar={len(grammar_issues)})",
                f"Words: {words}",
                f"Duration (s): {duration:.1f}" if duration else "Duration (s): —",
                f"WPM: {wpm:.1f}" if wpm else "WPM: —",
            ]
        )

        with open(pdf_path, "rb") as f:
            st.download_button("Download PDF Report", f, file_name=pdf_name, mime="application/pdf")
