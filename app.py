import hashlib
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from io import BytesIO
from typing import List

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from pypdf import PdfReader
from streamlit.errors import StreamlitSecretNotFoundError


load_dotenv()

# User view:
# These settings decide how the app behaves when it reads PDFs,
# creates summaries, and answers questions.
#
# Coder view:
# These are global configuration values for model choice, chunking,
# prompt size, and simple retrieval behavior.
DEFAULT_CHAT_MODEL = os.getenv("GROQ_CHAT_MODEL", "llama-3.1-8b-instant")
MAX_CONTEXT_CHUNKS = 4
CHUNK_SIZE = 1400
CHUNK_OVERLAP = 200
SUMMARY_CONTEXT_LIMIT = 12000
RECENT_CHAT_LIMIT = 4
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "for", "from",
    "give", "how", "i", "in", "is", "it", "me", "of", "on", "or", "please", "short",
    "show", "tell", "that", "the", "this", "to", "what", "when", "where", "which",
    "who", "why", "with", "you", "your",
}
VAGUE_PATTERNS = (
    "explain this",
    "what is the main idea",
    "main idea",
    "summary",
    "short revision",
    "revision answer",
    "explain in simple words",
    "simpler",
)


CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&display=swap');

html, body, [class*="css"], input, textarea, button {
    font-family: 'Manrope', sans-serif;
}

.stApp {
    background: linear-gradient(180deg, #f8fafc 0%, #eef4ff 100%);
    color: #172033;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ffffff 0%, #f7f9fc 100%);
    border-right: 1px solid rgba(80, 93, 122, 0.12);
    color: #172033;
}

[data-testid="stSidebar"] .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.hero-card {
    background: rgba(255, 255, 255, 0.92);
    border: 1px solid rgba(71, 85, 105, 0.14);
    border-radius: 20px;
    padding: 1.5rem;
    box-shadow: 0 10px 28px rgba(148, 163, 184, 0.12);
    margin-bottom: 1rem;
}

.hero-title {
    font-size: 2.3rem;
    line-height: 1.1;
    font-weight: 800;
    color: #162033;
    margin-bottom: 0.5rem;
}

.hero-subtitle {
    font-size: 1.08rem;
    line-height: 1.7;
    color: #334155;
    margin: 0;
}

.section-label {
    font-size: 1.05rem;
    font-weight: 800;
    color: #172033;
    margin: 0.4rem 0 0.8rem;
}

.sidebar-title {
    font-size: 1.35rem;
    font-weight: 800;
    color: #172033;
    margin-bottom: 0.35rem;
}

.sidebar-copy {
    font-size: 0.98rem;
    line-height: 1.6;
    color: #425066;
    margin-bottom: 1rem;
}

.surface-card {
    background: rgba(255, 255, 255, 0.92);
    border: 1px solid rgba(71, 85, 105, 0.12);
    border-radius: 18px;
    padding: 1rem 1.1rem;
    box-shadow: 0 10px 24px rgba(148, 163, 184, 0.10);
    margin-bottom: 1rem;
}

.app-footer {
    text-align: center;
    color: #5b6b82;
    font-size: 0.92rem;
    margin-top: 1.2rem;
    padding-top: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.01em;
    border-top: 1px solid rgba(71, 85, 105, 0.12);
}

.app-footer strong {
    color: #334155;
}

.file-list-card {
    background: rgba(255, 255, 255, 0.94);
    border: 1px solid rgba(71, 85, 105, 0.12);
    border-radius: 16px;
    padding: 0.85rem 1rem;
    margin-top: 0.8rem;
}

.file-list-card p {
    margin: 0.2rem 0;
    color: #172033;
    font-size: 0.96rem;
    line-height: 1.5;
}

.stButton > button {
    border-radius: 14px;
    font-weight: 700;
    font-size: 1rem;
    padding: 0.7rem 1rem;
    border: 1px solid transparent;
    background: linear-gradient(135deg, #ffb84d 0%, #ff8f5a 100%);
    color: white;
    box-shadow: 0 8px 18px rgba(255, 159, 90, 0.22);
}

.stButton > button:hover {
    background: linear-gradient(135deg, #f9ab35 0%, #f67c43 100%);
}

.stTextInput input,
textarea {
    background: white !important;
    color: #172033 !important;
    border-radius: 12px !important;
    caret-color: #172033 !important;
}

[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {
    font-size: 1.02rem;
    line-height: 1.75;
    color: #223046;
}

[data-testid="stChatMessage"] {
    background: rgba(255, 255, 255, 0.88);
    border: 1px solid rgba(71, 85, 105, 0.1);
    border-radius: 18px;
    padding: 0.55rem 0.85rem;
    margin-bottom: 0.75rem;
}

[data-testid="stFileUploader"] {
    background: rgba(255, 255, 255, 0.96);
    border-radius: 16px;
    padding: 0.4rem;
}

[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] div {
    color: #172033 !important;
}

[data-testid="stFileUploader"] section {
    color: #172033 !important;
}

[data-testid="stFileUploader"] button {
    color: #172033 !important;
}

[data-testid="stFileUploaderDropzone"] {
    background: #1e293b !important;
    border: 1px solid rgba(255, 255, 255, 0.14) !important;
    border-radius: 14px !important;
    color: #f8fafc !important;
}

[data-testid="stFileUploaderDropzone"] * {
    color: #f8fafc !important;
    opacity: 1 !important;
}

[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzone"] p,
[data-testid="stFileUploaderDropzone"] div {
    color: #f8fafc !important;
}

[data-testid="stFileUploaderDropzone"] button {
    background: #ffffff !important;
    color: #0f172a !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 10px !important;
    box-shadow: none !important;
    font-weight: 600 !important;
}

[data-testid="stFileUploaderDropzone"] button:hover {
    background: #f8fafc !important;
    border-color: #94a3b8 !important;
}

[data-testid="stFileUploaderFile"] {
    background: rgba(255, 255, 255, 0.92);
    border: 1px solid rgba(71, 85, 105, 0.14);
    border-radius: 14px;
}

[data-testid="stFileUploaderFileName"],
[data-testid="stFileUploaderFileData"] {
    color: #172033 !important;
    opacity: 1 !important;
}

[data-testid="stFileUploaderFile"] svg {
    fill: currentColor !important;
}

[data-testid="stFileUploaderFile"] button {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: #475569 !important;
    position: relative;
    width: 1.5rem;
    height: 1.5rem;
    min-width: 1.5rem !important;
    min-height: 1.5rem !important;
    padding: 0 !important;
    border-radius: 999px !important;
}

[data-testid="stFileUploaderFile"] button svg,
[data-testid="stFileUploaderFile"] button path {
    display: none !important;
}

[data-testid="stFileUploaderFile"] button::before {
    content: "×";
    font-size: 1.1rem;
    font-weight: 700;
    line-height: 1;
    color: #475569;
    display: flex;
    align-items: center;
    justify-content: center;
    width: 100%;
    height: 100%;
}

[data-testid="stFileUploaderFile"] button:hover {
    background: rgba(220, 38, 38, 0.08) !important;
}

[data-testid="stFileUploaderFile"] button:hover::before {
    color: #dc2626 !important;
}

[data-testid="stAlert"] * {
    color: inherit !important;
}

[data-testid="stSidebar"] .stButton {
    margin-top: 0.8rem;
    margin-bottom: 0.8rem;
}

[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4,
[data-testid="stMarkdownContainer"] strong {
    color: #1f2f45;
}

[data-testid="stChatInput"] {
    border-top: none;
    background: transparent;
}

[data-testid="stChatInput"] textarea,
[data-testid="stChatInput"] input {
    background: rgba(255, 255, 255, 0.96) !important;
    color: #172033 !important;
    border: 1px solid rgba(71, 85, 105, 0.14) !important;
    border-radius: 16px !important;
    caret-color: #172033 !important;
}

[data-testid="stChatInput"] input::placeholder,
[data-testid="stChatInput"] textarea::placeholder {
    color: #64748b !important;
    opacity: 1 !important;
}

[data-testid="stChatInput"] button,
[data-testid="stChatInput"] svg {
    color: #172033 !important;
    fill: #172033 !important;
}
</style>
"""


@dataclass
class ChunkRecord:
    # User view:
    # This is one small piece of study material taken from the uploaded PDF.
    #
    # Coder view:
    # Each record stores a chunk of text plus file/page metadata for citations.
    text: str
    source_name: str
    page_number: int


@dataclass
class RetrievedContext:
    # User view:
    # This stores the PDF content the app decided to use for answering.
    #
    # Coder view:
    # It carries selected chunks and flags describing whether fallback logic was used.
    chunks: List[ChunkRecord]
    used_fallback: bool = False
    used_summary: bool = False


def ensure_session_state() -> None:
    # User view:
    # Streamlit refreshes often, so we save important app memory here.
    #
    # Coder view:
    # Keep app data across reruns triggered by Streamlit interactions.
    st.session_state.setdefault("chunk_records", [])
    st.session_state.setdefault("document_fingerprint", None)
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("summary_cache", "")


def get_secret(name: str) -> str:
    # User view:
    # This safely reads your API key from `.env` or Streamlit secrets.
    #
    # Coder view:
    # Try Streamlit secrets first, then fall back to environment variables.
    try:
        secret_value = st.secrets.get(name, "")
    except StreamlitSecretNotFoundError:
        secret_value = ""
    return secret_value or os.getenv(name, "")


def reset_chat() -> None:
    # User view:
    # Clears the conversation so the user can start fresh.
    #
    # Coder view:
    # Reset only the chat history, not the uploaded document state.
    st.session_state.messages = []


def fingerprint_uploaded_files(
    uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile],
) -> str:
    # User view:
    # Helps the app notice when new notes are uploaded.
    #
    # Coder view:
    # Detect when the user uploads a different set of files so we can reprocess them.
    digest = hashlib.sha256()
    for uploaded_file in uploaded_files:
        digest.update(uploaded_file.name.encode("utf-8"))
        digest.update(uploaded_file.getvalue())
    return digest.hexdigest()


def split_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    # User view:
    # Long PDF text is cut into smaller readable pieces before the AI uses it.
    #
    # Coder view:
    # Break long PDF text into overlapping chunks so the model sees manageable context.
    cleaned = " ".join(text.split())
    if not cleaned:
        return []

    chunks = []
    start = 0
    text_length = len(cleaned)

    while start < text_length:
        end = min(text_length, start + chunk_size)
        if end < text_length:
            boundary = cleaned.rfind(" ", start, end)
            if boundary > start + 200:
                end = boundary

        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = max(end - overlap, start + 1)

    return chunks


def extract_chunk_records(
    uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile],
) -> List[ChunkRecord]:
    # User view:
    # This is the part where the app actually reads the uploaded PDF notes.
    #
    # Coder view:
    # Read every uploaded PDF page and convert it into chunk records with source metadata.
    chunk_records: List[ChunkRecord] = []

    for uploaded_file in uploaded_files:
        reader = PdfReader(BytesIO(uploaded_file.getvalue()))

        for page_index, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            for chunk in split_text(page_text):
                chunk_records.append(
                    ChunkRecord(
                        text=chunk,
                        source_name=uploaded_file.name,
                        page_number=page_index,
                    )
                )

    return chunk_records


def tokenize(text: str) -> set[str]:
    # User view:
    # Converts text into important keywords.
    #
    # Coder view:
    # Remove common filler words so matching focuses on useful terms.
    tokens = set(re.findall(r"\b[a-zA-Z0-9]{2,}\b", text.lower()))
    return {token for token in tokens if token not in STOPWORDS}


def is_vague_question(question: str) -> bool:
    # User view:
    # Detects broad questions like "Explain this" or "main idea".
    #
    # Coder view:
    # If the query is too short or generic, retrieval uses fallback behavior.
    normalized = " ".join(question.lower().split())
    if normalized in VAGUE_PATTERNS:
        return True
    return len(tokenize(question)) <= 2


def score_chunk(question_terms: set[str], chunk_text: str) -> int:
    # User view:
    # Gives each PDF chunk a score based on how well it matches the question.
    #
    # Coder view:
    # Lightweight keyword scoring with a small frequency bonus.
    chunk_terms = tokenize(chunk_text)
    overlap_terms = question_terms & chunk_terms
    if not overlap_terms:
        return 0

    lowered = chunk_text.lower()
    frequency_bonus = sum(lowered.count(term) for term in overlap_terms)
    return len(overlap_terms) * 3 + frequency_bonus


def retrieve_relevant_chunks(
    question: str,
    chunk_records: List[ChunkRecord],
) -> RetrievedContext:
    # User view:
    # Finds the most useful parts of the uploaded notes for the current question.
    #
    # Coder view:
    # Use lightweight keyword scoring with a fallback for vague questions.
    if not chunk_records:
        return RetrievedContext(chunks=[], used_fallback=True)

    question_terms = tokenize(question)
    vague_question = is_vague_question(question)

    if not question_terms or vague_question:
        return RetrievedContext(
            chunks=chunk_records[:MAX_CONTEXT_CHUNKS],
            used_fallback=True,
            used_summary=bool(st.session_state.summary_cache),
        )

    scored_chunks = []
    for record in chunk_records:
        score = score_chunk(question_terms, record.text)
        if score > 0:
            scored_chunks.append((score, len(record.text), record))

    scored_chunks.sort(key=lambda item: (item[0], item[1]), reverse=True)
    selected = [item[2] for item in scored_chunks[:MAX_CONTEXT_CHUNKS]]
    if selected:
        return RetrievedContext(
            chunks=selected,
            used_fallback=False,
            used_summary=bool(st.session_state.summary_cache),
        )

    return RetrievedContext(
        chunks=chunk_records[:MAX_CONTEXT_CHUNKS],
        used_fallback=True,
        used_summary=bool(st.session_state.summary_cache),
    )


def format_context(chunk_records: List[ChunkRecord]) -> str:
    # User view:
    # Formats note snippets so the AI can read them with source labels.
    #
    # Coder view:
    # Turn retrieved chunks into a prompt-ready context block.
    sections = []
    for index, record in enumerate(chunk_records, start=1):
        sections.append(
            f"[Source {index}] {record.source_name} - page {record.page_number}\n{record.text}"
        )
    return "\n\n".join(sections)


def build_sources_list(chunk_records: List[ChunkRecord]) -> str:
    
    # Builds the source list shown under answers.
    # Deduplicate citations by file name and page number.
    seen = set()
    lines = []
    for record in chunk_records:
        key = (record.source_name, record.page_number)
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"- `{record.source_name}` page {record.page_number}")
    return "\n".join(lines)


def get_recent_chat_context() -> str:
    
    # Helps the app remember the last few things the user asked.
    # Include a small amount of recent chat so follow-up questions make sense.
    recent_messages = st.session_state.messages[-RECENT_CHAT_LIMIT:]
    if not recent_messages:
        return ""

    lines = []
    for message in recent_messages:
        role = "Student" if message["role"] == "user" else "Tutor"
        lines.append(f"{role}: {message['content']}")
    return "\n".join(lines)


def get_llm(chat_model_name: str) -> ChatGroq:
    
    # Creates the AI model object used for summary and Q&A.
    # Single place to configure the Groq chat model.
    return ChatGroq(
        model=chat_model_name,
        temperature=0,
        groq_api_key=get_secret("GROQ_API_KEY"),
    )


def build_summary_context(chunk_records: List[ChunkRecord]) -> str:
    
    # Collect enough text for a summary, but balance it across uploaded files
    # so later PDFs are not ignored when the prompt reaches its size limit.
    chunks_by_source = defaultdict(list)
    for record in chunk_records:
        chunks_by_source[record.source_name].append(record)

    ordered_sources = list(chunks_by_source.keys())
    source_positions = {source: 0 for source in ordered_sources}
    included_headers = set()
    sections = []
    current_length = 0

    while True:
        added_any = False
        for source in ordered_sources:
            records = chunks_by_source[source]
            position = source_positions[source]
            if position >= len(records):
                continue

            record = records[position]
            header = ""
            if source not in included_headers:
                header = f"\n=== File: {source} ===\n"

            snippet = f"{header}Page {record.page_number}\n{record.text}\n\n"
            if current_length + len(snippet) > SUMMARY_CONTEXT_LIMIT:
                return "".join(sections)

            sections.append(snippet)
            current_length += len(snippet)
            source_positions[source] += 1
            included_headers.add(source)
            added_any = True

        if not added_any:
            break

    return "".join(sections)


def generate_summary(chunk_records: List[ChunkRecord], chat_model_name: str) -> str:
    
    # Ask the model to produce a structured beginner-friendly study summary.
    context = build_summary_context(chunk_records)
    if not context.strip():
        return "I could not find enough readable text to summarize."

    llm = get_llm(chat_model_name)
    prompt = [
        SystemMessage(
            content=(
                "You are a friendly study buddy. Summarize the material in a way that helps "
                "a beginner learn it. Stay accurate to the context, use simple language, "
                "and organize the answer clearly."
            )
        ),
        HumanMessage(
            content=(
                "Create a study summary for all uploaded PDFs together. If there are multiple files, "
                "cover each file fairly and mention file-specific differences when useful.\n\n"
                "Use these sections:\n"
                "1. Main idea\n"
                "2. Important points\n"
                "3. Easy explanation\n"
                "4. Quick revision list\n\n"
                f"Context:\n{context}"
            )
        ),
    ]
    response = llm.invoke(prompt)
    return response.content


def answer_question(
    question: str,
    retrieved_context: RetrievedContext,
    chat_model_name: str,
) -> str:
    
    # This is the heart of the chat feature. It answers like a study buddy.
    # Combine summary, recent chat, and matching chunks to answer like a tutor.
    llm = get_llm(chat_model_name)
    summary_context = st.session_state.summary_cache.strip()
    chunk_context = format_context(retrieved_context.chunks) if retrieved_context.chunks else ""
    recent_chat_context = get_recent_chat_context()

    prompt = [
        SystemMessage(
            content=(
                "You are a helpful study buddy. Use the provided PDF context and summary to answer "
                "like a supportive tutor. Explain in simple language, teach step by step when useful, "
                "and use a short example if it helps. If the question is broad or vague, still try to "
                "give the most helpful answer from the available material instead of refusing too early. "
                "If the material truly does not cover the answer, say that honestly. Mention source file "
                "names and page numbers when possible."
            )
        ),
        HumanMessage(
            content=(
                f"Question:\n{question}\n\n"
                f"Study Summary:\n{summary_context or 'No summary generated yet.'}\n\n"
                f"Recent Chat:\n{recent_chat_context or 'No prior chat.'}\n\n"
                f"Relevant PDF Context:\n{chunk_context or 'No direct chunk match found; use the study summary and general document context.'}\n\n"
                f"Fallback retrieval used: {'yes' if retrieved_context.used_fallback else 'no'}"
            )
        ),
    ]
    response = llm.invoke(prompt)
    return response.content


def process_documents(
    uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile],
) -> None:
    # User view:
    # Processes uploaded notes one time, unless the user uploads different files.
    #
    # Coder view:
    # Only re-read PDFs when the uploaded files actually change.
    fingerprint = fingerprint_uploaded_files(uploaded_files)
    if fingerprint == st.session_state.document_fingerprint:
        return

    with st.spinner("Reading PDF files..."):
        chunk_records = extract_chunk_records(uploaded_files)
        if not chunk_records:
            raise ValueError(
                "No readable text was found in the uploaded PDF files. "
                "If these are scanned PDFs, they likely need OCR support first."
            )

        st.session_state.chunk_records = chunk_records
        st.session_state.document_fingerprint = fingerprint
        st.session_state.summary_cache = ""
        reset_chat()


def render_sidebar() -> tuple[list, str]:
    # Sidebar handles file upload and a small amount of user guidance.
    with st.sidebar:
        st.markdown('<div class="sidebar-title">Study Setup</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="sidebar-copy">Upload your PDF notes, make a quick summary, and ask questions in simple language.</div>',
            unsafe_allow_html=True,
        )
        uploaded_files = st.file_uploader(
            "Upload your study PDFs",
            type=["pdf"],
            accept_multiple_files=True,
        )
        chat_model_name = DEFAULT_CHAT_MODEL
        with st.expander("Advanced settings"):
            chat_model_name = st.text_input("Groq chat model", value=DEFAULT_CHAT_MODEL)
        if st.button("Clear chat", use_container_width=True):
            reset_chat()
        st.caption("Tip: start with the summary, then ask short follow-up questions.")
        st.markdown("---")
        render_footer()

    return uploaded_files, chat_model_name


def validate_configuration(groq_api_key: str, chat_model_name: str) -> None:
    
    # Stops the app early if the setup is incomplete.
    # Basic guard checks before running the main app flow.
    if not groq_api_key:
        st.warning("Add `GROQ_API_KEY` to your `.env` file before using the app.")
        st.stop()

    if not chat_model_name.strip():
        st.error("Groq chat model cannot be empty.")
        st.stop()


def main() -> None:
    # The full app flow lives here: open app, upload notes, summarize, and ask questions.
    #
    # Coder view:
    # Main Streamlit app flow: setup, upload, summarize, and chat.
    st.set_page_config(page_title="GeekBuddy AI", layout="wide")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    ensure_session_state()

    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">GeekBuddy AI</div>
            <p class="hero-subtitle">
                Turn your PDF notes into a calm study space. Start with a summary,
                then ask questions and get simpler explanations like a study partner would give.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    groq_api_key = get_secret("GROQ_API_KEY")
    uploaded_files, chat_model_name = render_sidebar()
    validate_configuration(groq_api_key, chat_model_name)

    if not uploaded_files:
        st.info("Upload one or more PDF files from the left panel to get started.")
        st.markdown("You can use this app to make a quick summary, clarify tough topics, and ask follow-up questions.")
        with st.container(border=True):
            st.subheader("How To Study Here")
            st.write("1. Upload your notes from the sidebar.")
            st.write("2. Create the summary to get the big picture.")
            st.write("3. Ask questions like you would ask a tutor.")
            st.write("4. If needed, ask for a simpler explanation or shorter answer.")
        st.stop()

    try:
        process_documents(uploaded_files)
    except Exception as exc:
        st.error(f"Document processing failed: {exc}")
        st.stop()

    st.markdown('<div class="surface-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Study Tools</div>', unsafe_allow_html=True)
    action_col, helper_col = st.columns([1, 1.2])
    with action_col:
        st.markdown('<div class="surface-card">', unsafe_allow_html=True)
        if st.button("Generate Study Summary", use_container_width=True):
            try:
                with st.spinner("Analyzing your notes..."):
                    st.session_state.summary_cache = generate_summary(
                        st.session_state.chunk_records,
                        chat_model_name,
                    )
                st.success("Summary ready below.")
            except Exception as exc:
                st.error(f"Summary generation failed: {exc}")
        st.caption("Generate a clean revision summary before you start chatting.")
        st.markdown('</div>', unsafe_allow_html=True)
    with helper_col:
        st.write("Generate a clean revision summary before you start chatting, then ask follow-up questions below.")
    st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.summary_cache:
        st.subheader("Study Summary")
        with st.container(border=True):
            st.markdown(st.session_state.summary_cache)
            st.download_button(
                "Download Summary",
                data=st.session_state.summary_cache,
                file_name="geekbuddy-study-summary.md",
                mime="text/markdown",
                use_container_width=False,
            )
    else:
        st.subheader("Study Summary")
        with st.container(border=True):
            st.write("Generate the summary to see the main ideas, key points, and a quick revision list here.")

    st.subheader("Ask Questions")
    st.write("Ask things like `Explain this in simple words`, `What is the main idea?`, or `Can you give me a short revision answer?`")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask something about your PDFs")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            retrieved_context = retrieve_relevant_chunks(
                question=question,
                chunk_records=st.session_state.chunk_records,
            )
            answer = answer_question(question, retrieved_context, chat_model_name)
            sources = build_sources_list(retrieved_context.chunks)
            final_response = f"{answer}\n\n**Sources**\n{sources}" if sources else answer
        except Exception as exc:
            final_response = f"I ran into an error while answering: {exc}"

        st.markdown(final_response)

    st.session_state.messages.append({"role": "assistant", "content": final_response})


def render_footer() -> None:
    st.markdown(
        '<div class="app-footer"><strong>GeekBuddy AI</strong><br>Built by Theertha Raveendranath | Class of 2026</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

