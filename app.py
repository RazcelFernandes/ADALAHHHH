import streamlit as st
from groq import Groq
from dotenv import load_dotenv
import os

# =========================
# FILE SUPPORT
# =========================

from PIL import Image
import fitz
from docx import Document

# =========================
# AI IMAGE VISION
# =========================

from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration
)

# =========================
# LOAD ENV
# =========================

load_dotenv()

# =========================
# GROQ API
# =========================

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# =========================
# LOAD VISION MODEL
# =========================

@st.cache_resource
def load_vision_model():

    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )

    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )

    return processor, model

processor, vision_model = load_vision_model()

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Bot Razcel Fernandes",
    page_icon="🤖",
    layout="wide"
)

# =========================
# CUSTOM CSS
# =========================

st.markdown("""
<style>

.block-container {
    padding-top: 1rem;
}

[data-testid="stSidebar"] {
    background-color: #00000;
}

.stChatMessage {
    border-radius: 12px;
    padding: 10px;
}

</style>
""", unsafe_allow_html=True)

# =========================
# SESSION STATE
# =========================

if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = []

if "current_chat" not in st.session_state:
    st.session_state.current_chat = []

# =========================
# SIDEBAR
# =========================

with st.sidebar:

    st.title("🤖 BOT Razcel Fernandes")

    st.caption("AI Assistant")

    # =====================
    # NEW CHAT
    # =====================

    if st.button(
        "➕ New Chat",
        use_container_width=True
    ):

        if st.session_state.current_chat:

            title = st.session_state.current_chat[0]["content"][:30]

            st.session_state.chat_sessions.append(
                {
                    "title": title,
                    "messages": st.session_state.current_chat.copy()
                }
            )

        st.session_state.current_chat = []

        st.rerun()

    st.divider()

    st.subheader("📜 History")

    # =====================
    # SHOW HISTORY
    # =====================

    for i, session in enumerate(
        reversed(st.session_state.chat_sessions)
    ):

        if st.button(
            f"💬 {session['title']}",
            key=f"history_{i}",
            use_container_width=True
        ):

            st.session_state.current_chat = session["messages"]

            st.rerun()

# =========================
# MAIN HEADER
# =========================

st.title("🤖 BOT Razcel Fernandes")
st.caption("AI Chatbot with File & Image Support")

# =========================
# FILE UPLOADER
# =========================

uploaded_file = st.file_uploader(
    "📎 Upload file atau gambar",
    type=["png", "jpg", "jpeg", "pdf", "txt", "docx"]
)

file_content = ""

# =========================
# READ FILE
# =========================

if uploaded_file:

    file_type = uploaded_file.type

    # =====================
    # IMAGE
    # =====================

    if "image" in file_type:

        image = Image.open(uploaded_file)

        st.image(
            image,
            caption="Image Uploaded",
            use_container_width=True
        )

        # =================
        # AI IMAGE ANALYSIS
        # =================

        with st.spinner("AI sedang menganalisa gambar..."):

            inputs = processor(
                images=image,
                return_tensors="pt"
            )

            out = vision_model.generate(**inputs)

            caption = processor.decode(
                out[0],
                skip_special_tokens=True
            )

            file_content = f"""
Deskripsi gambar:
{caption}
"""

            st.success(
                f"AI membaca gambar: {caption}"
            )

    # =====================
    # TXT
    # =====================

    elif file_type == "text/plain":

        file_content = uploaded_file.read().decode(
            "utf-8"
        )

        with st.expander("📄 Isi TXT"):

            st.text(file_content[:5000])

    # =====================
    # PDF
    # =====================

    elif file_type == "application/pdf":

        pdf_text = ""

        pdf = fitz.open(
            stream=uploaded_file.read(),
            filetype="pdf"
        )

        for page in pdf:

            pdf_text += page.get_text()

        file_content = pdf_text

        with st.expander("📘 Isi PDF"):

            st.text(file_content[:5000])

    # =====================
    # DOCX
    # =====================

    elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":

        doc = Document(uploaded_file)

        doc_text = ""

        for para in doc.paragraphs:

            doc_text += para.text + "\n"

        file_content = doc_text

        with st.expander("📄 Isi DOCX"):

            st.text(file_content[:5000])

# =========================
# DISPLAY CHAT
# =========================

for message in st.session_state.current_chat:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

# =========================
# CHAT INPUT
# =========================

prompt = st.chat_input(
    "Ketik pesan..."
)

# =========================
# PROCESS CHAT
# =========================

if prompt:

    # =====================
    # USER MESSAGE
    # =====================

    st.session_state.current_chat.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    # DISPLAY USER
    with st.chat_message("user"):

        st.markdown(prompt)

    # =====================
    # AI RESPONSE
    # =====================

    with st.chat_message("assistant"):

        with st.spinner(
            "AI sedang mengetik..."
        ):

            try:

                messages_for_ai = []

                # =================
                # CHAT HISTORY
                # =================

                for msg in st.session_state.current_chat:

                    messages_for_ai.append(
                        {
                            "role": msg["role"],
                            "content": msg["content"]
                        }
                    )

                # =================
                # FILE CONTEXT
                # =================

                if file_content != "":

                    messages_for_ai.append(
                        {
                            "role": "user",
                            "content": f"""
Gunakan informasi berikut untuk membantu menjawab pertanyaan user.

ISI FILE / GAMBAR:
{file_content}

PERTANYAAN USER:
{prompt}
"""
                        }
                    )

                # =================
                # AI REQUEST
                # =================

                chat_completion = client.chat.completions.create(

                    messages=messages_for_ai,

                    model="llama-3.1-8b-instant",

                    temperature=0.7,

                    max_tokens=2048,
                )

                response = (
                    chat_completion
                    .choices[0]
                    .message
                    .content
                )

                st.markdown(response)

                # =================
                # SAVE AI RESPONSE
                # =================

                st.session_state.current_chat.append(
                    {
                        "role": "assistant",
                        "content": response
                    }
                )

            except Exception as e:

                st.error(f"Error: {e}")