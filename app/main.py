"""Streamlit interface for the BNS Legal Agent."""

import asyncio
from typing import AsyncIterator

import nest_asyncio
import streamlit as st

from core.retriever import HybridRetriever
from core.generator import LegalGenerator
from models.schema import LegalChunk

# Allow nested event loops (required for Streamlit Cloud)
nest_asyncio.apply()

# ============================================================
# Page Configuration
# ============================================================
st.set_page_config(
    page_title="BNS Legal Agent",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Session State Initialization
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "retriever" not in st.session_state:
    st.session_state.retriever = HybridRetriever()

if "generator" not in st.session_state:
    st.session_state.generator = LegalGenerator()


# ============================================================
# Async Helper
# ============================================================
def run_async(coro):
    """Run an async coroutine in a sync context."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


async def collect_stream(stream: AsyncIterator[str]) -> str:
    """Collect all chunks from an async stream."""
    chunks = []
    async for chunk in stream:
        chunks.append(chunk)
    return "".join(chunks)


def stream_generator(query: str, chunks: list[LegalChunk]):
    """Sync generator that wraps async streaming for st.write_stream."""
    async def _stream():
        generator: LegalGenerator = st.session_state.generator
        async for token in generator.get_answer_stream(query, chunks):
            yield token
    
    # Create new event loop for streaming
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        # Get the async generator
        async_gen = _stream()
        
        # Iterate and yield synchronously
        while True:
            try:
                token = loop.run_until_complete(async_gen.__anext__())
                yield token
            except StopAsyncIteration:
                break
    finally:
        loop.close()


# ============================================================
# Sidebar Configuration
# ============================================================
with st.sidebar:
    st.header("⚙️ Configuration")
    st.divider()
    
    # Temperature slider
    temperature = st.slider(
        "🎨 Creativity (Temperature)",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.1,
        help="Lower = more factual, Higher = more creative",
    )
    
    # Model selector (fixed for now)
    model = st.selectbox(
        "🤖 Model",
        options=["llama-3.3-70b-versatile"],
        index=0,
        disabled=True,
        help="Currently using Llama 3.3 70B via Groq",
    )
    
    st.divider()
    
    # Info section
    st.markdown("### 📚 Data Sources")
    st.markdown("""
    - **BNS** - Bharatiya Nyaya Sanhita
    - **BNSS** - Bharatiya Nagarik Suraksha Sanhita
    - **BSA** - Bharatiya Sakshya Adhiniyam
    - **Constitution** - Indian Constitution
    """)
    
    st.divider()
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ============================================================
# Main Header
# ============================================================
st.title("🇮🇳 Bharatiya Nyaya Sanhita (BNS) AI")
st.caption("Your intelligent assistant for Indian Criminal Law • Powered by RAG + Groq")

st.divider()


# ============================================================
# Chat History Display
# ============================================================
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ============================================================
# Chat Input & Processing
# ============================================================
if prompt := st.chat_input("Ask a legal question about BNS, BNSS, BSA, or Constitution..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Process and respond
    with st.chat_message("assistant"):
        # Thinking expander with search progress
        with st.expander("🔍 **Thinking...**", expanded=True):
            # Step 1: Intent identification
            intent_placeholder = st.empty()
            intent_placeholder.markdown("🔍 **Identifying Intent...** ⏳")
            
            # Step 2: Vector search
            search_placeholder = st.empty()
            search_placeholder.markdown("⚡ **Searching 11,000+ legal chunks...** ⏳")
            
            # Perform the search
            retriever: HybridRetriever = st.session_state.retriever
            chunks = run_async(retriever.search(prompt, k=5, candidates=25))
            
            # Update status
            intent_placeholder.markdown("🔍 **Identifying Intent...** ✅")
            search_placeholder.markdown(f"⚡ **Searching 11,000+ legal chunks...** ✅ Found {len(chunks)} relevant results")
            
            # Step 3: Reranking
            rerank_placeholder = st.empty()
            rerank_placeholder.markdown("🧠 **Re-ranking top 25 results with FlashRank...** ✅")
            
            # Display retrieved chunks
            st.divider()
            st.markdown("**📄 Retrieved Sources:**")
            
            for i, chunk in enumerate(chunks[:3], 1):
                section = chunk.metadata.section_number or "N/A"
                source = chunk.metadata.source_document or "Unknown"
                act_name = chunk.metadata.act_name or source.replace(".pdf", "").upper()
                preview = chunk.text[:150] + "..." if len(chunk.text) > 150 else chunk.text
                
                st.markdown(f"""
                **{i}. {act_name} - Section {section}**
                > {preview}
                """)
        
        # Streaming response
        response_placeholder = st.empty()
        
        # Stream the response
        full_response = response_placeholder.write_stream(
            stream_generator(prompt, chunks)
        )
        
        # Add assistant message to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
        })


# ============================================================
# Footer
# ============================================================
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: #666; font-size: 0.8em;'>
        <p>⚠️ <strong>Disclaimer:</strong> This AI assistant provides legal information, not legal advice. 
        Always consult a qualified legal professional for specific cases.</p>
        <p>Built with ❤️ using Streamlit, LangChain, Qdrant, and Groq</p>
    </div>
    """,
    unsafe_allow_html=True,
)
