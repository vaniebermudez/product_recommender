import streamlit as st
from utils.rag_pipeline import Conversation

# Initialize conversation object
if "chat" not in st.session_state:
    st.session_state.chat = Conversation()

st.title("💬 PrEX - Product Recommender Expert ")

# Display conversation history
for message in st.session_state.get("history", []):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input box at the bottom
user_input = st.chat_input("Ask me about AXA products...")

if user_input:
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate response using RAG pipeline
    response = st.session_state.chat.generate_response(user_input)

    # Display assistant message
    with st.chat_message("assistant"):
        st.markdown(response)

    # Update session history
    if "history" not in st.session_state:
        st.session_state.history = []

    st.session_state.history.append({"role": "user", "content": user_input})
    st.session_state.history.append({"role": "assistant", "content": response})
