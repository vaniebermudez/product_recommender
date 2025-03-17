import streamlit as st
import pandas as pd
from datetime import datetime
from utils.rag_pipeline import Conversation
import openai
import os

# Initialize OpenAI client


# Initialize conversation object
if "chat" not in st.session_state:
    st.session_state.chat = Conversation()
    st.session_state.history = []
    st.session_state.user_info = {}

# === Step 1: Function to Extract User Info ===
def extract_user_info():
    conversation = "\n".join(
        [f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.history]
    )

    prompt = f"""
    Extract the following structured information from the conversation:
    - Name
    - Age
    - Gender
    - Employment Type
    - Financial Goals and Priorities
    - Contact Details
    - Product Recommendation

    If any detail is missing, leave it blank.

    Conversation:
    {conversation}

    Output in JSON format:
    {{
        "name": "",
        "age": "",
        "gender": "",
        "employment": "",
        "financial_goals": "",
        "contact": "",
        "recommendation": ""
    }}
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        extracted_data = eval(response.choices[0].message.content.strip())
        st.session_state.user_info.update({k: v for k, v in extracted_data.items() if v})
        
        st.sidebar.markdown("### ✅ Extracted Info:")
        for key, value in st.session_state.user_info.items():
            st.sidebar.markdown(f"**{key.capitalize()}**: {value}")
    except Exception as e:
        st.sidebar.error(f"❌ Extraction Error: {e}")

# === Step 2: Chat Interface ===
st.title("💬 Product Recommender Chatbot")

# Display conversation history
for message in st.session_state.history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input box at the bottom
user_input = st.chat_input("Ask me about products...")

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
    st.session_state.history.append({"role": "user", "content": user_input})
    st.session_state.history.append({"role": "assistant", "content": response})

    # Extract user info dynamically
    extract_user_info()

# === Step 3: Save to Excel When Session Ends ===
def save_to_excel():
    if st.session_state.history and st.session_state.user_info:
        # Prepare data
        conversation = "\n".join(
            [f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.history]
        )

        data = {
            "Name": [st.session_state.user_info.get("name", "")],
            "Age": [st.session_state.user_info.get("age", "")],
            "Gender": [st.session_state.user_info.get("gender", "")],
            "Employment Type": [st.session_state.user_info.get("employment", "")],
            "Financial Goals and Priorities": [st.session_state.user_info.get("financial_goals", "")],
            "Contact Details": [st.session_state.user_info.get("contact", "")],
            "Product Recommendation": [st.session_state.user_info.get("recommendation", "")],
            "Conversation": [conversation],
            "Timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        }

        df = pd.DataFrame(data)

        # Append to existing file or create new
        file_path = "chat_history.xlsx"
        try:
            existing_df = pd.read_excel(file_path)
            df = pd.concat([existing_df, df], ignore_index=True)
        except FileNotFoundError:
            pass

        df.to_excel(file_path, index=False)
        st.success("✅ Chat history saved to Excel!")

# End chat and save button
if st.button("End Chat and Save"):
    save_to_excel()
    st.session_state.history = []
    st.session_state.user_info = {}
    st.success("✅ Chat ended and saved!")

