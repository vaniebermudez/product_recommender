import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
import openai

from utils.data_loader import load_scraped_data, load_pdf_data_from_folder

# Load OpenAI API Key

# === Step 1: Load Data ===
# Load scraped data from CSV
scraped_data = load_scraped_data("data/scraped_data.csv")

# Load combined PDF data (including OCR for scanned documents)
pdf_data = load_pdf_data_from_folder("data/pdfs")

# Combine all data into a single dataset
if 'content' in scraped_data.columns:
    combined_data = "\n".join(scraped_data['content'].dropna().tolist()) + "\n" + pdf_data
else:
    combined_data = pdf_data

# === Step 2: Prepare Data for Retrieval ===
# Split data into smaller chunks for efficient retrieval
splitter = CharacterTextSplitter(chunk_size=10000, chunk_overlap=100)
docs = splitter.split_text(combined_data)

# Create embeddings and vector store using FAISS
embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_texts(docs, embeddings)

# === Step 3: Define RAG Retrieval and Continuous Conversation ===
class Conversation:
    def __init__(self):
        self.history = []  # Stores conversation history
        self.retriever = vector_store.as_retriever()

    def generate_response(self, user_input):
        # Step 3a: Retrieve relevant chunks from FAISS
        retrieved_docs = self.retriever.get_relevant_documents(user_input)
        context = "\n".join([doc.page_content for doc in retrieved_docs])

        # Step 3b: Build conversation context with history
        conversation_history = "\n".join(self.history[-10:])  # Keep last 10 exchanges
        prompt = f"""
        You are a financial consultant expert for AXA Philippines. Your task is to recommend the best financial product based on the client’s profile, priorities, budget, and risk appetite. Make the conversation natural and engaging while remembering the details the client shares for future reference. Build on the information provided to make the conversation feel personal and tailored. Start by asking the following questions:

        Do not ask multiple questions at once that will confuse the user.
        
        Client Profile:

            What is your name?

            What is your age?

            What is your gender (if comfortable sharing)?

            Are you currently employed or self-employed?

            Do you have any dependents (e.g., spouse, children, parents)?

        Financial Goals and Priorities:

            What are your top financial goals? (e.g., retirement, education fund, health protection, wealth growth)

            Are you looking for short-term or long-term financial solutions?

            Do you have any specific financial concerns (e.g., health emergencies, market fluctuations, inflation)?

        Budget and Risk Tolerance:

            What is your available budget for financial products (monthly, quarterly, semi-annually or annually)?

        How would you describe your risk appetite?

            Low (conservative – prefer steady growth with minimal risk)

            Medium (balanced – willing to accept moderate risk for better returns)

            High (aggressive – comfortable with higher risk for potentially higher rewards)
        
        Contact Information:
        To provide you with more detailed recommendations and follow-ups, may I ask for your contact details?

            What is the best phone number to reach you?

            May I also have your email address so I can send you additional information and updates?
        

        Once the client shares their responses, remember these details and use them to guide the conversation naturally. Acknowledge their goals and concerns to make the client feel heard and valued. For example, if the client mentions wanting to save for retirement, respond with something like, 'That's a smart move. It's great that you're thinking ahead about retirement.'

        After gathering the information, analyze the responses and recommend the most suitable AXA product(s) based on the provided context. Ensure that the recommendations align with the client's profile, goals, and financial capacity.

        🚨 Strict Constraint: Do NOT recommend any product that is not explicitly part of the available AXA product portfolio or the context you have collected from the client. If the client asks about a product outside of the available offerings, politely clarify that it is not part of the AXA product line and refocus the conversation on suitable options within the provided context.

        After the conversation, ask the user for contact details (i.e., contact number, email address) so someone from AXA team can reach out after the conversation.
        
        Provide a clear and concise explanation of why the recommended product(s) fit the client’s needs and be prepared to answer any follow-up questions.

        Use the following context to answer the question:

        Context:
        {context}

        Conversation History:
        {conversation_history}

        User: {user_input}
        Assistant:
        """

        # Step 3c: Generate response using OpenAI (version 0.28)
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful product recommendation assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            if response and len(response.choices) > 0:
                reply = response.choices[0].message.content.strip()
            else:
                reply = "I'm sorry, I couldn't generate a response. Please try again."
        except Exception as e:
            reply = f"An error occurred: {str(e)}"


        # Step 3d: Update history with new interaction
        self.history.append(f"User: {user_input}")
        self.history.append(f"Assistant: {reply}")

        return reply

# === Step 4: Test Example ===
if __name__ == "__main__":
    chat = Conversation()
    print("\n=== Product Recommender Chatbot ===")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("👋 Goodbye!")
            break

        response = chat.generate_response(user_input)
        print(f"Assistant: {response}\n")
