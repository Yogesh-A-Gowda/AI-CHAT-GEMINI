import streamlit as st
import PyPDF2
from dotenv import load_dotenv
import google.generativeai as genai
import os
import io

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent

# Load environment variables from .env file
load_dotenv()

# --- Tool Definitions (from your agent code) ---
@tool
def calculator(a: float, b: float) -> str:
    """Useful for performing basic arithmetic calculations with numbers"""
    return f"The sum of {a} and {b} is {a + b}"

@tool
def say_hello(name: str) -> str:
    """Useful for greeting a user"""
    return f"Hello {name}, I hope you are well today"

# --- Initialize Gemini Model for LangChain Agent ---
@st.cache_resource
def get_gemini_agent_model():
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        st.error("Error: GOOGLE_API_KEY environment variable not set.")
        st.stop()
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, google_api_key=google_api_key)

# --- Create LangChain Agent ---
@st.cache_resource
def get_react_agent():
    model = get_gemini_agent_model()
    tools = [calculator, say_hello]
    return create_react_agent(model, tools)

# --- FILE ANALYZER ---
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text() + '\n'
    return text

def extract_text_from_file(uploaded_file):
    if uploaded_file.type == 'application/pdf':
        return extract_text_from_pdf(io.BytesIO(uploaded_file.read()))
    return uploaded_file.read().decode("utf-8")

def get_gemini_file_review_model():
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        st.error("Google API Key not found. Please set the GOOGLE_API_KEY environment variable in your .env file.")
        st.stop()
    genai.configure(api_key=google_api_key)
    return genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        system_instruction="You are an expert file reviewer with years of experience in HR and recruitment."
    )

# --- Streamlit App Layout ---
st.set_page_config(page_title="Gemini AI Assistant", page_icon="✨", layout="wide")
st.title("Gemini AI Assistant")

# Use tabs for different functionalities
tab1, tab2 = st.tabs(["Chat with AI Assistant", "File Review"])

with tab1:
    st.header("Chat with AI Assistant")

    # Initialize chat history in session state if not already present
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask me anything..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            agent_executer = get_react_agent()
            
            try:
                # Stream the response from the agent
                for chunk in agent_executer.stream({"messages": [HumanMessage(content=prompt)]}):
                    if "agent" in chunk and "messages" in chunk["agent"]:
                        for message in chunk["agent"]["messages"]:
                            full_response += message.content
                            message_placeholder.markdown(full_response + "▌")
                    elif "output" in chunk:
                        full_response += chunk["output"]
                        message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            except Exception as e:
                error_message = f"An error occurred: {e}"
                message_placeholder.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})


with tab2:
    st.header("File Reviewer")
    st.markdown("Upload your file and get AI feedback")

    uploaded_file = st.file_uploader('Upload your file here', type=["pdf","txt"])

    # This text input is for additional instructions, not the core content
    user_instructions = st.text_input("Enter specific instructions for the review (e.g., 'Summarize this report', 'Identify key arguments', 'Critique the writing style')")

    analyze = st.button("Analyze file")

    if analyze and uploaded_file:
        try:
            file_content = extract_text_from_file(uploaded_file)
            if not file_content.strip():
                st.error("File does not have any content")
                st.stop()
            
            file_model = get_gemini_file_review_model()
            
            generation_config = genai.types.GenerationConfig(
                temperature=0.7,
                max_output_tokens=1000
            )

            # --- CRITICAL CHANGE HERE ---
            # Construct the full prompt that includes both user instructions and file content
            full_review_prompt = f"""
            Analyze the following file content.
            
            User Instructions: {user_instructions if user_instructions else 'Provide a general review.'}

            File Content:
            {file_content}

            Please provide your analysis in a clear, structured format based on the instructions.
            """

            response = file_model.generate_content(
                contents=[
                    {"role": "user", "parts": [full_review_prompt]} # Send the combined prompt
                ],
                generation_config=generation_config
            )
            st.markdown('### Analyzed Results ###')
            st.markdown(response.text)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Add a rerun button if desired (though Streamlit often handles reruns implicitly)
# if st.button("Refresh Page"):
#     st.rerun()