import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import textwrap
from dotenv import load_dotenv
import speech_recognition as sr
import subprocess
import pyperclip  # Import pyperclip

# Load environment variables from .env file
load_dotenv()

# Retrieve Google API key from environment variables
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables.")

# Configure the Google Generative AI with the retrieved API key
import google.generativeai as genai
genai.configure(api_key=api_key)

# Set up a prompt template with system instructions and a placeholder for dynamic messages
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. You answer only DevOps-related queries."),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Initialize the Google Gemini AI model with specified parameters
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

# Combine the prompt and model into a single pipeline (chain)
chain = prompt | model

# Function to speak text using macOS say command
def speak_text(text):
    process = subprocess.Popen(["say", text])
    return process

# Function to copy text to clipboard
# def copy_to_clipboard(text):
#     pyperclip.copy(text)
#     st.success("Text copied to clipboard!")

# Initialize speech recognizer
recognizer = sr.Recognizer()

# Streamlit interface
st.title("DevOps Chatbot")

# Initialize an empty message history
messages_history = []

# Initialize session state
if "listening" not in st.session_state:
    st.session_state.listening = False
if "paused" not in st.session_state:
    st.session_state.paused = False
if "speech_process" not in st.session_state:
    st.session_state.speech_process = None

# Display buttons for controlling listening and speaking
col1, col2, col3 = st.columns(3)

# Start Listening button
if col1.button("Start Listening"):
    st.session_state.listening = True
    st.session_state.paused = False
    st.info("Listening... Click 'Pause' to stop temporarily.")
    if st.session_state.speech_process:
        st.session_state.speech_process.terminate()  # Stop any ongoing speech output

# Pause button
if col2.button("Pause"):
    st.session_state.listening = False
    st.session_state.paused = True
    st.info("Paused. Click 'Start Listening' to continue.")
    if st.session_state.speech_process:
        st.session_state.speech_process.terminate()  # Stop any ongoing speech output

# Copy Response button
# if col3.button("Copy Response"):
#     if messages_history:
#         last_response = messages_history[-1].content
#         copy_to_clipboard(last_response)
#     else:
#         st.warning("No response to copy.")

# Listening and processing
if st.session_state.listening:
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        st.write("Listening... Speak now.")
        audio = recognizer.listen(source)

        try:
            # Recognize speech using Google Web Speech API
            input_message = recognizer.recognize_google(audio)
            st.write(f'you : {input_message}')

            # Exit condition
            if input_message.lower() in ['bye-bye', 'q']:
                st.write("Goodbye!")
                if st.session_state.speech_process:
                    st.session_state.speech_process.terminate()
                st.session_state.listening = False
            else:
                # Append user message to history
                messages_history.append(HumanMessage(content=input_message))

                # Generate response using LangChain
                response = chain.invoke({"messages": messages_history})

                # Extract and display the output from the response
                output = response.content
                st.write('Bot : ' + textwrap.fill(output.strip().replace('*', '').replace('**', '')))

                # Speak the chatbot's response using macOS say command
                if st.session_state.speech_process:
                    st.session_state.speech_process.terminate()
                st.session_state.speech_process = speak_text(output)

                # Append the model's response to history
                messages_history.append(HumanMessage(content=output))

        except sr.UnknownValueError:
            st.write("Sorry, I did not understand that. Could you please repeat?")
        except sr.RequestError as e:
            st.write(f"Could not request results from Google Speech Recognition service; {e}")

# Initial welcome message
if not messages_history and not st.session_state.paused:
    welcome = 'Welcome to my Chatbot'
    st.write(welcome)
    # st.session_state.speech_process = speak_text(welcome)

# Display the conversation history
st.markdown("**Conversation History:**")
for message in messages_history:
    if message.content.startswith("Bot:"):
        # If it's the bot's response, wrap the content in a code block
        st.code(message.content[-5], language="yaml")
    else:
        st.markdown(f"You :{message.content}")
