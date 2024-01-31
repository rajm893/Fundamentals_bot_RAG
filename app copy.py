import streamlit as st
from streamlit_chat import message
# from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
import os
import speech_recognition as sr
from gtts import gTTS
import io
import pygame
from utils import *
# Import your utility functions from utils.py
from utils import listen_for_wake_word, listen_for_stop_word

if 'responses' not in st.session_state:
    st.session_state['responses'] = ["How can I assist you?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

if 'buffer_memory' not in st.session_state:
    st.session_state.buffer_memory = ConversationBufferWindowMemory(k=3, return_messages=True)

llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=os.getenv("OPENAI_API_KEY"))
system_msg_template = SystemMessagePromptTemplate.from_template(
    template="""You are a Q&A bot.  A highly intelligent system that answers
  user questions based on information provided by the user above each question.
  If the answer cannot be found in the information provided by the user, you truthfully say:
  "I don't know'.
  If the answer is found, show the answer.
  Below the answer, show citations of the company mentioned in Citations section"""
)

human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)

st.title("Fundamental Analysis Bot")

response_container = st.container()
text_container = st.container()

# Initialize the speech recognition recognizer and microphone
recognizer = sr.Recognizer()
microphone = sr.Microphone()

# start_listening = st.button("Start Listening")


def speech_to_text():
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    # Listen for the wake word before processing commands
    listen_for_wake_word(recognizer, microphone)

    # Now, listen for the actual command
    with microphone as source:
        # print("Listening for command...")
        recognizer.adjust_for_ambient_noise(source)
        audio_data = recognizer.listen(source)

    try:
        # print("Recognizing...")
        text_input = recognizer.recognize_google(audio_data)
        st.text("You said: " + text_input)
        conversation_string = get_conversation_string()  # Replace with your conversation string
        refined_query = query_refiner(conversation_string, text_input)  # Replace with your query refinement logic
        print(refined_query)
        context = find_match(refined_query)  # Replace with your context matching logic
        response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{text_input}")
            # Convert the chatbot's response back to speech audio
        # Append the user's voice command and chatbot response to the conversation
        # llm 
        # print(text)
        print("res: ",response)
        st.text("Chatbot Response: " + response)
        text_to_speech(response,recognizer=recognizer,microphone=microphone,stop=True)
        st.session_state.requests.append(text_input)
        st.session_state.responses.append(response)

    except sr.UnknownValueError:
        print("Sorry, could not understand audio.")
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")

with response_container:
    if st.session_state['responses']:
        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i], key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True, key=str(i) + '_user')



if __name__ == "__main__":
    while True:
        speech_to_text()
