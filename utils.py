from pinecone import Pinecone
from openai import OpenAI
import streamlit as st
from dotenv import load_dotenv
import textwrap
import os
import json
import speech_recognition as sr
import pyaudio
from gtts import gTTS
import io
import pygame

load_dotenv()
with open("config.json", 'r') as json_file:
    cites = json.load(json_file)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENVIRONMENT"))
index = pc.Index(os.getenv('PINECONE_INDEX'))

def query_refiner(conversation, query):
    temperature_ = 0.7

    completion = client.chat.completions.create(
        model = "gpt-3.5-turbo",
        temperature = temperature_,
        messages = [
            {
                "role":"system",
                "content": f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}"
            },
            {
                "role":"user",
                "content": f"\n\nQuery: {query}\n\nRefined Query:",
            }
        ]
    )
    lines = (completion.choices[0].message.content).split("\n")
    lists = (textwrap.TextWrapper(width=90, break_long_words=False).wrap(line) for line in lines)
    return "\n".join("\n".join(list) for list in lists)

def get_embedding(text, model= "text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=text, model=model).data[0].embedding

def find_match(input_):
    query_embedding = get_embedding(input_, model='text-embedding-ada-002')
    # input_em = model.encode(input).tolist()
    result = index.query(vector=query_embedding, top_k=2, includeMetadata=True)
    citations = "Citations: "+cites[result['matches'][0]['metadata']['citation']]+ \
                                "\n"+ cites[result['matches'][1]['metadata']['citation']]
    return result['matches'][0]['metadata']['text']+"\n"+ \
            result['matches'][1]['metadata']['text']+"\n"+citations

def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):        
        conversation_string += "Human: "+st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: "+ st.session_state['responses'][i+1] + "\n"
    return conversation_string

def listen_for_wake_word(recognizer, microphone):
    print("Listening for 'Friday'...")

    while True:
        with microphone as source:
            recognizer.adjust_for_ambient_noise(source)
            audio_data = recognizer.listen(source, timeout=10, phrase_time_limit=5)

        try:
            wake_word = recognizer.recognize_google(audio_data).lower()
            if "friday" in wake_word:
                text_to_speech('Uh huh?', recognizer=recognizer, microphone=microphone)
                print("Uh Huh?")
                return True
        except sr.UnknownValueError:
            pass
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")
    return False

def listen_for_stop_word(recognizer, microphone):
    print("Listening for 'Stop'...")

    while True:
        with microphone as source:
            recognizer.adjust_for_ambient_noise(source)
            audio_data = recognizer.listen(source, timeout=100, phrase_time_limit=5)

        try:
            wake_word = recognizer.recognize_google(audio_data).lower()
            if "stop" in wake_word:
                print("Stop word detected. You can now speak your command.")
                pygame.mixer.music.stop()
                return
        except sr.UnknownValueError:
            pass
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")



def text_to_speech(text, recognizer, microphone, language='en', stop=False):
    tts = gTTS(text=text, lang=language)
    buffer = io.BytesIO()
    tts.write_to_fp(buffer)
    buffer.seek(0)
    print(buffer)
    pygame.mixer.init()
    pygame.mixer.music.load(buffer)
    pygame.mixer.music.play()
    if stop:
        listen_for_stop_word(recognizer, microphone)

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
        text = recognizer.recognize_google(audio_data)
        # llm 
        print(text)
        
        text_to_speech('LLM Response',recognizer=recognizer,microphone=microphone,stop=True)
        print("You said:", text)
    except sr.UnknownValueError:
        print("Sorry, could not understand audio.")
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")