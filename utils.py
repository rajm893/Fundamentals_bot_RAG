from pinecone import Pinecone
from openai import OpenAI
import streamlit as st
from dotenv import load_dotenv
import textwrap
import os

load_dotenv()
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
    return result['matches'][0]['metadata']['text']+"\n"+result['matches'][1]['metadata']['text']

def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):        
        conversation_string += "Human: "+st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: "+ st.session_state['responses'][i+1] + "\n"
    return conversation_string