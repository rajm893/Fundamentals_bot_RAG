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
import streamlit as st
from streamlit_chat import message
from utils import *


if 'responses' not in st.session_state:
    st.session_state['responses'] = ["How can I assist you?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

if 'buffer_memory' not in st.session_state:
            st.session_state.buffer_memory=ConversationBufferWindowMemory(k=3,return_messages=True)

llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=os.getenv("OPENAI_API_KEY"))
system_msg_template = SystemMessagePromptTemplate.from_template(template="""You are a Q&A bot.  A highly intelligent system that answer
  user questions based on information provided by the user above each question.
  If the answer cannot be found  in the information provided by the user, you truthfully say:
  "I don't know'.
  If the answer is found
  show the answer,
  Below the answer show citations of the company mentioned in Citations section""")


human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm)

st.title("Fundamental Analysis Bot")

response_container = st.container()
textcontainer = st.container()


with textcontainer:

    query = st.text_input("Enter text query or click the microphone to speak", key="input")

    if st.button("🎙️ Speak"):
        query = speech_to_text()  
        
  
    if query:
        with st.spinner("typing..."):
            conversation_string = get_conversation_string()
            refined_query = query_refiner(conversation_string, query)
            context = find_match(refined_query) 
            response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}")
            text_to_speech(response)
        st.session_state.requests.append(query)
        st.session_state.responses.append(response) 
        
with response_container:
    if st.session_state['responses']:
        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i],key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True,key=str(i)+ '_user')
