# Fundamental Analysis of stocks using Retrieval Augmented Generation(RAG)

- Built a conversational chatbot to help individuals perform fundamental analysis using Retrieval Augmented Generation (RAG).
- Collected  Form 10 K Annual Reports data for 3  company stocks (NVIDIA, WEWORK and PALO ALTO NETWORKS) for the last 5 years.
- Splitted the documents into chunks and converted chunks into embedding using OpenAI text-embedding-ada-002 model.
- Used Pinecone as VectorDB to upsert the chunks embeddings and metadata (Chunk text and Citations). 
- For inference, used Streamlit library to create chat interface. The input can be given via audio or text and the output would be in audio as well as text.
- Used Langchain to store the previous chat conversations and OpenAI gpt 3.5 turbo to generate the response including citations based on the given context.
- Additionally, it includes a query refiner that corrects the  incorrect or incomplete queries.
- Used Speech Recognition library that uses Google Speech API to convert speech to text and gTTS library that uses Google Translate's text-to-speech API.
- Also deployed main branch code to streamlit cloud.

![Alt text](chat_UI.png)