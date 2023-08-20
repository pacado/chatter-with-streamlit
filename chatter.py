import os
import openai
import chromadb
import streamlit as st
from typing import List, Dict
from dotenv import load_dotenv, find_dotenv

load_status = load_dotenv(find_dotenv())

def build_prompt(query: str, context: List[str]) -> List[Dict[str, str]]:
    
    """
    Builds a prompt for the LLM. #

    This function builds a prompt for the LLM. It takes the original query,
    and the returned context, and asks the model to answer the question based only
    on what's in the context, not what's in its weights.

    More information: https://platform.openai.com/docs/guides/chat/introduction

    Args:
    query (str): The original query.
    context (List[str]): The context of the query, returned by embedding search.

    Returns:
    A prompt for the LLM (List[Dict[str, str]]).
    """

    system = {
        "role": "system",
        "content": "I am going to ask you a question, which I would like you to answer"
        "based only on the provided context, and not any other information."
        "If there is not enough information in the context to answer the question,"
        'say "I am not sure", then try to make a guess.'
        "Break your answer up into nicely readable paragraphs.",
    }
    user = {
        "role": "user",
        "content": f"The question is {query}. Here is all the context you have:"
        f'{(" ").join(context)}',
    }

    return [system, user]

def get_chatGPT_response(query: str, context: List[str]) -> str:
    """
    Queries the GPT API to get a response to the question.

    Args:
    query (str): The original query.
    context (List[str]): The context of the query, returned by embedding search.

    Returns:
    A response to the question.
    """

    openai.api_key = os.getenv("OPENAI_API_KEY")
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=build_prompt(query, context),
    )

    return response.choices[0].message.content  # type: ignore


persist_directory="documents/SS123"
client = chromadb.PersistentClient(path=persist_directory)

# Get the collection.
collection_name="SS123"
collection = client.get_collection(name=collection_name)

st.set_page_config(layout="centered")
st.title("Chat With Your Document" )

with st.form("qna"):
    query = st.text_area("Question: ",
                            value="What is the summary of this document?",
                            height=30, max_chars=8000, key="contextkey")
    
    st.write("Response: ")
    answer = st.empty()    

    if st.form_submit_button("Ask"):
        results = collection.query(
            query_texts=[query], n_results=5, include=["documents", "metadatas"]
        )

        response = get_chatGPT_response(query, results["documents"][0])
        answer.write(response)