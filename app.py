import streamlit as st
import json
from langchain_community.retrievers import WikipediaRetriever
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI  
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.prompts import ChatPromptTemplate 
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.schema import BaseOutputParser

with st.sidebar:
    openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")

if openai_api_key:
    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        model="gpt-3.5-turbo-1106",
        temperature=0.1,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()]
    ).bind(function_call="auto", functions=[function])
else:
    llm = ChatOpenAI(
        model="gpt-3.5-turbo-1106",
        temperature=0.1,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()]
    ).bind(function_call="auto", functions=[function])


class JsonOutputParser(BaseOutputParser):
    def parse(self, text):
        text = text.replace("```", "").replace("json", "")
        return json.loads(text)

output_parser = JsonOutputParser()

st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓",
)

st.title("QuizGPT")

difficulty = st.sidebar.selectbox(
    "Select Difficulty",
    ["Easy", "Medium", "Hard"],
    index=1
)

function = {
    "name": "create_quiz",
    "description": "Generate a quiz based on difficulty and provided content",
    "parameters": {
        "type": "object",
        "properties": {
            "difficulty": {"type": "string", "description": "Quiz difficulty level (easy, medium, hard)"},
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "answers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "answer": {"type": "string"},
                                    "correct": {"type": "boolean"},
                                },
                                "required": ["answer", "correct"],
                            },
                        },
                    },
                    "required": ["question", "answers"],
                },
            }
        },
        "required": ["difficulty", "questions"],
    },
}

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

questions_chain = {"context": format_docs} | llm
formatting_chain = questions_chain | output_parser

@st.cache_resource(show_spinner="Embedding file...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs

@st.cache_data(show_spinner="Making quiz...")
def run_quiz_chain(_docs, topic, difficulty):
    chain = {"context": _docs, "difficulty": difficulty.lower()} | formatting_chain
    return chain.invoke(_docs)

@st.cache_data(show_spinner="Searching Wikipedia...")
def wiki_search(term):
    retriever = WikipediaRetriever(top_k_results=5)
    docs = retriever.invoke(term)
    return docs

with st.sidebar:
    docs = None
    choice = st.selectbox(
        "Choose what you want to use.",
        (
            "File",
            "Wikipedia Article",
        ),
    )
    if choice == "File":
        file = st.file_uploader(
            "Upload a .docx , .txt or .pdf file",
            type=["pdf", "txt", "docx"],
        )
        if file:
            docs = split_file(file)
    else:
        topic = st.text_input("Search Wikipedia...")
        if topic:
            docs = wiki_search(topic)

if not docs:
    st.markdown(
        """
    Welcome to QuizGPT.
                
    I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.
                
    Get started by uploading a file or searching on Wikipedia in the sidebar.
    """
    )
else:
    response = run_quiz_chain(docs, topic if topic else file.name, difficulty)
    
    user_answers = []
    correct_count = 0
    total_questions = len(response["questions"])
    
    with st.form("questions_form"):
        for question in response["questions"]:
            st.write(question["question"])
            value = st.radio(
                "Select an option.",
                [answer["answer"] for answer in question["answers"]],
                index=None,
            )
            user_answers.append(value)
            
            if {"answer": value, "correct": True} in question["answers"]:
                correct_count += 1
                st.success("Correct!")
            elif value is not None:
                st.error("Wrong!")

        button = st.form_submit_button("Submit Quiz")

    if button:
        st.write(f"Your Score: {correct_count} / {total_questions}")
        
        if correct_count == total_questions:
            st.success("Congratulations! You got a perfect score! 🎉")
            st.balloons()
        else:
            st.warning("You can try again to improve your score!")
            retry = st.button("Retry Quiz")
            if retry:
                st.experimental_rerun()