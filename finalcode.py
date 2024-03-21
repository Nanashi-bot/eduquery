# !pip install langchain sentence-transformers chromadb pypdf unstructured pdf2image
# !pip install unstructured['pdf']
import streamlit as st
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.vectorstores import Chroma
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
from PyPDF2 import PdfReader

import os
from getpass import getpass

# HF_token = getpass()
HF_token = "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"

os.environ['HUGGINGFACEHUB_API_TOKEN'] = HF_token
file_path = "./MicroEco.txt"

data = UnstructuredFileLoader(file_path)
content = data.load()
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=512,chunk_overlap=0)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=100,chunk_overlap=0)
embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key = HF_token,model_name = "thenlper/gte-large"
)
vectorstore = Chroma(embedding_function=embeddings)
store = InMemoryStore()
model = HuggingFaceHub(repo_id="HuggingFaceH4/zephyr-7b-alpha",
                       model_kwargs={"temperature":0.85,"max_new_tokens":512,"max_length":64})
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)
retriever.add_documents(content,ids=None)

from langchain_core.prompts import ChatPromptTemplate
template = """
<|system|>>
You are an Ai education Doubt solver for students named EduQuery
whenever user asks you a question, you greet them first and explain the answer
of the question fairly and explain it thouroughly and in a simple manner which is understandable for
everyone. you can add a bit extra if you want own your own to explain the context and the answer
but do not deviate from the topic of the question.
the user is student so explain it simply and in full detail and make the answer organized with number points and right parsing

Give answers in points when explaining
CONTEXT: {context}
</s>
<|user|>
{query}
</s>
<|assistant|>
"""
prompt = ChatPromptTemplate.from_template(template)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
output_parser = StrOutputParser()
chain = (
    {"context": retriever, "query": RunnablePassthrough()}
    | prompt
    | model
    | output_parser
)

st.header("EDUQUERY+ :books:")

with st.sidebar:
    st.subheader("Your Notes")
    pdf_docs = st.file_uploader(
        "Upload your Notes here and click on 'Process'", accept_multiple_files=True)
    if st.button("Process"):
        with st.spinner("Processing"):
            raw_text = get_pdf_text(pdf_docs)

st.video("https://www.youtube.com/watch?v=p5jLH8A6S5c&list=PLlnEW8MeJ4z6xn7bgIvfs8sxL56QeTH6r&index=32")

question = st.text_input('Input question')
answer = (chain.invoke(question)).split("<|")
st.write('Eduquery assistant: ', answer[3][12:])
