import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.schema.document import Document
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from PIL import Image
import requests

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def load_model(model_name):
    if model_name == "gemini-pro":
        return ChatGoogleGenerativeAI(model="gemini-pro")
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash-8b")

def get_image(url, filename, extension):
    if not os.path.exists('content'):
        os.makedirs('content')
    content = requests.get(url).content
    file_path = f'content/{filename}.{extension}'
    with open(file_path, 'wb') as f:
        f.write(content)
    return Image.open(file_path)

def setup_text_chain():
    loader = TextLoader("content/nike_shoes.txt")
    text = loader.load()[0].page_content
    text_splitter = CharacterTextSplitter(chunk_size=20, chunk_overlap=10)
    docs = [Document(page_content=x) for x in text_splitter.split_text(text)]
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(docs, embedding=embeddings)
    retriever = vectorstore.as_retriever()
    
    template = """
    ```
    {context}
    ```
    {query}
    Provide brief information and store location.
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    llm_text = load_model("gemini-pro")
    
    rag_chain = (
        {"context": retriever, "query": RunnablePassthrough()}
        | prompt
        | llm_text
        | StrOutputParser()
    )
    return rag_chain

def create_full_chain():
    llm_vision = load_model("gemini-1.5-flash-8b")
    rag_chain = setup_text_chain()
    
    full_chain = (
        RunnablePassthrough() 
        | llm_vision 
        | StrOutputParser() 
        | rag_chain
    )
    return full_chain

def main():
    st.title("Nike Shoes Analysis System")
    
    url = st.text_input("Enter shoe image URL:")
    if url:
        try:
            # Display image
            image = get_image(url, "temp", "png")
            st.image(image)
            
            # Process with vision model
            message = HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": "Analyze this shoe and provide details about its design, model, and features"
                    },
                    {"type": "image_url", "image_url": url}
                ]
            )
            
            full_chain = create_full_chain()
            
            with st.spinner("Analyzing image and retrieving information..."):
                result = full_chain.invoke([message])
                st.markdown(result)
                
        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()