
![Screenshot 2025-01-25 034208](https://github.com/user-attachments/assets/d574bcea-1ddc-423c-af5e-e67cd4e92148)

Here's a `README.md` template that explains the provided code. It should give you a clear understanding of how each part works:

```markdown
# Vision and Text Processing with LangChain

This project demonstrates how to load models, process images, and text, and use different API integrations to provide answers and information about images and text using LangChain and various tools such as Groq, Google Generative AI, HuggingFace, and FAISS.

## Requirements

Before running this code, make sure to install the following dependencies:

```bash
pip install requests langchain matplotlib pillow langchain_community langchain_groq langchain_huggingface langchain_google_genai dotenv langchain_text_splitters langchain_vectorstores
```

### Environment Variables

This code requires certain API keys to function properly. Set these in your `.env` file:

```
GROK_API_KEY=your_grok_api_key
HF_TOKEN=your_huggingface_token
GOOGLE_API_KEY=your_google_api_key
```

## Project Overview

The project does the following:
- Loads and processes text and image data.
- Uses a variety of models for text summarization, image analysis, and question answering.
- Leverages LangChain for model chaining, vector storage, and document retrieval.

### Key Libraries Used:

1. **LangChain**: A framework for building complex chains of operations that combine models and data.
2. **FAISS**: A library for efficient similarity search and clustering of dense vectors.
3. **HuggingFace**: For embeddings.
4. **Google Generative AI**: For large language models like Google's Gemini series.
5. **Groq**: For managing complex model calls.

## Code Walkthrough

### 1. Importing Libraries

The script begins by importing necessary libraries such as `requests`, `PIL`, and `matplotlib` for handling images and visualizations. It also imports several LangChain modules for text and document processing.

```python
import os
import requests
from PIL import Image
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.schema.document import Document
from langchain_community.document_loaders import TextLoader
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
from groq import Groq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
```

### 2. Loading Environment Variables

Environment variables (such as API keys) are loaded from the `.env` file.

```python
load_dotenv()
grok_api_key = os.getenv("GROK_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
```

### 3. Load and Display Image

The function `get_image` retrieves an image from a URL and displays it using `PIL` and `matplotlib`.

```python
def get_image(url, filename, extension):
    content = requests.get(url).content
    with open(f'content/{filename}.{extension}', 'wb') as f:
        f.write(content)
    image = Image.open(f"content/{filename}.{extension}")
    image.show()
    return image
```

### 4. Text Processing and Chunking

The function `get_text_chunks_langchain` splits long text into smaller chunks for better processing by the model.

```python
from langchain_text_splitters import CharacterTextSplitter

def get_text_chunks_langchain(text):
    text_splitter = CharacterTextSplitter(chunk_size=20, chunk_overlap=10)
    docs = [Document(page_content=x) for x in text_splitter.split_text(text)]
    return docs
```

### 5. Loading Models

The function `load_model` loads different language models based on the specified model name. This allows switching between Google's Gemini models and others.

```python
def load_model(model_name):
    if model_name == "gemini-pro":
        llm = ChatGoogleGenerativeAI(model="gemini-pro")
    else:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-8b")
    return llm
```

### 6. Text and Vision Model Interactions

The script uses both text-based and image-based inputs for querying models. For example, querying an image URL for details:

```python
message = HumanMessage(
    content=[
        {
            "type": "text",
            "text": "Provide information on given sandal image Brand, design, and model.",
        },
        {"type": "image_url", "image_url": url_1},
    ]
)
```

### 7. Embedding and Search with FAISS

Text data is embedded into vector space using Google’s embeddings, stored in a FAISS vector store, and used to perform semantic search.

```python
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = FAISS.from_documents(docs, embedding=embeddings)
retriever = vectorstore.as_retriever()
```

### 8. Using the Chain of Operations

LangChain allows chaining operations together. For example, combining the retriever, prompt, and model response into a complete process:

```python
rag_chain = (
    {"context": retriever, "query": RunnablePassthrough()}
    | prompt
    | llm_text
    | StrOutputParser()
)
result = rag_chain.invoke("can you give me a detail of nike sandal?")
```

### 9. Visualizing and Displaying Results

Results, whether text or image-based, are displayed with `matplotlib` and `IPython.display`.

```python
plt.imshow(image)
plt.show()
display(Markdown(result))
```

## Running the Code

- Make sure you have set your environment variables correctly.
- Download the required models and ensure the image URLs and text files are accessible.
- Run the script, and the model will process both image and text data, responding with the relevant information.

## Conclusion

This project showcases how to use LangChain to build advanced AI workflows that combine vision and language models for dynamic responses. Whether you're analyzing images, extracting insights from documents, or building powerful search pipelines, this structure can be adapted for various AI-driven applications.
```

This README provides a comprehensive guide to understanding the code in the script, how each part works, and what tools and models are being used.
