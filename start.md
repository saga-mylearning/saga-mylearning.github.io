write a simple RAG using langchain read an FAQ document and respons to user queriesHere's a simple RAG (Retrieval-Augmented Generation) implementation using LangChain in Python. It reads an FAQ document (e.g., faq.txt or faq.pdf) and answers user queries based on its content.

1. Install the required packages
Bashpip install langchain langchain-community langchain-openai langchain-chroma pypdf  # for PDF support
# or use: pip install langchain langchain-community langchain-ollama chromadb  # for local models
2. Simple RAG Script (using OpenAI)
Create a file called rag_faq.py:
Pythonimport os
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ========================= CONFIG =========================
os.environ["OPENAI_API_KEY"] = "your-openai-api-key-here"   # Replace with your key

# Path to your FAQ document (supports .txt or .pdf)
FAQ_DOCUMENT_PATH = "faq.txt"   # Change to "faq.pdf" if using PDF

# =========================================================

# Step 1: Load the document
if FAQ_DOCUMENT_PATH.endswith(".pdf"):
    loader = PyPDFLoader(FAQ_DOCUMENT_PATH)
else:
    loader = TextLoader(FAQ_DOCUMENT_PATH, encoding="utf-8")

documents = loader.load()

# Step 2: Split into chunks (good for FAQs)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)
chunks = text_splitter.split_documents(documents)

print(f"Loaded {len(documents)} document(s) and split into {len(chunks)} chunks.")

# Step 3: Create embeddings and vector store (Chroma)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_faq_db"   # persists to disk
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})  # retrieve top 4 chunks

# Step 4: Define the prompt template
prompt_template = ChatPromptTemplate.from_template(
    """You are a helpful FAQ assistant. Answer the question based ONLY on the following context.
If you don't know the answer, just say "I don't have information about that in the FAQ."

Context:
{context}

Question: {question}
Answer:"""
)

# Step 5: Set up the LLM and RAG chain
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt_template
    | llm
    | StrOutputParser()
)

# ========================= Interactive Query Loop =========================
print("\n✅ FAQ RAG system is ready! Type 'exit' to quit.\n")

while True:
    query = input("Your question: ").strip()
    if query.lower() in ["exit", "quit", "bye"]:
        print("Goodbye!")
        break
    if not query:
        continue

    answer = rag_chain.invoke(query)
    print(f"\nAnswer: {answer}\n")
How to Run It

Create a text file named faq.txt (or PDF) with your FAQ content, for example:

textQ: What are your business hours?
A: We are open Monday to Friday from 9 AM to 6 PM.

Q: How do I reset my password?
A: Go to the login page and click "Forgot Password". ...

Q: What is the return policy?
A: You can return items within 30 days of purchase...

Replace "your-openai-api-key-here" with your actual key.
Run the script:

Bashpython rag_faq.py
Alternative: Using Local LLM (Ollama) – No API Key Needed
Replace the LLM and embeddings part with:
Pythonfrom langchain_ollama import OllamaLLM, OllamaEmbeddings

embeddings = OllamaEmbeddings(model="nomic-embed-text")
llm = OllamaLLM(model="llama3.2")   # or mistral, phi3, etc.
Make sure you have Ollama running with the models pulled.
Tips for Better Performance with FAQs

Use smaller chunk_size (500–1000) since FAQs are usually short and structured.
Try MMR retriever for more diverse results: retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 20})
Add metadata to chunks if you want to show sources.

This is a clean, minimal, and production-ready starting point for a FAQ chatbot using LangChain RAG.
Would you like a Streamlit web UI version, PDF-only version, or support for multiple FAQ documents? Just let me know!from langchain_community.document_loaders import TextLoader, PyPDFLoader
ModuleNotFoundError: No module named 'langchain_community'Fixed! ✅
The error ModuleNotFoundError: No module named 'langchain_community' happens because LangChain split its packages. You need to install langchain-community separately.
Quick Fix (Run this command)
Bashpip install langchain-community
Best Practice – Install All Required Packages at Once
Run this single command to install everything you need for the RAG script:
Bashpip install langchain langchain-community langchain-openai langchain-chroma pypdf
If you're using Ollama (local models) instead of OpenAI, use:
Bashpip install langchain langchain-community langchain-ollama langchain-chroma pypdf
After Installing

Close and reopen your terminal / VS Code / Jupyter Notebook (important!)
Run your script again:

Bashpython rag_faq.py
Still Getting the Error?
Try these extra steps:
Bash# Upgrade pip first
pip install --upgrade pip

# Force reinstall
pip install --upgrade --force-reinstall langchain-community langchain

# If using virtual environment (recommended)
python -m pip install langchain-community
Or if you're using Conda:
Bashconda install -c conda-forge langchain-community
Would you like me to give you the updated full script with the latest recommended imports (as of 2026)?
Also, tell me:

Are you using OpenAI or Ollama/local model?
Are you working in VS Code, Jupyter, or terminal?

Just reply and I'll give you the cleanest version.