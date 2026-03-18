import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM, OllamaEmbeddings

# ========================= CONFIG =========================
os.environ["OPENAI_API_KEY"] = "your-openai-api-key-here"   # Replace with your key

# Path to your FAQ document (supports .txt or .pdf)
FAQ_DOCUMENT_PATH = ".\\faq.txt"   # Change to "faq.pdf" if using PDF

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
# embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
embeddings = OllamaEmbeddings(model="nomic-embed-text")

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
#llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
llm = OllamaLLM(model="llama3.2:3b")   # or mistral, phi3, etc.

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