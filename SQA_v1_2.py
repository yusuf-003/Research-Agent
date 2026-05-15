import os
from typing import TypedDict
from dotenv import load_dotenv
from pypdf import PdfReader



# Vector Search
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# NEW: Import from langgraph and langchain_google_genai
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage

load_dotenv()

# 1. Define the State
class MultiPDFState(TypedDict):
    question: str
    context: str
    answer: str


# 2. Logic to Process Multiple PDFs
def create_vector_db(folder_path):
    print(f"Indexing all PDFs in: {folder_path}...")
    documents = []
    
    # Loop through all PDFs in the folder
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            reader = PdfReader(os.path.join(folder_path, file))
            for page in reader.pages:
                content = page.extract_text()
                if content:
                    documents.append(content)
    
    # Split text into smaller chunks for better accuracy
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.create_documents(documents)
    
    # Create Embeddings (Free, runs locally)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Build the FAISS database
    vector_db = FAISS.from_documents(texts, embeddings)
    return vector_db
    folder_path= "/pdfFolders"


# 3. The Agent Node
def research_node(state: MultiPDFState):
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
    
    template = """
    You are a PhD Research Assistant. Answer based ONLY on the provided research context.
    If you cannot find the answer, suggest what the user might look for or say not in context.

    CONTEXT:
    {context}

    QUESTION: {question}
    """
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm
    
    response = chain.invoke({"question": state["question"], "context": state["context"]})
    return {"answer": response.content}

# 4. Build Graph
builder = StateGraph(MultiPDFState)
builder.add_node("researcher", research_node)
builder.set_entry_point("researcher")
builder.add_edge("researcher", END)
agent = builder.compile()

# 5. Execution Loop
if __name__ == "__main__":
    folder = "./pdfFolder" # Create this folder and put your PDFs there
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"Created '{folder}' folder. Please add PDFs and restart.")
    else:
        db = create_vector_db(folder)
        print("Knowledge base ready!")
        
        while True:
            query = input("\nAsk about your papers (or 'exit'): ")
            if query.lower() in ['exit', 'quit']: break
            
            # Increase k to 6 to get more context
            docs = db.similarity_search(query, k=6)
            
            # NEW: Add the filename to the context so the Agent knows which paper is which
            context_chunks = []
            for d in docs:
                source_name = os.path.basename(d.metadata.get('source', 'Unknown'))
                context_chunks.append(f"[Source: {source_name}]\n{d.page_content}")
            
            context_text = "\n\n".join(context_chunks)
            
            # Run the agent
            result = agent.invoke({"question": query, "context": context_text})
            print(f"\n[RESEARCH ANSWER]:\n{result['answer']}")