import os
from typing import TypedDict
from dotenv import load_dotenv
from pypdf import PdfReader

# Vector Search
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# NEW: Web Search Integration Tool
from langchain_community.tools.tavily_search import TavilySearchResults

# Graph & AI
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

load_dotenv()

# 1. Define the State (Updated to hold web tracking fields)
class HybridSearchState(TypedDict):
    question: str
    local_context: str
    web_context: str
    answer: str

# 2. Logic to Process Multiple PDFs (v1.2 Base)
def create_vector_db(folder_path):
    print(f"Indexing all PDFs in: {folder_path}...")
    all_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
    
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            file_path = os.path.join(folder_path, file)
            reader = PdfReader(file_path)
            
            file_text = ""
            for page in reader.pages:
                content = page.extract_text()
                if content:
                    file_text += content
            
            chunks = text_splitter.create_documents(
                texts=[file_text], 
                metadatas=[{"source": file}]
            )
            all_chunks.extend(chunks)
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = FAISS.from_documents(all_chunks, embeddings)
    return vector_db

# 3. The Smart Agent Node (Combines Local Library + Real-time Web Search)
def hybrid_research_node(state: HybridSearchState):
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
    
    # Initialize the web search engine
    web_searcher = TavilySearchResults(k=3)
    current_web_context = ""
    
    # Check if local context is empty/unhelpful, or if the user explicitly asks for recent info
    local_context_missing = "This information is not present in my local research library." in state["local_context"] or not state["local_context"].strip()
    asking_for_latest = any(word in state["question"].lower() for word in ["latest", "recent", "2026", "current", "web", "internet"])
    
    if local_context_missing or asking_for_latest:
        print("🌐 Local info insufficient or recent data requested. Searching the web...")
        try:
            search_results = web_searcher.invoke({"query": state["question"]})
            current_web_context = "\n\n".join([f"[Web Source]: {r['url']}\n{r['content']}" for r in search_results])
        except Exception as e:
            current_web_context = f"Failed to fetch live web data: {e}"

    # Advanced Hybrid Prompt Template
    template = """
    You are a professional PhD Research Assistant with access to both a local academic paper database and the live web.
    Synthesize a comprehensive answer using the provided context blocks. 
    
    Always prioritize technical methodologies from LOCAL PAPERS if available. 
    Use WEB SEARCH CONTEXT to supplement missing data, verify information, or address recent 2026 developments.

    LOCAL PAPERS CONTEXT:
    {local_context}

    WEB SEARCH CONTEXT:
    {web_context}

    QUESTION: {question}
    
    ACADEMIC ANSWER:"""
    
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm
    
    response = chain.invoke({
        "question": state["question"], 
        "local_context": state["local_context"] if state["local_context"].strip() else "No local files matched this query.",
        "web_context": current_web_context if current_web_context.strip() else "No web search executed."
    })
    return {"answer": response.content}

# 4. Build Graph
builder = StateGraph(HybridSearchState)
builder.add_node("hybrid_researcher", hybrid_research_node)
builder.set_entry_point("hybrid_researcher")
builder.add_edge("hybrid_researcher", END)
agent = builder.compile()

# 5. Execution Loop
if __name__ == "__main__":
    folder = "./pdfFolder" 
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"Created '{folder}' folder. Add PDFs and restart.")
    else:
        db = create_vector_db(folder)
        print("Knowledge base ready with hybrid capabilities!")
        
        while True:
            query = input("\nAsk your hybrid research question (or 'exit'): ")
            if query.lower() in ['exit', 'quit']: break
            
            # 1. Look up files locally first
            docs = db.similarity_search(query, k=4)
            
            context_chunks = []
            for d in docs:
                source_name = d.metadata.get('source', 'Unknown Paper')
                context_chunks.append(f"[PAPER TITLE: {source_name}]\n{d.page_content}")
            
            local_context_text = "\n\n".join(context_chunks)
            
            # 2. Run our smart hybrid agent
            result = agent.invoke({
                "question": query, 
                "local_context": local_context_text,
                "web_context": "" # Will be populated by the node if needed
            })
            
            print(f"\n[RESEARCH ANSWER]:\n{result['answer']}")