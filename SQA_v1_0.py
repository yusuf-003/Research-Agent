import os
from typing import TypedDict
from dotenv import load_dotenv
from pypdf import PdfReader


# NEW: Import from langgraph and langchain_google_genai
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage

load_dotenv()

# 1. Define the State
# This keeps track of the question and the generated answer
class PDFState(TypedDict):
    question: str
    context: str
    answer: str


# 2. Define the QA Logic 
def answer_from_pdf(state: PDFState):
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
    # This prompt tells the AI to only use the provided PDF text

    template = """
    You are a PhD Research Assistant. Use the provided context from a research paper to answer the question.
    If the answer is not in the context, say that you don't know based on this document.

    CONTEXT FROM PDF:
    {context}

    QUESTION: 
    {question}

    SCIENTIFIC ANSWER: """
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm
    
    # We pass both the question and the context into the chain
    response = chain.invoke({
        "question": state["question"], 
        "context": state["context"]
    })
    return {"answer": response.content}


# 3. Build the Graph
builder = StateGraph(PDFState)
builder.add_node("pdf_qa", answer_from_pdf)
builder.set_entry_point("pdf_qa")
builder.add_edge("pdf_qa", END)
qa_agent = builder.compile()

# Compile
qa_agent = builder.compile()

# 4. Helper function to extract text from PDF
def load_pdf_text(filepath):
    reader = PdfReader(filepath)
    text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content
    return text

# 5. Run the Agent
if __name__ == "__main__":
    file_name = "research.pdf" # Make sure this file exists in your folder!
    
    if os.path.exists(file_name):
        print(f"Reading {file_name}...")
        pdf_content = load_pdf_text(file_name)
        
        user_query = input("\nWhat would you like to know from this document? ")
        
        # We pass the question AND the PDF text into the agent
        result = qa_agent.invoke({
            "question": user_query,
            "context": pdf_content[:15000] # Sending the first ~15k characters
        })
        
        print("\n--- Answer from Document ---")
        print(result["answer"])
    else:
        print(f"Error: Could not find '{file_name}'. Please place it in the folder.")