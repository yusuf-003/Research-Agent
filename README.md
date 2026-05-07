# ResearchAgent-v1.0 (Alpha)

An intelligent, document-aware research assistant built with LangGraph and Google Gemini. This is the base version (v1.0) of an ongoing project to create a full-scale automated research assistant.

## 🚀 Vision
The goal of this project is to develop a comprehensive AI agent specifically designed for academic researchers. 

Why Google Gemini? To ensure this tool is accessible to researchers worldwide without financial barriers, I have integrated Google Gemini 1.5 Flash/Pro. This allows users to leverage a massive context window and powerful reasoning for free (within Google's AI Studio tiers), making it more accessible than GPT-based alternatives.

## 📌 Version 1.0 (Current Release)
This is the "Simple SQA" (Simple Question Answering) version. It establishes the bridge between your local PDF documents and the LLM.

## ⚙️ The App Flow is given in the picture
This ResearchAgent v1.0 utilizes a state-of-the-art LangGraph architecture to transform static PDFs into interactive research assets. The workflow shown below begins by extracting raw text from a local PDF file using pypdf, which is then stored in a centralized PDFState container along with the user's research query. This state is passed to a specialized reasoning node powered by Google Gemini 1.5 Flash, where a low-temperature (0.1) configuration ensures that the generated responses are grounded strictly in the provided document to maintain high academic integrity. By leveraging Google's expansive context window, this system provides an accessible, cost-free alternative to proprietary models, establishing a scalable foundation for future multi-document analysis and automated literature reviews.
```
workflow1.jpg
```

### Features:
- **Local PDF Processing:** Extracts text from your research papers using `pypdf`.
- **Stateful Logic:** Utilizes `LangGraph` to manage the flow of information.
- **Controlled Reasoning:** Set to low temperature (0.1) for high factual accuracy.

### ⚠️ Current Limitations:
- **Single File:** Can only process one PDF file per session.
- **File Placement:** The PDF must be in the same directory as the script.
- **Single Question:** Currently optimized for one question per execution.
- **Context Limit:** Best suited for standard research papers (under 30 pages) to ensure Gemini's free tier token limits are respected.

## 🛠️ Setup Instructions

### 1. Prerequisites
- Python 3.10+
- A Google Gemini API Key (Get yours for free at [Google AI Studio](https://aistudio.google.com/))

### 2. Installation
```bash
pip install -U langgraph langchain-google-genai langchain-core pypdf python-dotenv
```

### 3. Security & Environment Variables
**IMPORTANT:** Never upload your API key to GitHub.
1. Create a `.env` file in the project root.
2. Add your key:
   ```text
   GOOGLE_API_KEY=your_gemini_api_key_here
   ```

### 4. Running the Agent
1. Place your target paper in the folder and name it `research.pdf`.
2. Execute the script:
   ```bash
   python SQA.py
   ```

---
*Created by Yusuf Aliyu - PhD Candidate in IT*
