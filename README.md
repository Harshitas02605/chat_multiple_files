# chat_multiple_files
Project Summary
This project is a "Chat with PDF" application built using Streamlit.

Workflow: The user uploads one or more PDF files via the sidebar. The application extracts all text from these documents using PyPDF2 and then divides the text into manageable chunks.
Vector Search: These text chunks are converted into mathematical vectors (embeddings) and stored in a local FAISS vector database.
Conversational QA: When the user asks a question, the application retrieves the most relevant chunks from the database and passes them to a language model as context, generating a highly accurate and context-aware answer.
3. AI Tools & Frameworks Used
Google Generative AI Embeddings (models/embedding-001): Used to calculate semantic embeddings for the document chunks and user queries.
Google Gemini-Pro (gemini-pro): The Large Language Model (LLM) employed to read the context and formulate human-like, detailed answers.
LangChain: The development framework holding everything together. It is used to systematically slice text (RecursiveCharacterTextSplitter), connect to Google’s GenAI services, and manage the conversational QA sequence (load_qa_chain).
4. How to Run the Project
When I attempted to run the project, the system indicated that streamlit wasn't globally installed on your system. To run it properly, open your terminal inside the chat_multiple_pdfs directory and follow these steps to use a virtual environment:

bash
# 1. Create a virtual environment (recommended)
python3 -m venv venv
# 2. Activate the virtual environment
source venv/bin/activate
# 3. Install the required dependencies
pip install -r requirements.txt
# 4. Start the Streamlit application
streamlit run chatpdf.py
Once executed successfully, it will host the web UI locally and automatically open the application in your browser (usually at http://localhost:8501).

