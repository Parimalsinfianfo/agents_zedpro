# app.py
import streamlit as st
import requests
from bs4 import BeautifulSoup
import sqlite3
import pandas as pd
import re
import time
import hashlib
import smtplib
import os
import uuid
import json
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
import tempfile

# -----------------------------
# Configuration
# -----------------------------
# Set up environment variables from Streamlit secrets
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_API_KEY"] = st.secrets["LANGSMITH_API_KEY"]
os.environ["LANGSMITH_PROJECT"] = "RAG_QNA_DOC"
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
os.environ["HF_TOKEN"] = st.secrets["HF_TOKEN"]

# PDF Chat Constants
DEFAULT_MODEL = "Llama3-70b-8192"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 4000
CHUNK_OVERLAP = 500
TEMPERATURE = 0.2
MAX_TOKENS = 2048

# Email Configuration
EMAIL_CONFIG = {
    "provider": "gmail",
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 587,
    "email": st.secrets["EMAIL_USER"],
    "password": st.secrets["EMAIL_PASSWORD"],
    "enabled": True
}

# URLs to scrape for ZedPro Digital knowledge
URLS = [
    "https://www.zedprodigital.com/",
    "https://www.zedprodigital.com/our-ai-services/",
    "https://www.zedprodigital.com/projects-case-studies/",
    "https://www.zedprodigital.com/odoo-partners/",
    "https://www.zedprodigital.com/lets-get-connected/"
]

# -----------------------------
# Database Setup
# -----------------------------
def init_database():
    conn = sqlite3.connect('ai_service_agent.db')
    cursor = conn.cursor()
    
    tables = [
        '''CREATE TABLE IF NOT EXISTS visitors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT UNIQUE,
            name TEXT, email TEXT, phone TEXT, city TEXT,
            visit_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            pages_visited TEXT, time_spent INTEGER DEFAULT 0,
            status TEXT DEFAULT 'new', lead_score INTEGER DEFAULT 0,
            ip_address TEXT, user_agent TEXT
        )''',
        '''CREATE TABLE IF NOT EXISTS chat_conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT, message TEXT, response TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            message_type TEXT DEFAULT 'user'
        )''',
        '''CREATE TABLE IF NOT EXISTS company_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT, question TEXT, answer TEXT, keywords TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )''',
        '''CREATE TABLE IF NOT EXISTS email_campaigns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            visitor_id INTEGER, email_type TEXT, subject TEXT, content TEXT,
            sent_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP, status TEXT DEFAULT 'sent',
            recipient_email TEXT
        )''',
        '''CREATE TABLE IF NOT EXISTS analytics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date DATE, visitors_count INTEGER DEFAULT 0,
            leads_generated INTEGER DEFAULT 0, emails_sent INTEGER DEFAULT 0,
            conversions INTEGER DEFAULT 0, page_views INTEGER DEFAULT 0
        )''',
        '''CREATE TABLE IF NOT EXISTS page_tracking (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT, page_url TEXT, time_spent INTEGER,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )''',
        '''CREATE TABLE IF NOT EXISTS pdf_files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT,
            file_hash TEXT UNIQUE,
            upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )'''
    ]
    
    for table in tables:
        cursor.execute(table)
    
    conn.commit()
    conn.close()

def init_company_data():
    conn = sqlite3.connect('ai_service_agent.db')
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM company_data")
    if cursor.fetchone()[0] == 0:
        sample_data = [
            ("services", "What services do you offer?", "We offer AI solutions, web development, mobile app development, digital marketing, cloud services, and automation solutions.", "services, offerings, solutions"),
            ("pricing", "What are your pricing plans?", "Our pricing varies based on project scope. We offer competitive rates starting from $500 for basic solutions to $50,000+ for enterprise AI implementations.", "pricing, cost, rates"),
            ("contact", "How can I contact you?", "You can reach us via email at contact@zedprodigital.com, call us at +91-9876543210, or schedule a meeting through our website.", "contact, reach, support"),
            ("experience", "How experienced are you?", "We have over 8 years of experience in AI and digital solutions with a team of 25+ professionals serving 200+ clients globally.", "experience, years, expertise"),
            ("location", "Where are you located?", "We have offices in Mumbai, Pune, and Bangalore, with remote teams across India and international presence.", "location, office, address"),
            ("ai_services", "What AI services do you provide?", "We provide custom AI chatbots, machine learning solutions, natural language processing, computer vision, predictive analytics, and AI automation.", "AI, artificial intelligence, machine learning"),
            ("support", "Do you provide support?", "Yes, we provide 24/7 technical support, maintenance services, and dedicated account management for all our clients.", "support, maintenance, help"),
            ("industries", "Which industries do you serve?", "We serve healthcare, finance, e-commerce, education, manufacturing, real estate, and technology sectors with tailored AI solutions.", "industries, sectors, domains")
        ]
        cursor.executemany("INSERT INTO company_data (category, question, answer, keywords) VALUES (?, ?, ?, ?)", sample_data)
        conn.commit()
    conn.close()

# -----------------------------
# Utility Functions
# -----------------------------
def validate_email(email):
    return re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email) is not None

def validate_phone(phone):
    cleaned_phone = re.sub(r'[^\d+]', '', phone)
    return re.match(r'^[\+]?[1-9][\d]{3,14}$', cleaned_phone) and len(cleaned_phone) >= 10

def get_session_id():
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id

def save_visitor_data(session_id, name=None, email=None, phone=None, city=None):
    conn = sqlite3.connect('ai_service_agent.db')
    cursor = conn.cursor()
    cursor.execute('''INSERT OR REPLACE INTO visitors (session_id, name, email, phone, city, visit_time)
                     VALUES (?, ?, ?, ?, ?, ?)''', (session_id, name, email, phone, city, datetime.now()))
    conn.commit()
    conn.close()

def save_chat_message(session_id, message, response, message_type="user"):
    conn = sqlite3.connect('ai_service_agent.db')
    cursor = conn.cursor()
    cursor.execute('''INSERT INTO chat_conversations (session_id, message, response, message_type)
                     VALUES (?, ?, ?, ?)''', (session_id, message, response, message_type))
    conn.commit()
    conn.close()

def send_email(to_email, subject, body, email_type="greeting"):
    if not EMAIL_CONFIG.get('enabled', False):
        st.warning("📧 Email sending is disabled.")
        log_email_attempt(to_email, subject, body, email_type, "disabled")
        return False
    
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_CONFIG['email']
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'html'))
        
        server = smtplib.SMTP(EMAIL_CONFIG['smtp_server'], EMAIL_CONFIG['smtp_port'])
        server.starttls()
        server.login(EMAIL_CONFIG['email'], EMAIL_CONFIG['password'])
        server.sendmail(EMAIL_CONFIG['email'], to_email, msg.as_string())
        server.quit()
        
        log_email_attempt(to_email, subject, body, email_type, "sent")
        return True
    except Exception as e:
        st.error(f"📧 Email sending failed: {str(e)}")
        log_email_attempt(to_email, subject, body, email_type, f"error: {str(e)}")
        return False

def log_email_attempt(to_email, subject, body, email_type, status):
    try:
        conn = sqlite3.connect('ai_service_agent.db')
        cursor = conn.cursor()
        cursor.execute('''INSERT INTO email_campaigns (visitor_id, email_type, subject, content, status, recipient_email)
                         VALUES (?, ?, ?, ?, ?, ?)''', (0, email_type, subject, body, status, to_email))
        conn.commit()
        conn.close()
    except:
        pass

def test_email_connection():
    if not EMAIL_CONFIG.get('enabled', False):
        return False, "Email is disabled in configuration"
    
    try:
        server = smtplib.SMTP(EMAIL_CONFIG['smtp_server'], EMAIL_CONFIG['smtp_port'])
        server.starttls()
        server.login(EMAIL_CONFIG['email'], EMAIL_CONFIG['password'])
        server.quit()
        return True, "Email configuration is working correctly!"
    except smtplib.SMTPAuthenticationError:
        return False, "Authentication failed. Check your email and password/app password."
    except Exception as e:
        return False, f"Connection failed: {str(e)}"

def update_analytics():
    conn = sqlite3.connect('ai_service_agent.db')
    cursor = conn.cursor()
    today = datetime.now().date()
    
    visitors_today = cursor.execute("SELECT COUNT(*) FROM visitors WHERE DATE(visit_time) = ?", (today,)).fetchone()[0]
    leads_today = cursor.execute("SELECT COUNT(*) FROM visitors WHERE DATE(visit_time) = ? AND email IS NOT NULL", (today,)).fetchone()[0]
    emails_today = cursor.execute("SELECT COUNT(*) FROM email_campaigns WHERE DATE(sent_time) = ?", (today,)).fetchone()[0]
    
    cursor.execute('''INSERT OR REPLACE INTO analytics (date, visitors_count, leads_generated, emails_sent)
                     VALUES (?, ?, ?, ?)''', (today, visitors_today, leads_today, emails_today))
    conn.commit()
    conn.close()

# -----------------------------
# Scrape Website Text
# -----------------------------
@st.cache_data(show_spinner=True)
def scrape_website(urls):
    all_text = ""
    headers = {"User-Agent": "Mozilla/5.0"}
    for url in urls:
        try:
            res = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(res.text, 'html.parser')
            for script in soup(["script", "style", "noscript", "footer", "nav"]):
                script.extract()
            text = soup.get_text(separator=' ', strip=True)
            # Clean text by removing extra whitespace
            text = re.sub(r'\s+', ' ', text)
            all_text += text + "\n\n"
        except Exception as e:
            st.error(f"Error scraping {url}: {e}")
    return all_text

# -----------------------------
# Create QA Chain
# -----------------------------
@st.cache_resource(show_spinner=True)
def create_qa_chain():
    # Scrape website content
    raw_text = scrape_website(URLS)
    
    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_text(raw_text)
    
    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(chunks, embeddings)
    
    # Load Groq LLM
    llm = ChatGroq(
        temperature=0.7,
        model_name="llama3-8b-8192",
        max_tokens=1024
    )
    
    # Create retrieval-based QA chain
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=True
    )
    return chain

# -----------------------------
# PDF Chat Functions
# -----------------------------
def get_file_hash(file_content):
    return hashlib.md5(file_content).hexdigest()

def process_pdf(file_path):
    try:
        # Try PyMuPDF first (faster and more reliable)
        loader = PyMuPDFLoader(file_path)
        docs = loader.load()
    except Exception as e:
        st.warning(f"PyMuPDF failed, falling back to PyPDFLoader: {str(e)}")
        loader = PyPDFLoader(file_path)
        docs = loader.load()
    return docs

def generate_session_id():
    return f"pdf_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

def setup_pdf_chat():
    # Initialize session state for PDF chat
    if 'store' not in st.session_state:
        st.session_state.store = {}
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = set()
    if 'file_hashes' not in st.session_state:
        st.session_state.file_hashes = set()
    
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Model selection
    model_options = {
        "Llama3-70b-8192": "Best quality (recommended)",
        "Llama3-8b-8192": "Faster with good quality",
        "Mixtral-8x7b-32768": "Large context window",
        "Gemma2-9b-It": "Fastest but lower quality"
    }
    
    # PDF Chat UI
    st.markdown('<div class="main-header"><h1>⚡ Document Intelligence Assistant</h1></div>', unsafe_allow_html=True)
    st.markdown("Upload and analyze PDF documents with AI-powered insights")
    
    # Settings expander
    with st.expander("⚙️ PDF Processing Settings"):
        col1, col2 = st.columns(2)
        
        with col1:
            selected_model = st.selectbox(
                "Select Groq Model",
                options=list(model_options.keys()),
                index=0,
                help="Select model based on your quality/speed needs"
            )
            
            chunk_size = st.slider("Chunk Size", 1000, 10000, CHUNK_SIZE, 
                                 help="Larger chunks capture more context but may reduce precision")
            
            temperature = st.slider("Temperature", 0.0, 1.0, TEMPERATURE,
                                  help="Lower for precise answers, higher for creativity")
        
        with col2:
            chunk_overlap = st.slider("Chunk Overlap", 100, 2000, CHUNK_OVERLAP,
                                    help="Helps maintain context between chunks")
            
            max_tokens = st.slider("Max Response Tokens", 100, 4096, MAX_TOKENS,
                                  help="Maximum length of responses")
            
            search_k = st.slider("Document Chunks to Retrieve", 1, 10, 4,
                                help="Number of relevant document chunks to use for answers")
    
    # File upload and processing
    uploaded_files = st.file_uploader("Upload PDF files", type="pdf", 
                                    accept_multiple_files=True,
                                    help="Upload multiple PDFs for analysis")
    
    if uploaded_files:
        with st.spinner("🔍 Processing and indexing documents..."):
            documents = []
            new_files = []
            
            for uploaded_file in uploaded_files:
                file_content = uploaded_file.getvalue()
                file_hash = get_file_hash(file_content)
                
                if file_hash not in st.session_state.file_hashes:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                        temp_file.write(file_content)
                        temp_path = temp_file.name
                    
                    try:
                        docs = process_pdf(temp_path)
                        documents.extend(docs)
                        st.session_state.file_hashes.add(file_hash)
                        new_files.append(uploaded_file.name)
                        os.unlink(temp_path)
                        
                        # Save to database
                        conn = sqlite3.connect('ai_service_agent.db')
                        cursor = conn.cursor()
                        cursor.execute('''INSERT OR IGNORE INTO pdf_files (file_name, file_hash)
                                         VALUES (?, ?)''', (uploaded_file.name, file_hash))
                        conn.commit()
                        conn.close()
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                        continue
            
            if documents:
                # Enhanced text splitting with metadata preservation
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    separators=["\n\n", "\n", ". ", " ", ""],
                    length_function=len,
                    add_start_index=True
                )
                
                splits = text_splitter.split_documents(documents)
                
                # Create or update FAISS vector store
                if st.session_state.vectorstore is None:
                    st.session_state.vectorstore = FAISS.from_documents(splits, embedding=embeddings)
                else:
                    st.session_state.vectorstore.add_documents(splits)
                
                st.success(f"✅ Processed {len(documents)} pages from {len(new_files)} new files")
                st.session_state.processed_files.update(new_files)
    
    # Session management
    col1, col2 = st.columns(2)
    with col1:
        session_id = st.text_input("Session ID", value=generate_session_id(),
                                 help="Unique ID for your conversation session")
    with col2:
        if st.button("🔄 New Session", help="Start a fresh conversation"):
            session_id = generate_session_id()
            st.session_state.store[session_id] = ChatMessageHistory()
            st.success(f"New session created: {session_id}")
            st.rerun()
    
    # Only proceed if documents are processed
    if st.session_state.vectorstore:
        try:
            # Initialize Groq with enhanced settings
            llm = ChatGroq(
                groq_api_key=os.environ["GROQ_API_KEY"],
                model_name=selected_model,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=0.9,
                frequency_penalty=0.1,
                presence_penalty=0.1
            )
            
            # Configure retriever with MMR for diverse results
            retriever = st.session_state.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": search_k,
                    "fetch_k": min(20, search_k*3),
                    "lambda_mult": 0.5
                }
            )
            
            # Enhanced Contextual Question Reformulation Prompt
            contextualize_q_system_prompt = """You are an expert at understanding and refining questions based on conversation history. 
            
            Your responsibilities:
            1. Analyze the full conversation history and current question
            2. Identify any implicit context, references, or pronouns
            3. Reformulate the question to be completely standalone while:
               - Preserving all original intent and nuance
               - Expanding any ambiguous references
               - Maintaining technical specificity
            4. Never answer the question - only clarify and expand it
            5. For follow-up questions, ensure connection to previous context is explicit
            """
            
            contextualize_q_prompt = ChatPromptTemplate.from_messages([
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])
            
            history_aware_retriever = create_history_aware_retriever(
                llm, retriever, contextualize_q_prompt
            )

            # Optimized QA System Prompt
            qa_system_prompt = """You are an expert research assistant with deep knowledge of the provided documents. Follow these guidelines:

1. **Answer Quality**:
   - Be precise, concise, and professional
   - Use academic tone for technical content
   - Break complex answers into logical paragraphs
   - Use bullet points for lists or comparisons

2. **Source Handling**:
   - ONLY use information from the provided context
   - Cite sources with exact page numbers when possible
   - If unsure, say "The documents don't contain this information"
   - For partial information, indicate what's available

3. **Context Awareness**:
   - Maintain conversation context
   - Recognize follow-up questions
   - Connect related concepts across documents

4. **Response Structure**:
   - Start with direct answer
   - Follow with supporting evidence
   - End with potential implications or connections

Context:
{context}

Current conversation:
{chat_history}

Question: {input}
            """
            
            qa_prompt = ChatPromptTemplate.from_messages([
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])

            # Create chains with enhanced configuration
            question_answer_chain = create_stuff_documents_chain(
                llm, 
                qa_prompt,
                document_prompt=ChatPromptTemplate.from_template(
                    "Document excerpt (Page {page}):\n{page_content}\n---"
                )
            )
            
            rag_chain = create_retrieval_chain(
                history_aware_retriever, 
                question_answer_chain
            )

            conversational_rag_chain = RunnableWithMessageHistory(
                rag_chain,
                get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer"
            )

            # Chat Interface
            st.markdown("---")
            st.subheader("💬 Document Analysis Chat")
            
            # Initialize chat history
            if "messages" not in st.session_state:
                st.session_state.messages = []

            # Display chat messages
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # Accept user input
            if prompt := st.chat_input("Ask a question about your documents..."):
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt})
                # Display user message
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Display assistant response
                with st.chat_message("assistant"):
                    with st.spinner("Analyzing documents..."):
                        try:
                            session_history = get_session_history(session_id)
                            
                            response = conversational_rag_chain.invoke(
                                {"input": prompt},
                                config={"configurable": {"session_id": session_id}}
                            )
                            
                            # Display answer
                            answer = response['answer']
                            st.markdown(answer)
                            
                            # Add assistant response to chat history
                            st.session_state.messages.append({"role": "assistant", "content": answer})
                            
                            # Enhanced source display
                            if 'context' in response:
                                with st.expander("🔍 Detailed Source References"):
                                    for i, doc in enumerate(response['context']):
                                        source = os.path.basename(doc.metadata.get('source', 'Unknown'))
                                        page = doc.metadata.get('page', 'N/A')
                                        st.subheader(f"Source {i+1}: {source} (Page {page})")
                                        
                                        # Fixed score display
                                        score = doc.metadata.get('score')
                                        if isinstance(score, (float, int)):
                                            st.caption(f"Relevance score: {score:.2f}")
                                        else:
                                            st.caption("Relevance score: N/A")
                                        
                                        st.write(doc.page_content)
                                        st.markdown("---")
                        except Exception as e:
                            st.error(f"Error generating response: {str(e)}")

            # Chat history management
            st.markdown("---")
            with st.expander("📜 Session History Management"):
                if st.button("Clear Current Session History"):
                    if session_id in st.session_state.store:
                        st.session_state.store[session_id].clear()
                        st.session_state.messages = []
                        st.success("Current session history cleared!")
                        st.rerun()
                
                if st.session_state.store.get(session_id):
                    st.write(f"### Messages in session: {session_id}")
                    for msg in st.session_state.store[session_id].messages:
                        st.text(f"{msg.type.upper()}: {msg.content}")
                    
                    # Export chat history
                    export_text = "\n\n".join(
                        [f"{msg.type.upper()}:\n{msg.content}" 
                         for msg in st.session_state.store[session_id].messages]
                    )
                    
                    st.download_button(
                        label="📥 Export Full Chat History",
                        data=export_text,
                        file_name=f"chat_history_{session_id}.txt",
                        mime="text/plain"
                    )

        except Exception as e:
            st.error(f"Error initializing Groq client: {str(e)}")
    elif not os.environ["GROQ_API_KEY"]:
        st.error("Please configure your GROQ_API_KEY in Streamlit secrets")
    else:
        st.info("Upload PDF documents to begin analysis. The system will process and index them for searching.")

    # Document Management
    with st.expander("📂 Document Management"):
        if st.session_state.processed_files:
            st.write("### Processed Documents:")
            for file in st.session_state.processed_files:
                st.write(f"- {file}")
            
            if st.button("Clear All Documents"):
                st.session_state.vectorstore = None
                st.session_state.processed_files = set()
                st.session_state.file_hashes = set()
                st.success("All documents cleared. You can upload new files.")
                st.rerun()
        else:
            st.info("No documents currently processed")

# -----------------------------
# Streamlit App Configuration
# -----------------------------
st.set_page_config(
    page_title="ZedPro Digital AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    margin-bottom: 2rem;
    color: white;
}

.chat-container {
    background: #f8f9fa;
    padding: 1rem;
    border-radius: 10px;
    height: 500px;
    overflow-y: auto;
    border: 1px solid #e9ecef;
    margin-bottom: 1rem;
}

.user-message {
    background: #007bff;
    color: white;
    padding: 0.75rem 1.25rem;
    border-radius: 18px 18px 0 18px;
    margin: 0.5rem 0 0.5rem auto;
    max-width: 80%;
    display: block;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.bot-message {
    background: #e9ecef;
    color: #333;
    padding: 0.75rem 1.25rem;
    border-radius: 18px 18px 18px 0;
    margin: 0.5rem 0;
    max-width: 80%;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}

.lead-form {
    background: #ffffff;
    padding: 1.5rem;
    border-radius: 10px;
    border: 1px solid #dee2e6;
    box-shadow: 0 4px 6px rgba(0,0,0,0.05);
}

.metric-card {
    background: white;
    padding: 1.5rem;
    border-radius: 10px;
    border-left: 4px solid #667eea;
    box-shadow: 0 4px 6px rgba(0,0,0,0.05);
}

.section-title {
    border-bottom: 2px solid #667eea;
    padding-bottom: 0.5rem;
    margin-bottom: 1.5rem;
    color: #343a40;
}

.pdf-chat-container {
    background: #f0f5ff;
    padding: 1.5rem;
    border-radius: 10px;
    border: 1px solid #d0ddf0;
    margin-bottom: 1rem;
}

.pdf-settings {
    background: #e6f7ff;
    padding: 1rem;
    border-radius: 8px;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# Initialize database
init_database()
init_company_data()

# -----------------------------
# Sidebar Navigation
# -----------------------------
st.sidebar.title("🤖 ZedPro Digital AI")
st.sidebar.markdown("""
<div style="margin-bottom: 2rem;">
    <p>AI-powered assistant for ZedPro Digital services</p>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio("Navigation", [
    "🏠 Home - Chat Assistant",
    "📊 Analytics Dashboard", 
    "👥 Lead Management",
    "📧 Email Campaigns",
    "⚙️ Knowledge Base & Settings",
    "📄 Document Intelligence"
])

# -----------------------------
# Page: Home - Chat Assistant
# -----------------------------
if page == "🏠 Home - Chat Assistant":
    st.markdown('<div class="main-header"><h1>ZedPro Digital AI Assistant</h1><p>Powered by Groq & LangChain</p></div>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'visitor_info' not in st.session_state:
        st.session_state.visitor_info = {}
    
    session_id = get_session_id()
    
    # Two column layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="section-title"><h3>💬 AI-Powered Chat Assistant</h3></div>', unsafe_allow_html=True)
        
        # Initialize QA chain
        if 'qa_chain' not in st.session_state:
            with st.spinner("🔍 Loading AI knowledge base..."):
                st.session_state.qa_chain = create_qa_chain()
        
        # Chat container
        chat_container = st.container()
        
        with chat_container:
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            
            # Display chat history
            for chat in st.session_state.chat_history:
                if chat['type'] == 'user':
                    st.markdown(f'<div class="user-message">{chat["message"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="bot-message">{chat["message"]}</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Chat input
        user_message = st.text_input("Type your message here...", key="chat_input", placeholder="Ask about our services, pricing, or expertise...")
        
        col_send, col_clear = st.columns([1, 1])
        with col_send:
            if st.button("Send Message", type="primary", use_container_width=True):
                if user_message:
                    # Add user message
                    st.session_state.chat_history.append({
                        'type': 'user',
                        'message': user_message,
                        'timestamp': datetime.now()
                    })
                    
                    # Get AI response
                    with st.spinner("🤖 Thinking..."):
                        try:
                            result = st.session_state.qa_chain(user_message)
                            ai_response = result["result"]
                            
                            # Format response with bullet points if appropriate
                            if ":" in ai_response or "- " in ai_response:
                                ai_response = ai_response.replace("\n", "<br>")
                        except Exception as e:
                            ai_response = f"I'm sorry, I encountered an error. Please try again. ({str(e)})"
                    
                    # Add AI response
                    st.session_state.chat_history.append({
                        'type': 'bot',
                        'message': ai_response,
                        'timestamp': datetime.now()
                    })
                    
                    # Save to database
                    save_chat_message(session_id, user_message, ai_response)
                    st.rerun()
        
        with col_clear:
            if st.button("Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
    
    with col2:
        st.markdown('<div class="lead-form">', unsafe_allow_html=True)
        st.markdown('<div class="section-title"><h3>📝 Contact Information</h3></div>', unsafe_allow_html=True)
        
        # Lead form
        with st.form("visitor_form"):
            name = st.text_input("Name *", value=st.session_state.visitor_info.get('name', ''))
            email = st.text_input("Email *", value=st.session_state.visitor_info.get('email', ''))
            phone = st.text_input("Phone *", value=st.session_state.visitor_info.get('phone', ''))
            city = st.selectbox("City *", [
                "", "Mumbai", "Pune", "Bangalore", "Delhi", "Hyderabad",
                "Chennai", "Kolkata", "Ahmedabad", "Other"
            ], index=0)
            
            project_type = st.selectbox("Project Interest", [
                "AI Solutions", "Web Development", "Mobile App", 
                "Digital Marketing", "Cloud Services", "Consulting"
            ])
            
            submit_button = st.form_submit_button("Get Started", type="primary")
            
            if submit_button:
                errors = []
                if not name: errors.append("Name is required")
                if not email or not validate_email(email): errors.append("Valid email is required")
                if not phone or not validate_phone(phone): errors.append("Valid phone number is required")
                if not city: errors.append("City is required")
                
                if errors:
                    for error in errors:
                        st.error(error)
                else:
                    st.session_state.visitor_info = {'name': name, 'email': email, 'phone': phone, 'city': city}
                    save_visitor_data(session_id, name, email, phone, city)
                    
                    # Send welcome email
                    welcome_subject = f"Welcome to ZedPro Digital, {name}!"
                    welcome_body = f"""
                    <html>
                    <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
                        <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                            <h2 style="color: #667eea;">Welcome to ZedPro Digital, {name}!</h2>
                            <p>Thank you for your interest in our AI and digital solutions!</p>
                            <p>We specialize in:</p>
                            <ul>
                                <li>🤖 Custom AI Solutions & Chatbots</li>
                                <li>🌐 Web & Mobile Development</li>
                                <li>📊 Data Analytics & Machine Learning</li>
                                <li>☁️ Cloud Services & Automation</li>
                                <li>📱 Digital Marketing Solutions</li>
                            </ul>
                            <p>Our team will reach out to you shortly to discuss your {project_type.lower()} project requirements.</p>
                            <div style="background: #f8f9fa; padding: 15px; border-radius: 5px; margin: 20px 0;">
                                <p><strong>Quick Connect:</strong></p>
                                <p>📧 Email: contact@zedprodigital.com</p>
                                <p>📞 Phone: +91-9876543210</p>
                                <p>🌐 Website: https://www.zedprodigital.com</p>
                            </div>
                            <p>Best regards,<br><strong>ZedPro Digital Team</strong></p>
                        </div>
                    </body>
                    </html>
                    """
                    
                    if send_email(email, welcome_subject, welcome_body, "welcome"):
                        st.success("✅ Information saved! Welcome email sent.")
                    else:
                        st.success("✅ Information saved successfully!")
                    
                    update_analytics()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Quick actions
        st.markdown('<div class="section-title"><h3>🚀 Quick Actions</h3></div>', unsafe_allow_html=True)
        if st.button("📞 Connect with Live Agent", use_container_width=True):
            st.info("Our team will contact you shortly. Business hours: Mon-Fri 9AM-6PM IST")
        
        if st.button("📅 Schedule a Meeting", use_container_width=True):
            st.info("Schedule a meeting at: https://calendly.com/zedprodigital")

# -----------------------------
# Page: Analytics Dashboard
# -----------------------------
elif page == "📊 Analytics Dashboard":
    st.markdown('<div class="main-header"><h1>Analytics Dashboard</h1></div>', unsafe_allow_html=True)
    update_analytics()
    
    conn = sqlite3.connect('ai_service_agent.db')
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        visitors_today = pd.read_sql_query("SELECT COUNT(*) as count FROM visitors WHERE DATE(visit_time) = CURRENT_DATE", conn).iloc[0]['count']
        st.metric("Today's Visitors", visitors_today)
    
    with col2:
        leads_today = pd.read_sql_query("SELECT COUNT(*) as count FROM visitors WHERE DATE(visit_time) = CURRENT_DATE AND email IS NOT NULL", conn).iloc[0]['count']
        st.metric("Leads Generated", leads_today)
    
    with col3:
        emails_today = pd.read_sql_query("SELECT COUNT(*) as count FROM email_campaigns WHERE DATE(sent_time) = CURRENT_DATE", conn).iloc[0]['count']
        st.metric("Emails Sent", emails_today)
    
    with col4:
        total_visitors = pd.read_sql_query("SELECT COUNT(*) as count FROM visitors", conn).iloc[0]['count']
        st.metric("Total Visitors", total_visitors)
    
    # Charts
    st.markdown('<div class="section-title"><h3>📈 Performance Metrics</h3></div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        visitors_df = pd.read_sql_query("""SELECT DATE(visit_time) as date, COUNT(*) as visitors FROM visitors 
                                          WHERE visit_time >= date('now', '-30 days') GROUP BY DATE(visit_time) ORDER BY date""", conn)
        if not visitors_df.empty:
            fig = px.line(visitors_df, x='date', y='visitors', title='Visitors Over Time (Last 30 Days)')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No visitor data available yet")
    
    with col2:
        city_df = pd.read_sql_query("""SELECT city, COUNT(*) as count FROM visitors WHERE city IS NOT NULL 
                                      GROUP BY city ORDER BY count DESC LIMIT 10""", conn)
        if not city_df.empty:
            fig = px.pie(city_df, values='count', names='city', title='Visitors by City')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No city data available yet")
    
    # Recent activity
    st.markdown('<div class="section-title"><h3>🕒 Recent Activity</h3></div>', unsafe_allow_html=True)
    recent_visitors = pd.read_sql_query("""SELECT name, email, city, visit_time, status FROM visitors 
                                          WHERE visit_time >= datetime('now', '-7 days') ORDER BY visit_time DESC LIMIT 10""", conn)
    if not recent_visitors.empty:
        st.dataframe(recent_visitors, use_container_width=True, hide_index=True)
    else:
        st.info("No recent activity")
    
    conn.close()

# -----------------------------
# Page: Lead Management
# -----------------------------
elif page == "👥 Lead Management":
    st.markdown('<div class="main-header"><h1>Lead Management</h1></div>', unsafe_allow_html=True)
    
    conn = sqlite3.connect('ai_service_agent.db')
    
    # Lead filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        status_filter = st.selectbox("Filter by Status", ["All", "new", "contacted", "qualified", "converted"])
    
    with col2:
        city_filter = st.selectbox("Filter by City", ["All"] + [
            "Mumbai", "Pune", "Bangalore", "Delhi", "Hyderabad",
            "Chennai", "Kolkata", "Ahmedabad", "Other"
        ])
    
    with col3:
        date_filter = st.selectbox("Filter by Date", ["All", "Today", "This Week", "This Month"])
    
    # Build query
    query = "SELECT * FROM visitors WHERE email IS NOT NULL"
    params = []
    
    if status_filter != "All":
        query += " AND status = ?"
        params.append(status_filter)
    
    if city_filter != "All":
        query += " AND city = ?"
        params.append(city_filter)
    
    if date_filter == "Today":
        query += " AND DATE(visit_time) = date('now')"
    elif date_filter == "This Week":
        query += " AND visit_time >= date('now', '-7 days')"
    elif date_filter == "This Month":
        query += " AND visit_time >= date('now', '-30 days')"
    
    query += " ORDER BY visit_time DESC"
    
    leads_df = pd.read_sql_query(query, conn, params=params)
    
    if not leads_df.empty:
        st.markdown(f'<div class="section-title"><h3>📋 Leads ({len(leads_df)} total)</h3></div>', unsafe_allow_html=True)
        
        # Display leads with action buttons
        for idx, lead in leads_df.iterrows():
            with st.expander(f"{lead['name']} - {lead['email']} ({lead['status']})"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**Phone:** {lead['phone']}")
                    st.write(f"**City:** {lead['city']}")
                    st.write(f"**Visit Time:** {lead['visit_time']}")
                    
                    # Get chat history for this lead
                    chat_history = pd.read_sql_query(
                        "SELECT message, response, timestamp FROM chat_conversations WHERE session_id = ? ORDER BY timestamp",
                        conn, params=(lead['session_id'],)
                    )
                    
                    if not chat_history.empty:
                        st.write("**Chat History:**")
                        for _, chat in chat_history.iterrows():
                            st.write(f"👤 User: {chat['message']}")
                            st.write(f"🤖 Bot: {chat['response']}")
                            st.write("---")
                
                with col2:
                    new_status = st.selectbox(
                        "Update Status",
                        ["new", "contacted", "qualified", "converted"],
                        index=["new", "contacted", "qualified", "converted"].index(lead['status']),
                        key=f"status_{lead['id']}"
                    )
                    
                    if st.button(f"Update Status", key=f"update_{lead['id']}"):
                        cursor = conn.cursor()
                        cursor.execute("UPDATE visitors SET status = ? WHERE id = ?", (new_status, lead['id']))
                        conn.commit()
                        st.success("Status updated!")
                        st.rerun()
                    
                    if st.button(f"Send Follow-up Email", key=f"email_{lead['id']}"):
                        subject = f"Following up on your inquiry, {lead['name']}"
                        body = f"""
                        <html>
                        <body style="font-family: Arial, sans-serif;">
                            <h2>Hi {lead['name']},</h2>
                            <p>I hope this email finds you well!</p>
                            <p>I wanted to follow up on your recent inquiry about our services at ZedPro Digital.</p>
                            <p>We're here to help and would love to discuss how we can assist with your project.</p>
                            <p>Would you be available for a quick call this week?</p>
                            <p>Best regards,<br>ZedPro Digital Team</p>
                        </body>
                        </html>
                        """
                        
                        if send_email(lead['email'], subject, body, "follow_up"):
                            st.success("Follow-up email sent!")
                        else:
                            st.error("Failed to send email")
    else:
        st.info("No leads found matching your criteria")
    
    conn.close()

# -----------------------------
# Page: Email Campaigns
# -----------------------------
elif page == "📧 Email Campaigns":
    st.markdown('<div class="main-header"><h1>Email Campaigns</h1></div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Send Campaign", "Email Templates", "Campaign History"])
    
    with tab1:
        st.markdown('<div class="section-title"><h3>📤 Send Email Campaign</h3></div>', unsafe_allow_html=True)
        
        # Get all leads
        conn = sqlite3.connect('ai_service_agent.db')
        leads_df = pd.read_sql_query("SELECT name, email FROM visitors WHERE email IS NOT NULL", conn)
        
        if not leads_df.empty:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                campaign_type = st.selectbox("Campaign Type", [
                    "Newsletter", "Product Update", "Festival Greeting", "Follow-up", "Custom"
                ])
                
                recipient_type = st.selectbox("Send to", [
                    "All Leads", "New Leads Only", "Qualified Leads", "Custom Selection"
                ])
            
            with col2:
                subject = st.text_input("Email Subject")
                
                if campaign_type == "Festival Greeting":
                    festival = st.selectbox("Festival", [
                        "Diwali", "Christmas", "New Year", "Eid", "Thanksgiving", "Other"
                    ])
            
            # Email content
            email_content = st.text_area("Email Content (HTML supported)", height=300, 
                                       value="""<h2>Hello {name},</h2>
<p>We hope this email finds you well!</p>
<p>We wanted to share some exciting updates with you...</p>
<p>Best regards,<br>ZedPro Digital Team</p>""")
            
            if st.button("Send Campaign", type="primary"):
                if subject and email_content:
                    # Get recipients based on selection
                    if recipient_type == "All Leads":
                        recipients = leads_df
                    elif recipient_type == "New Leads Only":
                        recipients = pd.read_sql_query(
                            "SELECT name, email FROM visitors WHERE email IS NOT NULL AND status = 'new'", conn
                        )
                    elif recipient_type == "Qualified Leads":
                        recipients = pd.read_sql_query(
                            "SELECT name, email FROM visitors WHERE email IS NOT NULL AND status = 'qualified'", conn
                        )
                    else:
                        recipients = leads_df  # For now, default to all
                    
                    if not recipients.empty:
                        success_count = 0
                        progress_bar = st.progress(0)
                        
                        for idx, recipient in recipients.iterrows():
                            personalized_content = email_content.replace("{name}", recipient['name'])
                            
                            if send_email(recipient['email'], subject, personalized_content, campaign_type.lower()):
                                success_count += 1
                            
                            progress_bar.progress((idx + 1) / len(recipients))
                            time.sleep(0.1)  # Small delay to prevent rate limiting
                        
                        st.success(f"Campaign sent successfully to {success_count}/{len(recipients)} recipients!")
                    else:
                        st.warning("No recipients found matching your criteria")
                else:
                    st.error("Please fill in all required fields")
        else:
            st.info("No leads available for email campaigns")
        
        conn.close()
    
    with tab2:
        st.markdown('<div class="section-title"><h3>📝 Email Templates</h3></div>', unsafe_allow_html=True)
        
        templates = {
            "Welcome Email": """<h2>Welcome {name}!</h2>
<p>Thank you for your interest in ZedPro Digital services!</p>
<p>We're excited to help you with your project requirements.</p>
<p>Our team will be in touch soon.</p>
<p>Best regards,<br>ZedPro Digital Team</p>""",
            
            "Follow-up Email": """<h2>Hi {name},</h2>
<p>I hope this email finds you well!</p>
<p>I wanted to follow up on your recent inquiry about our services.</p>
<p>Would you be available for a quick call to discuss your requirements?</p>
<p>Best regards,<br>ZedPro Digital Team</p>""",
            
            "Festival Greeting": """<h2>Season's Greetings {name}!</h2>
<p>Wishing you and your family a wonderful festival season!</p>
<p>As we celebrate, we wanted to take a moment to thank you for your continued interest in our services.</p>
<p>May this festival bring joy, prosperity, and success to your endeavors!</p>
<p>Warm wishes,<br>ZedPro Digital Team</p>""",
            
            "Newsletter": """<h2>Monthly Newsletter - {name}</h2>
<p>Here are the latest updates from ZedPro Digital:</p>
<ul>
    <li>New AI service offerings</li>
    <li>Industry insights and trends</li>
    <li>Success stories from our clients</li>
    <li>Upcoming events and webinars</li>
</ul>
<p>Stay connected with us for more updates!</p>
<p>Best regards,<br>ZedPro Digital Team</p>"""
        }
        
        for template_name, template_content in templates.items():
            with st.expander(f"📄 {template_name}"):
                st.code(template_content, language="html")
                if st.button(f"Use {template_name}", key=f"use_{template_name}"):
                    st.session_state.selected_template = template_content
                    st.success(f"{template_name} template selected!")
    
    with tab3:
        st.markdown('<div class="section-title"><h3>📊 Campaign History</h3></div>', unsafe_allow_html=True)
        
        conn = sqlite3.connect('ai_service_agent.db')
        campaigns_df = pd.read_sql_query("""
            SELECT email_type, subject, sent_time, status, COUNT(*) as recipient_count
            FROM email_campaigns
            GROUP BY email_type, subject, DATE(sent_time)
            ORDER BY sent_time DESC
        """, conn)
        
        if not campaigns_df.empty:
            st.dataframe(campaigns_df, use_container_width=True, hide_index=True)
            
            # Show error analysis
            error_emails = campaigns_df[campaigns_df['status'].str.contains('error|failed', case=False, na=False)]
            if not error_emails.empty:
                st.warning(f"⚠️ {len(error_emails)} email(s) failed to send. Check your email configuration.")
        else:
            st.info("No email campaigns sent yet")
        
        conn.close()

# -----------------------------
# Page: Knowledge Base & Settings
# -----------------------------
elif page == "⚙️ Knowledge Base & Settings":
    st.markdown('<div class="main-header"><h1>Knowledge Base & Settings</h1></div>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Company Knowledge", "Email Settings"])
    
    with tab1:
        st.markdown('<div class="section-title"><h3>🧠 Company Knowledge Base</h3></div>', unsafe_allow_html=True)
        
        # Add new knowledge
        with st.expander("➕ Add New Knowledge"):
            with st.form("add_knowledge"):
                category = st.selectbox("Category", [
                    "services", "pricing", "contact", "experience", "location", "support", "other"
                ])
                question = st.text_input("Question")
                answer = st.text_area("Answer")
                keywords = st.text_input("Keywords (comma-separated)")
                
                if st.form_submit_button("Add Knowledge"):
                    if question and answer:
                        conn = sqlite3.connect('ai_service_agent.db')
                        cursor = conn.cursor()
                        cursor.execute('''
                            INSERT INTO company_data (category, question, answer, keywords)
                            VALUES (?, ?, ?, ?)
                        ''', (category, question, answer, keywords))
                        conn.commit()
                        conn.close()
                        st.success("Knowledge added successfully!")
                        st.rerun()
                    else:
                        st.error("Please fill in all required fields")
        
        # Display existing knowledge
        st.markdown('<div class="section-title"><h3>📚 Existing Knowledge Base</h3></div>', unsafe_allow_html=True)
        conn = sqlite3.connect('ai_service_agent.db')
        knowledge_df = pd.read_sql_query("SELECT * FROM company_data ORDER BY category, id", conn)
        
        if not knowledge_df.empty:
            for idx, item in knowledge_df.iterrows():
                with st.expander(f"{item['category'].title()}: {item['question']}"):
                    st.write(f"**Answer:** {item['answer']}")
                    st.write(f"**Keywords:** {item['keywords']}")
                    
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        if st.button(f"Edit", key=f"edit_{item['id']}"):
                            st.session_state[f"editing_{item['id']}"] = True
                    
                    with col2:
                        if st.button(f"Delete", key=f"delete_{item['id']}"):
                            cursor = conn.cursor()
                            cursor.execute("DELETE FROM company_data WHERE id = ?", (item['id'],))
                            conn.commit()
                            st.success("Knowledge deleted!")
                            st.rerun()
                    
                    # Edit form
                    if st.session_state.get(f"editing_{item['id']}", False):
                        with st.form(f"edit_form_{item['id']}"):
                            new_question = st.text_input("Question", value=item['question'])
                            new_answer = st.text_area("Answer", value=item['answer'])
                            new_keywords = st.text_input("Keywords", value=item['keywords'])
                            
                            col1, col2 = st.columns([1, 1])
                            with col1:
                                if st.form_submit_button("Save Changes"):
                                    cursor = conn.cursor()
                                    cursor.execute('''
                                        UPDATE company_data 
                                        SET question = ?, answer = ?, keywords = ?
                                        WHERE id = ?
                                    ''', (new_question, new_answer, new_keywords, item['id']))
                                    conn.commit()
                                    st.session_state[f"editing_{item['id']}"] = False
                                    st.success("Knowledge updated!")
                                    st.rerun()
                            
                            with col2:
                                if st.form_submit_button("Cancel"):
                                    st.session_state[f"editing_{item['id']}"] = False
                                    st.rerun()
        
        conn.close()
    
    with tab2:
        st.markdown('<div class="section-title"><h3>📧 Email Configuration</h3></div>', unsafe_allow_html=True)
        
        # Email provider selection
        current_provider = EMAIL_CONFIG.get('provider', 'gmail')
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.write("**Current Email Configuration:**")
            st.info(f"Provider: {current_provider.title()}")
            st.info(f"Status: {'✅ Enabled' if EMAIL_CONFIG.get('enabled') else '❌ Disabled'}")
            st.info(f"Email: {EMAIL_CONFIG.get('email', 'Not configured')}")
        
        with col2:
            st.write("**Test Email Connection:**")
            if st.button("🧪 Test Email Configuration"):
                if EMAIL_CONFIG.get('enabled'):
                    success, message = test_email_connection()
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
                else:
                    st.warning("Email is disabled. Enable it first to test.")
        
        # Email setup instructions
        st.write("**📋 Email Setup Instructions:**")
        st.markdown("""
        1. **Gmail Users:**
           - Enable 2-Factor Authentication
           - Generate App Password: Google Account → Security → 2-Step Verification → App passwords
           - Use the 16-character app password (not your regular password)
        
        2. **Other Providers:**
           - Use your full email address and password
           - Ensure SMTP access is enabled in your email settings
        """)
        
        # Email logs
        st.markdown('<div class="section-title"><h3>📊 Recent Email Attempts</h3></div>', unsafe_allow_html=True)
        
        conn = sqlite3.connect('ai_service_agent.db')
        recent_emails = pd.read_sql_query("""
            SELECT email_type, subject, sent_time, status
            FROM email_campaigns
            ORDER BY sent_time DESC
            LIMIT 10
        """, conn)
        conn.close()
        
        if not recent_emails.empty:
            st.dataframe(recent_emails, use_container_width=True, hide_index=True)
            
            # Show error analysis
            error_emails = recent_emails[recent_emails['status'].str.contains('error|failed', case=False, na=False)]
            if not error_emails.empty:
                st.warning(f"⚠️ {len(error_emails)} email(s) failed to send. Check your email configuration.")
        else:
            st.info("No email attempts recorded yet.")

# -----------------------------
# Page: Document Intelligence
# -----------------------------
elif page == "📄 Document Intelligence":
    setup_pdf_chat()

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>🤖 ZedPro Digital AI Assistant v3.0 | Built with Streamlit, Groq AI, and SQLite</p>
    <p>© 2024 ZedPro Digital. All rights reserved.</p>
</div>
""", unsafe_allow_html=True)

# Auto-refresh for real-time updates
if st.sidebar.checkbox("Auto-refresh (30s)", False):
    time.sleep(30)
    st.rerun()