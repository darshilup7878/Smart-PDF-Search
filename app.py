import os
import io
import nltk
import fitz
import random
import base64
import json
import pycountry
import urllib.parse
from PIL import Image
import streamlit as st
from langdetect import detect
from config import load_config
from dotenv import load_dotenv
from nltk.corpus import stopwords
from langchain_groq import ChatGroq
from collections import defaultdict
from log_utils import setup_logging
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from langchain.chains import RetrievalQA
from upload_pdf import update_or_add_pdf 
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain_community.embeddings import HuggingFaceEmbeddings
from pdf_details_page import display_pdf_details, display_romanized_text_page

logger = setup_logging('app')

# Constants
CONFIG_FILE = 'config.json'

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

def create_dirs_if_needed():
    """Create the necessary directories if they don't exist."""
    if os.path.exists('/tmp'):
        # We're in Hugging Face space
        os.makedirs('/tmp/data', exist_ok=True)
        os.makedirs('/tmp/db', exist_ok=True)
    else:
        # Local environment
        os.makedirs('data', exist_ok=True)
        os.makedirs('db', exist_ok=True)

# Call the function at the start of your app
create_dirs_if_needed()

# Must be the first Streamlit command
st.set_page_config(
    page_title="Smart PDF Search",
    page_icon="📚",
    layout="wide"
)

# Load GROQ_API_KEY: Check st.secrets first (for Streamlit Cloud), then .env (for local)
GROQ_API_KEY = None

# Check if we're in Streamlit Cloud (secrets available without file)
# Streamlit Cloud sets these environment variables
is_streamlit_cloud = (
    os.environ.get("STREAMLIT_SHARING_MODE") is not None or
    os.environ.get("STREAMLIT_SERVER_ADDRESS") is not None
)

# Check if local secrets file exists (to avoid warning message)
secrets_paths = [
    os.path.expanduser("~/.streamlit/secrets.toml"),
    os.path.join(os.getcwd(), ".streamlit", "secrets.toml")
]
local_secrets_exists = any(os.path.exists(path) for path in secrets_paths)

# Only access st.secrets if we're in Streamlit Cloud or local secrets file exists
# This prevents the warning message when running locally without secrets file
if is_streamlit_cloud or local_secrets_exists:
    try:
        if hasattr(st, 'secrets'):
            try:
                GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
                # Strip whitespace in case there are any spaces
                if GROQ_API_KEY:
                    GROQ_API_KEY = str(GROQ_API_KEY).strip()
                    os.environ["GROQ_API_KEY"] = GROQ_API_KEY
                    logger.info(f"GROQ_API_KEY loaded from st.secrets (length: {len(GROQ_API_KEY)})")
            except KeyError:
                # Key doesn't exist in st.secrets
                logger.debug("GROQ_API_KEY not found in st.secrets, trying .env")
            except (FileNotFoundError, AttributeError) as e:
                # st.secrets file doesn't exist or other error
                logger.debug(f"Could not access st.secrets: {e}, trying .env")
    except Exception as e:
        # Any other error accessing st.secrets, will try .env next
        logger.debug(f"Exception accessing st.secrets: {e}, trying .env")

# If not found in st.secrets, fallback to .env file (for local development)
if not GROQ_API_KEY:
    load_dotenv()
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    if GROQ_API_KEY:
        GROQ_API_KEY = str(GROQ_API_KEY).strip()
        os.environ["GROQ_API_KEY"] = GROQ_API_KEY
        logger.info(f"GROQ_API_KEY loaded from .env file (length: {len(GROQ_API_KEY)})")
    elif not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not found. Please set it in Streamlit secrets (for cloud) or .env file (for local)")

# Final validation
if not GROQ_API_KEY or len(GROQ_API_KEY.strip()) == 0:
    raise ValueError("GROQ_API_KEY is empty. Please check your Streamlit secrets or .env file.")

st.markdown("""
    <style>
    img { border: 1px solid rgb(221, 221, 221); }
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    .stMarkdown {
        color: #2c3e50;
    }
    .stTextInput > div > div > input {
        border: 2px solid #3498db;
        border-radius: 12px;
        padding: 12px;
        font-size: 16px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    .stTextInput > div > div > input:focus {
        border-color: #2980b9;
        outline: none;
        box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.2);
    }
    .stButton > button {
        background-color: #3498db !important;
        color: white !important;
        border-radius: 10px;
        padding: 5px 10px !important;
        font-weight: 600;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .stButton > button:hover {
        background-color: #2980b9 !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stExpander {
        border-radius: 12px;
        background-color: #f9f9f9;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stMarkdown, .stSubheader {
        color: #34495e;
    }
    mark {
        background-color: #c6e6fb;
        color: #2c3e50;
        padding: 2px 4px;
        border-radius: 4px;
    }
    .st-emotion-cache-1104ytp h2 {
        font-size: 1rem;
        font-weight: 400;
        font-family: "Source Sans Pro", sans-serif";
        margin: 0px 0px 1rem;
        line-height: 1.6;
    }
    .st-emotion-cache-1v0mbdj.e115fcil1 {
        width: 100%;
    }
    .page-number {
        display: inline-block;
        background-color: #6C757D;
        color: white;
        font-weight: bold;
        font-size: 14px;
        padding: 2px 20px;
        border-radius: 5px;
        border: 1px solid #6C757D;
        margin-top: 0px;
        text-align: center;
    }
    .document-name {
        color: dimgray;
        font-size: 18px;
        margin-bottom: .5rem;
        font-weight: 500;
        line-height: 1.2;
        }
    .source-content {
        background-color: #f9f9f9;
        padding: 10px;
        border-radius: 5px;
    }   
    .response-block { 
        background-color: #f9f9f9; 
        padding: 15px; 
        border-radius: 5px; 
        margin-bottom: 20px; 
    }       
    </style>
    <script>
    // Disable autocomplete for text inputs, especially the question input
    function disableAutocomplete() {
        // Target all text inputs
        const inputs = document.querySelectorAll('input[type="text"]');
        inputs.forEach(input => {
            input.setAttribute('autocomplete', 'off');
            input.setAttribute('autocorrect', 'off');
            input.setAttribute('autocapitalize', 'off');
            input.setAttribute('spellcheck', 'false');
            input.setAttribute('name', 'smart-pdf-search-question');
        });
        
        // Specifically target the question input by key
        const questionInput = document.querySelector('input[data-baseweb="input"]');
        if (questionInput) {
            questionInput.setAttribute('autocomplete', 'off');
            questionInput.setAttribute('autocorrect', 'off');
            questionInput.setAttribute('autocapitalize', 'off');
            questionInput.setAttribute('spellcheck', 'false');
            questionInput.setAttribute('name', 'smart-pdf-search-question');
        }
    }
    
    // Run on page load
    window.addEventListener('load', disableAutocomplete);
    
    // Also run after Streamlit renders (for dynamic content)
    if (window.parent.postMessage) {
        window.parent.postMessage({type: 'streamlit:renderComplete'}, '*');
    }
    
    // Use MutationObserver to catch dynamically added inputs
    const observer = new MutationObserver(disableAutocomplete);
    observer.observe(document.body, { childList: true, subtree: true });
    </script>
    """, unsafe_allow_html=True)

# Initialize session state variables
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'vectordb' not in st.session_state:
    st.session_state.vectordb = None
if 'config' not in st.session_state:
    st.session_state.config = None
if 'pdf_uploaded' not in st.session_state:
    st.session_state.pdf_uploaded = False
if 'last_uploaded_file' not in st.session_state:
    st.session_state.last_uploaded_file = None

def initialize_embedding_model():
    """Initialize and return the embedding model."""
    logger.info("Initializing embedding model")
    try:
        with st.spinner('Loading embedding model...'):
            embedding_model = HuggingFaceEmbeddings(
                model_name='all-MiniLM-L6-v2',
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            # st.success("Embedding model loaded successfully")
            logger.info("Embedding model initialized successfully")
        return embedding_model
    except Exception as e:
        logger.error(f"Error initializing embedding model: {str(e)}", exc_info=True)
        raise

def load_vectordb(persist_directory, embedding_model, collection_name):
    """Load existing ChromaDB instance."""
    logger.info(f"Loading ChromaDB from {persist_directory}")
    try:
        with st.spinner('Loading ChromaDB...'):
            vectordb = Chroma(
                persist_directory=persist_directory,
                embedding_function=embedding_model,
                collection_name=collection_name
            )
            # st.success("ChromaDB loaded successfully")
            logger.info("ChromaDB loaded successfully")
        return vectordb
    except Exception as e:
        logger.error(f"Error loading ChromaDB: {str(e)}", exc_info=True)
        raise

def create_qa_chain(vectordb, groq_api_key, k=4):
    """Create and return a QA chain."""
    logger.info("Creating QA chain")
    try:
        with st.spinner('Creating QA chain...'):
            retriever = vectordb.as_retriever(search_kwargs={'k': k})
            llm = ChatGroq(model = "llama-3.3-70b-versatile", api_key=groq_api_key, temperature=0)
            
            prompt_messages = [
                ("system", """You are a helpful AI assistant who provides accurate answers based on the given context. 
                If you don't know the answer, just say that you don't know, don't try to make up an answer."""),
                ("user", """Use the following context to answer my question:
                
                Context: {context}
                
                Question: {question}"""),
                ("assistant", "I'll help answer your question based on the provided context.")
            ]

            chat_prompt = ChatPromptTemplate.from_messages(prompt_messages)

            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": chat_prompt}
            )
            # st.success("QA chain created successfully")
            logger.info("QA chain created successfully")
        return qa_chain
    except Exception as e:
        logger.error(f"Error creating QA chain: {str(e)}", exc_info=True)
        raise

def format_inline_citations(response_text, source_documents):
    """Format the response text with citations at the end of lines or paragraphs and return citations."""
    logger.info("Starting inline citations formatting")
    
    inline_response = response_text.strip()
    
    # Extract text and metadata from source documents
    try:
        doc_texts = [
            source.page_content for source in source_documents if source.page_content
        ]
        doc_citations = [
            {
                "pdf_name": os.path.basename(source.metadata.get("file_path", "Unknown")),
                "page": source.metadata.get("page", "Unknown") + 1,
            }
            for source in source_documents
        ]
        logger.debug(f"Extracted {len(doc_texts)} document texts and citations")

        if not doc_texts or not inline_response:
            logger.warning("No documents or response text to process")
            return inline_response, []

        # Split response text into paragraphs
        paragraphs = [p.strip() for p in response_text.split("\n") if p.strip()]
        logger.debug(f"Split response into {len(paragraphs)} paragraphs")

        # Vectorize response paragraphs and source document texts
        vectorizer = TfidfVectorizer()
        all_texts = doc_texts + paragraphs
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        
        # Initialize a list to store relevant citations
        relevant_citations = []

        # Match each paragraph to its most similar source documents
        for i, paragraph in enumerate(paragraphs):
            paragraph_idx = len(doc_texts) + i
            similarities = cosine_similarity(tfidf_matrix[paragraph_idx:paragraph_idx + 1], tfidf_matrix[:len(doc_texts)])[0]
            
            # Collect relevant citations based on similarity
            paragraph_citations = [
                doc_citations[j] for j, score in enumerate(similarities) if score > 0.2
            ]
            
            if paragraph_citations:
                logger.debug(f"Found {len(paragraph_citations)} citations for paragraph {i+1}")
                relevant_citations.extend(paragraph_citations)

                # Group citations by document name and collect pages
                grouped_citations = defaultdict(set)
                for citation in paragraph_citations:
                    grouped_citations[citation["pdf_name"]].add(citation["page"])

                # Format grouped citations
                combined_citations = []
                for pdf_name, pages in grouped_citations.items():
                    pages = sorted(pages)
                    pages_text = f"Page {pages[0]}" if len(pages) == 1 else f"Pages {', '.join(map(str, pages))}"
                    combined_citations.append(f"{pdf_name}: {pages_text}")

                formatted_citations = f" <b>(" + "; ".join(combined_citations) + ")</b> \n"
                paragraphs[i] = f"{paragraph}{formatted_citations}"

        # Combine paragraphs back into the final response
        inline_response = "\n".join(paragraphs)
        logger.info("Successfully formatted inline citations")
        return inline_response, relevant_citations

    except Exception as e:
        logger.error(f"Error formatting inline citations: {str(e)}", exc_info=True)
        return response_text, []

def display_citation_details(source_documents):
    """Display detailed information about citation details."""
    logger.info("Displaying citation details")
    
    try:
        st.subheader("Citation Details")

        grouped_sources = defaultdict(list)
        for source in source_documents:
            key = (source.metadata.get('file_path', 'Unknown'), source.metadata.get('page', 'Unknown'))
            grouped_sources[key].append(source.page_content)
        
        logger.debug(f"Grouped {len(grouped_sources)} unique sources")

        for key, content_list in grouped_sources.items():
            file_path, page_number = key
            try:
                full_page_content = next(
                    (source.metadata.get('full_page_content', 'No full content available') 
                     for source in source_documents
                     if source.metadata.get('file_path', 'Unknown') == file_path 
                     and source.metadata.get('page', 'Unknown') == page_number),
                    'No full content available'
                )

                merged_content = "\n".join(content_list)
                highlighted_content = full_page_content
                
                for line in merged_content.splitlines():
                    if line.strip() and line in full_page_content:
                        highlighted_content = highlighted_content.replace(line, f"<mark>{line}</mark>", 1)
                
                with st.expander(f"Source: {os.path.basename(file_path)} - Page {page_number + 1}"):
                    st.markdown(highlighted_content, unsafe_allow_html=True)
                
                logger.debug(f"Displayed citation details for {os.path.basename(file_path)} - Page {page_number + 1}")
            
            except Exception as e:
                logger.error(f"Error processing citation for {file_path}: {str(e)}")
                continue

    except Exception as e:
        logger.error(f"Error displaying citation details: {str(e)}", exc_info=True)
        st.error("Error displaying citation details")

def initialize_system():
    """Initialize the QA system components."""
    logger.info("Starting system initialization")
    
    try:
        config = load_config()
        if not config:
            logger.error("Configuration not found")
            st.error("Configuration not found. Please run the preprocessing script first.")
            return False

        st.session_state.config = config
        logger.debug("Configuration loaded successfully")

        embedding_model = initialize_embedding_model()
        st.session_state.vectordb = load_vectordb(config['persist_directory'], embedding_model, config['collection_name'])
        st.session_state.qa_chain = create_qa_chain(st.session_state.vectordb, config['groq_api_key'])
        
        logger.info("System initialized successfully")
        st.success("System initialized successfully!")
        return True

    except Exception as e:
        logger.error(f"Error during system initialization: {str(e)}", exc_info=True)
        st.error(f"An error occurred: {e}")
        return False

def extract_page_image(file_path, page_number):
    """Extract the image of a specific page from a PDF file and return it as a PIL image."""
    logger.debug(f"Extracting page image from {file_path}, page {page_number}")
    
    try:
        doc = fitz.open(file_path)
        page = doc.load_page(page_number)
        pix = page.get_pixmap()
        image = Image.open(io.BytesIO(pix.tobytes("png")))
        logger.debug("Successfully extracted page image")
        return image
    except Exception as e:
        logger.error(f"Error extracting page image: {str(e)}")
        return None
    
def highlight_query_words(text, query):
    """Highlights words from the query in the provided text."""
    logger.debug(f"Highlighting query words for query: {query}")
    
    try:
        stop_words = set(stopwords.words('english'))
        query_words = set(word_tokenize(query.lower())) - stop_words
        
        words = text.split()
        highlighted_text = " ".join(
            f"<mark>{word}</mark>" 
            if word.lower().strip(".,!?") in query_words else word
            for word in words
        )
        
        logger.debug("Successfully highlighted query words")
        return highlighted_text
    except Exception as e:
        logger.error(f"Error highlighting query words: {str(e)}")
        return text

def display_source_documents_with_images(source_documents, query):
    """Display unique source document images and formatted text snippets with query highlights."""
    logger.info("Displaying source documents with images")
    
    try:
        st.subheader("📝 Source Documents")
        
        unique_sources = {}
        for source in source_documents:
            key = (source.metadata.get('file_path', 'Unknown'), source.metadata.get('page', 'Unknown'))
            if key not in unique_sources:
                unique_sources[key] = source
        
        logger.debug(f"Processing {len(unique_sources)} unique sources")

        for (file_path, page_number), source in unique_sources.items():
            try:
                pdf_name = os.path.basename(file_path)
                page_content = source.metadata["full_page_content"] or "No content available"
                
                logger.debug(f"Processing document: {pdf_name}, page {page_number + 1}")
                
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    page_image = extract_page_image(file_path, page_number)
                    if page_image:
                        st.image(page_image, caption=f"Page {page_number + 1}", use_container_width=True)
                    else:
                        logger.warning(f"Preview not available for {pdf_name}, page {page_number + 1}")
                        st.warning("⚠️ Preview not available for this page")
                
                with col2:
                    st.markdown(f'<span class="document-name">{pdf_name}</span>', unsafe_allow_html=True)
                    st.markdown(f'<span class="page-number">Page {page_number + 1}</span>', unsafe_allow_html=True)
                    
                    sentences = sent_tokenize(page_content)
                    random.shuffle(sentences)
                    
                    selected_snippet = []
                    for sentence in sentences:
                        words = sentence.split()
                        chunked_snippet = [" ".join(words[i:i+17]) for i in range(0, len(words), 17)]
                        selected_snippet.extend(chunked_snippet)
                        if len(selected_snippet) >= 7:
                            break

                    snippet = "  ...  ".join(selected_snippet)
                    highlighted_snippet = highlight_query_words(snippet, query)
                    
                    st.markdown(f'<div class="source-content">{highlighted_snippet}</div>', unsafe_allow_html=True)

                    pdf_name = urllib.parse.quote(pdf_name)
                    
                    # Define the base URL for Hugging Face Spaces (replace this with your actual space URL)
                    # BASE_URL = "https://huggingface.co/spaces/bacancydataprophets/Smart-PDF-Search/"
                    
                    # Construct the full URL
                    url = f"{BASE_URL}?page=pdf_details&filename={pdf_name}&page_number={page_number}"
                    
                    # Use markdown to display the link
                    st.markdown(f"[View other results in this book]({url})", unsafe_allow_html=True)
                    # st.markdown(f"[View other results in this book](?page=pdf_details&filename={pdf_name}&page_number={page_number})", unsafe_allow_html=True)
                    
                    logger.debug(f"Successfully displayed content for {pdf_name}, page {page_number + 1}")
            
            except Exception as e:
                logger.error(f"Error processing document {pdf_name}: {str(e)}")
                continue

    except Exception as e:
        logger.error(f"Error displaying source documents: {str(e)}", exc_info=True)
        st.error("Error displaying source documents")

def is_query_relevant(question, source_documents, threshold=0.1):
    """Check query relevance using multiple similarity methods."""
    logger.info(f"Checking relevance for query: {question}")
    
    try:
        if not source_documents:
            logger.warning("No source documents provided for relevance check")
            return False
        
        # Keyword-based check
        keywords = set(question.lower().split())
        
        for doc in source_documents:
            doc_words = set(doc.page_content.lower().split())
            if keywords.intersection(doc_words):
                logger.debug("Query relevant based on keyword match")
                return True
        
        # TF-IDF similarity check
        try:
            doc_texts = [doc.page_content for doc in source_documents]
            texts_to_compare = doc_texts + [question]
            
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(texts_to_compare)
            
            similarities = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1])[0]
            
            is_relevant = any(sim > threshold for sim in similarities)
            logger.debug(f"Query relevance (TF-IDF): {is_relevant}")
            return is_relevant
        
        except Exception as e:
            logger.warning(f"TF-IDF similarity check failed: {str(e)}")
            # Fallback to simple text match
            is_relevant = any(question.lower() in doc.page_content.lower() for doc in source_documents)
            logger.debug(f"Query relevance (fallback): {is_relevant}")
            return is_relevant

    except Exception as e:
        logger.error(f"Error checking query relevance: {str(e)}", exc_info=True)
        return False
        
def get_pdf_details(filename, page_number):
    """Get details of a specific PDF page."""
    logger.info(f"Processing PDF details for file: {filename}, page: {page_number}")
    try:
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
            
        data_path = config.get('data_path', '/tmp/data')
        file_path = os.path.join(data_path, filename)
        
        # Open the PDF
        logger.debug(f"Opening PDF file: {file_path}")
        doc = fitz.open(file_path)
        
        # Extract full PDF text
        full_text = ""
        for page in doc:
            full_text += page.get_text()
            
        # Get PDF metadata
        pdf_metadata = doc.metadata or {}
        
        # Extract page text and render page image
        page = doc.load_page(page_number)
        page_text = page.get_text()
        
        # Render page as image
        pix = page.get_pixmap()
        img_bytes = pix.tobytes("png")
        page_image_base64 = base64.b64encode(img_bytes).decode('utf-8')
        
        # Detect language
        try:
            lang_code = detect(page_text)
            language = pycountry.languages.get(alpha_2=lang_code).name
        except Exception as e:
            logger.warning(f"Language detection failed: {str(e)}")
            language = 'Unknown'
        
        # Prepare response
        return {
            "file_path": file_path,
            "filename": os.path.basename(file_path),
            "total_pages": len(doc),
            "current_page": page_number + 1,
            "full_text": full_text,
            "page_text": page_text,
            "page_image": page_image_base64,
            "file_size_bytes": os.path.getsize(file_path),
            "file_size_kb": f"{os.path.getsize(file_path) / 1024:.2f} KB",
            "language": language,
            "metadata": {
                "title": pdf_metadata.get('title', 'Unknown'),
                "author": pdf_metadata.get('author', 'Unknown'),
                "creator": pdf_metadata.get('creator', 'Unknown'),
                "producer": pdf_metadata.get('producer', 'Unknown')
            }
        }
    
    except Exception as e:
        logger.error(f"Error processing PDF details: {str(e)}", exc_info=True)
        raise

def get_romanized_text(filename):
    """Get romanized text from a PDF."""
    logger.info(f"Processing romanized text for file: {filename}")
    try:
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
            
        data_path = config.get('data_path', '/tmp/data')
        file_path = os.path.join(data_path, filename)
        
        # Open the PDF
        logger.debug(f"Opening PDF file for romanization: {file_path}")
        doc = fitz.open(file_path)
        
        # Extract full PDF text
        full_text = ""
        pages_text = []
    
        for page in doc:
            page_text = page.get_text()
            full_text += page_text
            pages_text.append({
                "page_number": page.number + 1,
                "text": page_text
            })
                
        # Get PDF metadata
        pdf_metadata = doc.metadata or {}
        
        return {
            "filename": os.path.basename(file_path),
            "total_pages": len(doc),
            "full_text": full_text,
            "pages": pages_text, 
            "file_size_kb": f"{os.path.getsize(file_path) / 1024:.2f} KB",
            "metadata": {
                "title": pdf_metadata.get('title', 'Unknown'),
                "author": pdf_metadata.get('author', 'Unknown'),
                "creator": pdf_metadata.get('creator', 'Unknown'),
                "producer": pdf_metadata.get('producer', 'Unknown')
            }
        }
        
    except Exception as e:
        logger.error(f"Error processing romanized text: {str(e)}", exc_info=True)
        raise
    
def main():
    logger.info("Starting Smart PDF Search application")

    # Ensure directories are created before file processing starts
    create_dirs_if_needed()
    
    # Detect page from query parameters
    query_params = st.query_params
    page = query_params.get('page', 'home')
    logger.debug(f"Current page: {page}")

    encoded_filename = query_params.get('filename', '')
    filename = urllib.parse.unquote(encoded_filename)
    page_number = int(query_params.get('page_number', 0))
    
    # Routing logic
    if page == 'pdf_details':
        filename = query_params.get('filename', '')
        page_number = int(query_params.get('page_number', 0))
        logger.info(f"Displaying PDF details for {filename}, page {page_number}")
        
        if filename:
            try:
                pdf_details = get_pdf_details(filename, page_number)
                display_pdf_details(pdf_details, filename) 
            except Exception as e:
                logger.error(f"Error displaying PDF details: {str(e)}")
                st.error(f"Error displaying PDF details: {str(e)}")
                
    elif page == 'romanized_text':
        filename = query_params.get('filename', '')
        logger.info(f"Displaying romanized text for {filename}")
        
        if filename:
            try:
                romanized_data = get_romanized_text(filename)
                display_romanized_text_page(romanized_data)
            except Exception as e:
                logger.error(f"Error displaying romanized text: {str(e)}")
                st.error(f"Error displaying romanized text: {str(e)}")
        else:
            logger.warning("No filename provided for Romanized text")
            st.error("No filename provided for Romanized text")
    else:
        logger.info("Displaying main search page")
        st.markdown("<h1 style='text-align: center;'>📚 Smart PDF Search</h1>", unsafe_allow_html=True)

        # PDF Upload Section in Sidebar
        st.sidebar.header("📤 Upload PDF")
        uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf", key="pdf_uploader")
        
        # Reset QA system if no file is uploaded
        if uploaded_file is None:
            if st.session_state.pdf_uploaded:
                logger.info("PDF removed, clearing QA system")
                st.session_state.vectordb = None
                st.session_state.qa_chain = None
                st.session_state.pdf_uploaded = False
                st.session_state.last_uploaded_file = None
            
        # Process the uploaded PDF if a new file is uploaded
        if uploaded_file is not None:
            logger.info(f"Processing uploaded file: {uploaded_file.name}")
            # Only process the PDF if it's a new upload and not an existing one
            if st.session_state.last_uploaded_file != uploaded_file.name:
                try:
                    config = st.session_state.config if 'config' in st.session_state and st.session_state.config is not None else load_config()
                    
                    if config is None:
                        logger.error("Configuration not loaded. Cannot process PDF.")
                        st.sidebar.error("❌ Configuration error. Please check config.json and GROQ_API_KEY environment variable.")
                        st.session_state.pdf_uploaded = False
                    else:
                        with st.spinner('Processing uploaded PDF...'):
                            success = update_or_add_pdf(
                                uploaded_file, 
                                config['data_path'], 
                                config['persist_directory'], 
                                config['collection_name']
                            )

                        if success:
                            logger.info(f"Successfully processed uploaded file: {uploaded_file.name}")
                            st.sidebar.success(f"Successfully uploaded {uploaded_file.name}")
                            # Clear existing QA system to force reinitialization with new PDF
                            st.session_state.vectordb = None
                            st.session_state.qa_chain = None
                            st.session_state.pdf_uploaded = True
                            st.session_state.last_uploaded_file = uploaded_file.name
                        else:
                            logger.warning(f"Failed to process uploaded file: {uploaded_file.name}")
                            st.sidebar.warning("🚨 Please upload a valid PDF file to proceed.")
                            st.session_state.pdf_uploaded = False
                except Exception as e:
                    logger.error(f"Error processing uploaded file: {str(e)}", exc_info=True)
                    st.sidebar.error(f"Error processing file: {str(e)}")
                    st.session_state.pdf_uploaded = False
            else:
                logger.info(f"PDF {uploaded_file.name} is already uploaded")
                st.sidebar.info(f"PDF {uploaded_file.name} is already uploaded.")
                st.session_state.pdf_uploaded = True

        # Only initialize QA system if a PDF has been uploaded
        if not st.session_state.pdf_uploaded:
            st.info("📄 Please upload a PDF file to start asking questions.")
            st.stop()
        
        ## Initialize QA system only after PDF is uploaded
        if st.session_state.qa_chain is None:
            logger.info("Initializing QA system")
            if not initialize_system():
                logger.error("Failed to initialize system")
                st.error("Failed to initialize system. Please try uploading the PDF again.")
                st.stop()
            
        st.subheader("🔍 Ask a Question")
        question = st.text_input(
            "Enter your question:",
            key="question_input"
        )
        if st.button("Get Answer") and question:
            logger.info(f"Processing question: {question}")
            try:
                with st.spinner('🧠 Finding answer...'):
                    llm_response = st.session_state.qa_chain.invoke({"query": question})
                    logger.debug("Successfully got response from QA chain")
                    response_text = llm_response['result']
                    source_documents = llm_response['source_documents']
                    
                    # Check if the query is relevant to the documents
                    if is_query_relevant(question, source_documents):
                        # Format citations only if the query is relevant
                        inline_response, relevant_citations = format_inline_citations(response_text, source_documents)
                        
                        # Only show detailed response if we have relevant citations
                        if relevant_citations:
                            col3, col4 = st.columns([2, 1])
                            with col3:                  
                                st.subheader("🧠 Summary")
                                st.markdown(f'<div class="response-block">{inline_response}</div>', unsafe_allow_html=True)
                                display_source_documents_with_images(source_documents, question)
                            with col4:    
                                display_citation_details(source_documents)
                        else:
                            st.warning("⚠️ While your question seems related to the documents, I couldn't find specific relevant information to answer it. Please try rephrasing your question or asking about a different topic.")
                    else:
                        st.warning("⚠️ Your question appears to be unrelated to the content in the uploaded documents. Please ask a question about the information contained in the PDFs.")

            except Exception as e:
                logger.error(f"Error processing question: {str(e)}", exc_info=True)
                st.error(f"⚠️ An error occurred while processing your question: {e}")
                
        # Sidebar content
        st.sidebar.markdown("""
        <div style="background-color: #f0f4ff; padding: 5%; border-left: 4px solid #3b82f6; border-radius: 8px; box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1); margin-top: 35%; margin-bottom: 0%;">
        <h3 style="margin-top: 0;">💡 Smart PDF Search Features</h3>
            <ul style="padding-left: 20px;">
                <li>🔍 Intelligent document search across multiple PDFs</li>
                <li>🧠 Context-aware question answering</li>
                <li>📄 Precise citations and source tracking</li>
                <li>🖼️ Visual page previews with highlighted results</li>
                <li>⚡ Fast and accurate information retrieval</li>
            </ul>
        <p style="color: #1e3a8a; font-weight: bold;">
        Explore your PDFs with intelligent, context-aware search. Ask questions and get precise answers from your document collection.
        </p>
        </div>
        """, unsafe_allow_html=True)
    
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"Critical application error: {str(e)}", exc_info=True)
        st.error("A critical error occurred. Please check the logs for details.")