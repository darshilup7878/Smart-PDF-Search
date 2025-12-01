import os
import uuid
import json
import logging
from typing import List
from config import save_config
from dotenv import load_dotenv
from log_utils import setup_logging
from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

CONFIG_FILE = 'config.json'

# Load environment variables
load_dotenv()

logger = setup_logging('upload_pdf')

def load_documents(data_path):
    """Load PDF documents from the specified directory."""
    logger.info(f"Starting document loading from directory: {data_path}")
    
    if not os.path.exists(data_path):
        logger.error(f"Directory not found: {data_path}")
        raise FileNotFoundError(f"Directory not found: {data_path}")
    
    directory_loader = DirectoryLoader(
        data_path,
        loader_cls=PyMuPDFLoader,
        show_progress=True
    )
    
    try:
        documents = directory_loader.load()
        logger.info(f"Successfully loaded {len(documents)} documents")
        return documents
    except Exception as e:
        logger.error(f"Error loading documents: {str(e)}", exc_info=True)
        raise

def store_full_content(documents):
    """Store full page content in document metadata."""
    logger.info("Starting to store full page content in metadata")
    try:
        for doc in documents:
            doc.metadata['full_page_content'] = doc.page_content
            logger.debug(f"Stored full content for page {doc.metadata.get('page', 'Unknown')} "
                        f"from {os.path.basename(doc.metadata.get('file_path', 'Unknown'))}")
        logger.info(f"Successfully stored full content for {len(documents)} documents")
        return documents
    except Exception as e:
        logger.error(f"Error storing full content: {str(e)}", exc_info=True)
        raise

def process_documents(documents):
    """Process documents into chunks and add metadata."""
    logger.info("Starting document processing")
    
    try:
        # First store full page content
        documents = store_full_content(documents)
        
        logger.info("Converting documents to chunks")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=384, chunk_overlap=20)
        chunks = text_splitter.split_documents(documents)
        
        # Add UUID and store full page content in metadata
        for chunk in chunks:
            chunk.metadata['chunk_id'] = str(uuid.uuid4())
            if 'full_page_content' not in chunk.metadata:
                chunk.metadata['full_page_content'] = chunk.metadata.get('full_page_content', chunk.page_content)
        
        logger.info(f"Document processing completed. Total chunks created: {len(chunks)}")
        return chunks
    except Exception as e:
        logger.error(f"Error processing documents: {str(e)}", exc_info=True)
        raise

def initialize_embedding_model():
    """Initialize and return the embedding model."""
    logger.info("Initializing embedding model")
    try:
        embedding_model = HuggingFaceEmbeddings(
            model_name='all-MiniLM-L6-v2',
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        logger.info("Embedding model initialized successfully")
        return embedding_model
    except Exception as e:
        logger.error(f"Error initializing embedding model: {str(e)}", exc_info=True)
        raise
    
def create_vectordb(chunks, embedding_model, persist_directory, collection_name):
    """Create and persist ChromaDB instance."""
    logger.info(f"Creating Chroma instance with collection name: {collection_name}")
    try:
        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=persist_directory,
            collection_name=collection_name
        )
        vectordb.persist()
        logger.info("Vector database created and persisted successfully")
        return vectordb
    except Exception as e:
        logger.error(f"Error creating vector database: {str(e)}", exc_info=True)
        raise

def update_or_add_pdf(uploaded_file, data_path, persist_directory, collection_name):
    """Add or replace a PDF in the system. Clears all old PDFs and vector database."""
    logger.info(f"Processing uploaded file: {uploaded_file.name}")
    
    if not uploaded_file.name.lower().endswith('.pdf'):
        logger.warning(f"Rejected non-PDF file: {uploaded_file.name}")
        return False
    
    file_path = os.path.join(data_path, uploaded_file.name)
    
    try:
        # Clear ALL existing PDFs from data directory
        if os.path.exists(data_path):
            for filename in os.listdir(data_path):
                if filename.lower().endswith('.pdf'):
                    old_file_path = os.path.join(data_path, filename)
                    os.remove(old_file_path)
                    logger.info(f"Deleted old PDF: {filename}")
        
        # Save the uploaded PDF
        os.makedirs(data_path, exist_ok=True)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getvalue())
        logger.info(f"Saved new PDF: {uploaded_file.name}")

        # Load and process ONLY the new document
        documents = load_documents(data_path)
        new_documents = [doc for doc in documents if os.path.basename(doc.metadata.get('file_path', '')) == uploaded_file.name]
        
        if not new_documents:
            logger.error(f"No documents found for uploaded file: {uploaded_file.name}")
            return False

        chunks = process_documents(new_documents)
        embedding_model = initialize_embedding_model()
        
        # Clear the entire vector database collection and recreate with only the new PDF
        try:
            # Try to get existing collection
            vectordb = Chroma(
                persist_directory=persist_directory,
                embedding_function=embedding_model,
                collection_name=collection_name
            )
            
            # Get all document IDs in the collection
            all_docs = vectordb.get()
            if all_docs and all_docs.get('ids'):
                # Delete all existing documents
                vectordb.delete(ids=all_docs['ids'])
                logger.info(f"Cleared all existing documents from vector database ({len(all_docs['ids'])} documents)")
        except Exception as e:
            # Collection might not exist yet, that's okay
            logger.info(f"Vector database collection doesn't exist yet or is empty: {str(e)}")
            # Create new collection
            vectordb = Chroma(
                persist_directory=persist_directory,
                embedding_function=embedding_model,
                collection_name=collection_name
            )
        
        # Add only the new PDF's vectors
        vectordb.add_documents(documents=chunks)
        vectordb.persist()
        logger.info(f"Successfully added {len(chunks)} chunks from {uploaded_file.name} to vector database")
        
        return True
    except Exception as e:
        logger.error(f"Error processing uploaded PDF {uploaded_file.name}: {str(e)}", exc_info=True)
        return False

def main():
    logger.info("Starting PDF processing pipeline")
    try:
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
            
        # Configuration
        data_path = config.get('data_path')
        persist_directory = os.environ.get('PERSIST_DIRECTORY')
        collection_name = config.get('collection_name')
        
        logger.info(f"Using configuration - data_path: {data_path}, "
                   f"persist_directory: {persist_directory}, "
                   f"collection_name: {collection_name}")
        
        # Save configuration
        save_config(data_path, persist_directory, collection_name)
        logger.info("Configuration saved successfully")
        
        # Process pipeline
        documents = load_documents(data_path)
        chunks = process_documents(documents)
        embedding_model = initialize_embedding_model()
        create_vectordb(chunks, embedding_model, persist_directory, collection_name)
        
        logger.info("PDF processing pipeline completed successfully!")
    
    except Exception as e:
        logger.error("Fatal error in PDF processing pipeline", exc_info=True)
        raise

if __name__ == "__main__":
    main()