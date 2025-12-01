import io
import base64
from PIL import Image
import streamlit as st
from typing import Dict, Any
from log_utils import setup_logging

logger = setup_logging('pdf_details_page')

def display_pdf_details(pdf_details, filename):
    """
    Display detailed information about a specific PDF page.
    Parameters:
    - pdf_details: dict containing the PDF information
    - filename: name of the PDF file
    """
    logger.info(f"Displaying PDF details for file: {filename}")
    
    # Initialize reader mode state
    if 'reader_mode' not in st.session_state:
        st.session_state.reader_mode = False
        logger.debug("Initialized reader mode state")

    def toggle_reader_mode():
        """Toggle reader mode state with logging."""
        previous_state = st.session_state.reader_mode
        st.session_state.reader_mode = not previous_state
        logger.info(f"Reader mode toggled from {previous_state} to {st.session_state.reader_mode}")
        
    try:
        # Enhanced CSS for better styling
        st.markdown("""
        <style>
        .page-container {
            background-color: #ffffff;
            padding: 30px;
            margin: 20px auto;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            max-width: 1200px;
            font-family: Arial, sans-serif;
        }
        .stApp {
            background-color: #f8f9fa;
        }
        .detail-box {
            border-radius: 12px;
            padding: 25px;
            margin-bottom: 25px;
        }
        .header {
            text-align: center;
            color: #1a237e;
            margin-bottom: 30px;
            font-family: 'Helvetica Neue', sans-serif;
        }
        .metadata-table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-family: 'Helvetica Neue', sans-serif;
        }
        .metadata-table td {
            padding: 12px 15px;
            border: 1px solid #e0e0e0;
        }
        .metadata-table tr:nth-child(even) {
            background-color: #f8f9fa;
        }
        .metadata-table tr:hover {
            background-color: #f5f5f5;
        }
        .metadata-table td:first-child {
            font-weight: 600;
            width: 30%;
            color: #2c3e50;
        }
        .stButton>button {
            width: 100%;
            border-radius: 8px;
            height: 45px;
            margin-top: 10px;
        }
        .stTextArea>div>div {
            border-radius: 8px;
        }
        .page-preview {
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            max-width: 100%;
            max-height: 500px;
            margin: auto;
        }
        .reader-mode {
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background: rgba(0, 0, 0, 0.9);
            z-index: 9999;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 2rem;
        }
        .reader-mode img {
            max-height: 90vh;
            max-width: 90vw;
            object-fit: contain;
        }
        </style>
        """, unsafe_allow_html=True)
        logger.debug("Applied CSS styling")
        
        # Reader mode display (if active)
        if st.session_state.reader_mode:
            logger.info("Displaying reader mode view")
            st.markdown('<div class="reader-mode-container">', unsafe_allow_html=True)
            if st.button("❌ Close Reader Mode", key="close_reader", help="Exit reader mode"):
                logger.info("Reader mode closed")
                st.session_state.reader_mode = False
                st.rerun()
            
            # Display zoomed image
            page_image_bytes = base64.b64decode(pdf_details['page_image'])
            page_image = Image.open(io.BytesIO(page_image_bytes))
            st.image(page_image, use_container_width=True, caption=f"Page {pdf_details['current_page']}")
            st.markdown('</div>', unsafe_allow_html=True)
            return
        
        logger.info("Displaying regular interface")
        # Header
        st.markdown('<h1 class="header">📚 Smart PDF Search</h1>', unsafe_allow_html=True)
        
        # Main content
        col1, col2 = st.columns([1.5, 2])
        
        with col1:
            logger.debug("Rendering details section")
            st.markdown("<h3 style='color: #1a237e; margin-bottom: 15px;'>🖼️ Page Preview</h3>", unsafe_allow_html=True)
            current_page = pdf_details['current_page']
            st.markdown(f"<div style='text-align: center; padding: 15px;'>Page {current_page} of {pdf_details['total_pages']}</div>", unsafe_allow_html=True)
            
            page_image_bytes = base64.b64decode(pdf_details['page_image'])
            page_image = Image.open(io.BytesIO(page_image_bytes))
            st.image(page_image, caption=f"Page {current_page}", use_container_width=True)
        
        with col2:
            st.markdown("<div class='detail-box'>", unsafe_allow_html=True)
            
            # Create 3 equal-width columns
            col1, col2, col3 = st.columns(3)

            # Action buttons inside the columns
            with col1:
                logger.info("Reader mode button clicked")
                st.button("📖 Reader Mode", on_click=toggle_reader_mode)
                
            with col2:
                if st.button("🔍 Ask a Question"):
                    logger.info("Ask a Question button clicked")
                    st.query_params["page"] = "home"
                    st.rerun()

            with col3:
                logger.debug("Rendering Romanized Text link")
                st.markdown(f"""
                    <a href="?page=romanized_text&filename={filename}" style="
                        display: inline-block;
                        padding: 10px 10px;
                        font-size: 16px;
                        font-weight: 400; 
                        color: white;
                        background-color: #3498db;
                        border: none;
                        border-radius: 8px; 
                        text-align: center;
                        text-decoration: none;
                        margin-top: 10px;
                        transition: all 0.3s ease;
                        text-transform: uppercase;
                        letter-spacing: 0.5px;
                        width: -webkit-fill-available;
                    ">
                        📄 Romanized Text
                    </a>
                """, unsafe_allow_html=True)      

            # Page content in expander
            with st.expander("📄 Page Content", expanded=True):
                logger.debug("Displaying page content in expander")
                st.markdown(pdf_details['page_text'], unsafe_allow_html=True)
            
            # Metadata table
            metadata_html = f"""
            <table class="metadata-table">
                <tr><td>PDF Name</td><td>{pdf_details['metadata'].get('title', filename)}</td></tr>
                <tr><td>Page</td><td>{pdf_details['current_page']}</td></tr>
                <tr><td>Author</td><td>{pdf_details['metadata'].get('author', 'N/A')}</td></tr>
                <tr><td>Total Pages</td><td>{pdf_details['total_pages']}</td></tr>
                <tr><td>Language</td><td>{pdf_details['language']}</td></tr>
                <tr><td>File Size</td><td>{pdf_details['file_size_kb']}</td></tr>
            </table>
            """
            st.markdown(metadata_html, unsafe_allow_html=True)
            logger.info(f"Completed rendering PDF details page for {filename}")
            
    except Exception as e:
        logger.error(f"Error in display_pdf_details: {str(e)}", exc_info=True)
        st.error(f"An error occurred: {e}")
        
def display_romanized_text_page(data):
    """
    Displays romanized text and PDF details in a Streamlit layout.
    Takes preprocessed data instead of filename.
    """
    logger.info(f"Displaying romanized text page for file: {data['filename']}")
    try:
        st.markdown(
            """
            <style>
            .metadata {
                display: flex;
                justify-content: space-between;
                margin-bottom: 20px;
                font-family: SFMono-Regular, Menlo, Monaco, Consolas, Liberation Mono, Courier New, monospace;
                font-size: 16px;
                color: #34495e;
                margin-top: 20px;
            }
            .metadata div {
                text-align: left;
            }
            .page-section {
                margin-bottom: 40px;
            }
            .page-header {
                font-size: 20px;
                color: #3498db;
                font-family: SFMono-Regular, Menlo, Monaco, Consolas, Liberation Mono, Courier New, monospace;
                margin-bottom: 10px;
                font-weight: bold;
            }
            .page-text {
                font-family: SFMono-Regular, Menlo, Monaco, Consolas, Liberation Mono, Courier New, monospace;
                font-size: 16px;
                color: #2c3e50;
                line-height: 1.5;
                margin-bottom: 20px;
            }
            hr {
                border: 0;
                height: 1px;
                background: #ddd;
                margin: 30px 0;
            }
            </style>
            """, 
            unsafe_allow_html=True
        )
        
        # Page Title
        st.markdown("<h1 style='text-align: center; margin-top: -1%;}'>📚 Smart PDF Search</h1>", unsafe_allow_html=True)
        
        # Document Info Section
        word_count = len(data['full_text'].split())
        st.markdown(
            f"""
            <div class='metadata'>
                <div>
                    <strong>Filename: </strong>{data['filename']} <br>
                    <strong>Total Pages: </strong>{data['total_pages']} <br>
                    <strong>File Size: </strong>{data['file_size_kb']}KB <br>
                    <strong>Total Words: </strong>{word_count}
                </div>
            </div>
            """, 
            unsafe_allow_html=True
        )   

        # Display Each Page's Text
        for page in data['pages']:
            st.markdown(
                f"""
                <div class='page-section'>
                    <div class='page-header'>Page {page['page_number']}</div>
                    <div class='page-text'>{page['text']}</div>
                    <hr>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
    except Exception as e:
        logger.error(f"Unexpected error in display_romanized_text_page: {str(e)}", exc_info=True)
        st.error(f"An unexpected error occurred: {e}")