# 📚 Smart PDF Search

An intelligent PDF search and question-answering system powered by AI. Upload PDF documents and ask questions to get accurate, context-aware answers with precise citations and source tracking.

## ✨ Features

- 🔍 **Intelligent Document Search** - Search across PDF documents using semantic understanding
- 🧠 **Context-Aware Q&A** - Get accurate answers based on document content
- 📄 **Precise Citations** - View exact page numbers and source documents for every answer
- 🖼️ **Visual Page Previews** - See highlighted results with page images
- ⚡ **Fast Retrieval** - Efficient vector-based search using ChromaDB
- 📤 **Easy Upload** - Simple drag-and-drop PDF upload interface

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- GROQ API Key (get it from [groq.com](https://groq.com))

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Smart-PDF-Search
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
# Create a .env file and add your GROQ API key
echo "GROQ_API_KEY=your_api_key_here" > .env
```

4. Create configuration:
```bash
# Ensure config.json exists with the following structure:
{
    "data_path": "data",
    "persist_directory": "db",
    "collection_name": "smart_pdf_search"
}
```

### Running the Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## 📖 Usage

1. **Upload PDF**: Use the sidebar to upload a PDF file
2. **Ask Questions**: Enter your question in the search field
3. **Get Answers**: Receive detailed answers with citations and source pages
4. **View Sources**: Click on citations to see the original page content

## 🛠️ Technology Stack

- **Streamlit** - Web application framework
- **LangChain** - LLM orchestration and retrieval
- **ChromaDB** - Vector database for embeddings
- **Groq** - Fast LLM inference (Llama 3.3 70B)
- **HuggingFace** - Embedding models (all-MiniLM-L6-v2)
- **PyMuPDF** - PDF processing

## 📝 Notes

- Only one PDF is active at a time - uploading a new PDF replaces the previous one
- The system automatically clears old documents when a new PDF is uploaded
- Ensure your PDFs contain readable text (not just images) for best results

## 🔧 Configuration

The application requires:
- `config.json` - Contains data paths and collection settings
- `.env` - Contains your GROQ_API_KEY
- `data/` - Directory for uploaded PDFs
- `db/` - Directory for vector database storage

---

**Note**: This is a POC (Proof of Concept) project for intelligent PDF search and question-answering.
