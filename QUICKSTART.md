# 🚀 Quick Start Guide

Get your RAG system running in 5 minutes!

## Step 1: Install Ollama & Models

```bash
# Install Ollama (if not installed)
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama
ollama serve

# In another terminal, install models
ollama pull nomic-embed-text    # Required (274MB)
ollama pull llama3.2:1b        # Fast model (1.2GB)
```

## Step 2: Setup RAG System

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Make executable
chmod +x cli.py
```

## Step 3: Test the System

```bash
# Check system resources
python resource_monitor.py

# Add a test document
python cli.py add test_docs/python_guide.txt

# Ask a question
python cli.py query "What is Python?"
```

## Step 4: Add Your Documents

```bash
# Add your own documents
python cli.py add /path/to/your/document.pdf
python cli.py add /path/to/another/document.docx

# Start interactive mode
python cli.py interactive
```

## 🎯 You're Ready!

Your RAG system is now running. Key commands:

- **Add document:** `python cli.py add file.pdf`
- **Ask question:** `python cli.py query "your question"`
- **Interactive mode:** `python cli.py interactive`
- **Check resources:** `python resource_monitor.py`

## 🔧 If Something Goes Wrong

1. **Check Ollama is running:** `ollama list`
2. **Check system resources:** `python resource_monitor.py`
3. **Test simple query:** `python cli.py query "hello"`
4. **See full documentation:** `README.md`

Happy querying! 🎉