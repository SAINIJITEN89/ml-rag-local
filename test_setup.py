#!/usr/bin/env python3

import requests
import sys
import json

def test_ollama_connection():
    """Test if Ollama is running and accessible"""
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json()
            print("✓ Ollama is running")
            print("Available models:")
            for model in models.get('models', []):
                print(f"  - {model['name']}")
            return True
        else:
            print("✗ Ollama is not responding properly")
            return False
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to Ollama at localhost:11434")
        print("  Make sure Ollama is running or use port forwarding if on remote server")
        return False

def test_qwen3_model():
    """Test if Qwen3:14b model is available and working"""
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "qwen3:14b",
                "prompt": "Hello, respond with just 'Working!'",
                "stream": False
            }
        )
        if response.status_code == 200:
            result = response.json()
            print("✓ Qwen3:14b model is working")
            print(f"  Response: {result['response'].strip()}")
            return True
        else:
            print("✗ Qwen3:14b model failed to respond")
            print(f"  Status: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Error testing Qwen3:14b: {e}")
        return False

def test_embedding_model():
    """Test if embedding model is available"""
    try:
        response = requests.post(
            "http://localhost:11434/api/embeddings",
            json={
                "model": "nomic-embed-text",
                "prompt": "test"
            }
        )
        if response.status_code == 200:
            result = response.json()
            if "embedding" in result:
                print("✓ nomic-embed-text model is working")
                print(f"  Embedding dimension: {len(result['embedding'])}")
                return True
        print("✗ nomic-embed-text model not working")
        return False
    except Exception as e:
        print(f"✗ Error testing embedding model: {e}")
        return False

def main():
    print("Testing RAG System Setup")
    print("=" * 30)
    
    all_good = True
    
    all_good &= test_ollama_connection()
    print()
    
    all_good &= test_qwen3_model()
    print()
    
    all_good &= test_embedding_model()
    print()
    
    if all_good:
        print("🎉 All tests passed! Your setup is ready.")
        print("\nTo get started:")
        print("1. pip install -r requirements.txt")
        print("2. python cli.py add your_document.pdf")
        print("3. python cli.py query 'your question'")
    else:
        print("❌ Some tests failed. Please check:")
        print("- Ollama is running: ollama serve")
        print("- Models are available: ollama list")
        print("- If on remote server, use port forwarding: ssh -L 11434:localhost:11434 user@server")
        sys.exit(1)

if __name__ == "__main__":
    main()