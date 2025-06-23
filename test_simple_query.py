#!/usr/bin/env python3

import requests
import json
import time

def test_simple_ollama_query():
    """Test simple query directly to Ollama to isolate the issue"""
    
    print("Testing simple query to Ollama...")
    
    # Very simple prompt
    simple_prompt = "Answer in one sentence: What is Python?"
    
    print(f"Prompt: '{simple_prompt}'")
    print("Sending request...")
    
    start_time = time.time()
    
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "qwen3:14b", 
                "prompt": simple_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "max_tokens": 100
                }
            },
            timeout=180
        )
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Success in {elapsed:.1f} seconds")
            print(f"Response: {result['response']}")
            return True
        else:
            print(f"✗ HTTP Error {response.status_code}: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print(f"✗ Timeout after {time.time() - start_time:.1f} seconds")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_streaming_query():
    """Test streaming response"""
    
    print("\nTesting streaming query...")
    
    start_time = time.time()
    
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "qwen3:14b",
                "prompt": "What is Python? Answer briefly.",
                "stream": True
            },
            stream=True,
            timeout=180
        )
        
        if response.status_code == 200:
            print("✓ Streaming response:")
            full_response = ""
            
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        if 'response' in data:
                            chunk = data['response']
                            print(chunk, end='', flush=True)
                            full_response += chunk
                        if data.get('done', False):
                            break
                    except json.JSONDecodeError:
                        continue
            
            end_time = time.time()
            print(f"\n✓ Completed in {end_time - start_time:.1f} seconds")
            return True
        else:
            print(f"✗ HTTP Error {response.status_code}")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Debugging Ollama Query Performance")
    print("=" * 40)
    
    # Test simple query
    success1 = test_simple_ollama_query()
    
    # Test streaming
    success2 = test_streaming_query()
    
    if success1 or success2:
        print("\n✓ At least one method worked - the model is functional")
        print("The timeout issue might be related to prompt complexity or context length")
    else:
        print("\n✗ Both methods failed - there may be a model loading or performance issue")
        print("Consider checking Ollama logs: journalctl -u ollama -f")