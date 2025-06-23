#!/usr/bin/env python3

import requests
import psutil
import time
import json

def get_system_resources():
    """Get current system resource usage"""
    memory = psutil.virtual_memory()
    cpu_percent = psutil.cpu_percent(interval=1)
    
    return {
        'memory_total_gb': round(memory.total / (1024**3), 1),
        'memory_used_gb': round(memory.used / (1024**3), 1),
        'memory_available_gb': round(memory.available / (1024**3), 1),
        'memory_percent': memory.percent,
        'cpu_percent': cpu_percent,
        'cpu_count': psutil.cpu_count()
    }

def get_ollama_models():
    """Get list of available Ollama models with sizes"""
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json().get('models', [])
            model_info = []
            for model in models:
                size_gb = round(model['size'] / (1024**3), 1)
                model_info.append({
                    'name': model['name'],
                    'size_gb': size_gb
                })
            return model_info
        return []
    except:
        return []

def get_model_recommendations(available_memory_gb):
    """Get model recommendations based on available memory"""
    recommendations = []
    
    if available_memory_gb >= 12:
        recommendations.append({
            'model': 'qwen3:14b',
            'description': 'High quality, slow inference (uses ~10-14GB RAM)',
            'use_case': 'Best quality responses, can handle complex reasoning'
        })
    
    if available_memory_gb >= 8:
        recommendations.append({
            'model': 'codellama:7b',
            'description': 'Good quality, moderate speed (uses ~6-8GB RAM)',
            'use_case': 'Good balance of quality and speed'
        })
    
    if available_memory_gb >= 4:
        recommendations.append({
            'model': 'llama3.2:3b',
            'description': 'Decent quality, fast inference (uses ~3-4GB RAM)',
            'use_case': 'Good for quick responses with reasonable quality'
        })
    
    recommendations.append({
        'model': 'llama3.2:1b',
        'description': 'Basic quality, very fast (uses ~1-2GB RAM)',
        'use_case': 'Fast responses, basic question answering'
    })
    
    return recommendations

def test_model_performance(model_name, timeout=30):
    """Test a model's response time and resource usage"""
    print(f"Testing {model_name}...")
    
    # Get baseline resources
    baseline = get_system_resources()
    
    start_time = time.time()
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model_name,
                "prompt": "What is Python? Answer briefly.",
                "stream": False
            },
            timeout=timeout
        )
        
        end_time = time.time()
        response_time = end_time - start_time
        
        # Get resources after
        after = get_system_resources()
        
        if response.status_code == 200:
            result = response.json()
            return {
                'model': model_name,
                'success': True,
                'response_time': round(response_time, 1),
                'memory_used_gb': round(after['memory_used_gb'] - baseline['memory_used_gb'], 1),
                'cpu_peak': after['cpu_percent'],
                'response_length': len(result['response'])
            }
        else:
            return {
                'model': model_name,
                'success': False,
                'error': f"HTTP {response.status_code}"
            }
            
    except requests.exceptions.Timeout:
        return {
            'model': model_name,
            'success': False,
            'error': f"Timeout after {timeout}s"
        }
    except Exception as e:
        return {
            'model': model_name,
            'success': False,
            'error': str(e)
        }

def main():
    print("🔍 RAG System Resource Monitor")
    print("=" * 40)
    
    # System resources
    resources = get_system_resources()
    print(f"System Resources:")
    print(f"  Memory: {resources['memory_used_gb']:.1f}GB / {resources['memory_total_gb']:.1f}GB ({resources['memory_percent']:.1f}%)")
    print(f"  Available Memory: {resources['memory_available_gb']:.1f}GB")
    print(f"  CPU: {resources['cpu_percent']:.1f}% ({resources['cpu_count']} cores)")
    
    # Ollama models
    print(f"\nInstalled Models:")
    models = get_ollama_models()
    for model in models:
        print(f"  {model['name']}: {model['size_gb']}GB")
    
    # Recommendations
    print(f"\nModel Recommendations for {resources['memory_available_gb']:.1f}GB available:")
    recommendations = get_model_recommendations(resources['memory_available_gb'])
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec['model']}")
        print(f"     {rec['description']}")
        print(f"     Use case: {rec['use_case']}")
        print()
    
    # Performance test option
    print("Would you like to test model performance? (y/n)")
    if input().lower().startswith('y'):
        print("\nTesting available models...")
        test_models = ['llama3.2:1b']
        
        # Add other models if they fit in memory
        if resources['memory_available_gb'] >= 4:
            test_models.append('codellama:7b')
        
        for model in test_models:
            result = test_model_performance(model, timeout=60)
            if result['success']:
                print(f"✓ {result['model']}: {result['response_time']}s, +{result['memory_used_gb']}GB RAM")
            else:
                print(f"✗ {result['model']}: {result['error']}")

if __name__ == "__main__":
    main()