#!/usr/bin/env python3

import psutil
import requests

class ModelConfig:
    """Configuration manager for selecting appropriate models based on system resources"""
    
    MODEL_CONFIGS = {
        'llama3.2:1b': {
            'memory_gb': 1.5,
            'quality': 'basic',
            'speed': 'very_fast',
            'use_case': 'Quick responses, basic QA'
        },
        'llama3.2:3b': {
            'memory_gb': 3.5,
            'quality': 'good',
            'speed': 'fast',
            'use_case': 'Balanced quality and speed'
        },
        'codellama:7b': {
            'memory_gb': 4.5,
            'quality': 'very_good',
            'speed': 'moderate',
            'use_case': 'Code-focused, good reasoning'
        },
        'llama2:13b': {
            'memory_gb': 8.0,
            'quality': 'excellent',
            'speed': 'slow',
            'use_case': 'High quality responses'
        },
        'qwen3:14b': {
            'memory_gb': 10.0,
            'quality': 'excellent',
            'speed': 'very_slow',
            'use_case': 'Best quality, complex reasoning'
        }
    }
    
    @classmethod
    def get_available_memory_gb(cls):
        """Get available system memory in GB"""
        memory = psutil.virtual_memory()
        return memory.available / (1024**3)
    
    @classmethod
    def get_suitable_models(cls, memory_buffer_gb=2.0):
        """Get models that can fit in available memory with buffer"""
        available_memory = cls.get_available_memory_gb()
        usable_memory = available_memory - memory_buffer_gb
        
        suitable = []
        for model, config in cls.MODEL_CONFIGS.items():
            if config['memory_gb'] <= usable_memory:
                suitable.append({
                    'model': model,
                    'memory_gb': config['memory_gb'],
                    'quality': config['quality'],
                    'speed': config['speed'],
                    'use_case': config['use_case']
                })
        
        # Sort by quality preference
        quality_order = {'excellent': 4, 'very_good': 3, 'good': 2, 'basic': 1}
        suitable.sort(key=lambda x: quality_order.get(x['quality'], 0), reverse=True)
        
        return suitable
    
    @classmethod
    def get_recommended_model(cls, preference='balanced'):
        """Get recommended model based on preference"""
        suitable = cls.get_suitable_models()
        
        if not suitable:
            return 'llama3.2:1b'  # Fallback to smallest model
        
        if preference == 'speed':
            # Return fastest suitable model
            speed_order = {'very_fast': 4, 'fast': 3, 'moderate': 2, 'slow': 1, 'very_slow': 0}
            suitable.sort(key=lambda x: speed_order.get(x['speed'], 0), reverse=True)
        elif preference == 'quality':
            # Already sorted by quality
            pass
        else:  # balanced
            # Balance between quality and speed
            for model in suitable:
                if model['quality'] in ['good', 'very_good'] and model['speed'] in ['fast', 'moderate']:
                    return model['model']
        
        return suitable[0]['model'] if suitable else 'llama3.2:1b'
    
    @classmethod
    def is_model_available(cls, model_name):
        """Check if a model is available in Ollama"""
        try:
            response = requests.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                return any(model['name'] == model_name for model in models)
            return False
        except:
            return False
    
    @classmethod
    def print_recommendations(cls):
        """Print model recommendations"""
        available_gb = cls.get_available_memory_gb()
        suitable = cls.get_suitable_models()
        
        print(f"Available Memory: {available_gb:.1f}GB")
        print("\nSuitable Models:")
        
        if not suitable:
            print("  ⚠️ Very low memory - only basic model available")
            print("  Recommendation: llama3.2:1b")
        else:
            for model in suitable:
                available = "✓" if cls.is_model_available(model['model']) else "✗"
                print(f"  {available} {model['model']} ({model['memory_gb']}GB)")
                print(f"     Quality: {model['quality']}, Speed: {model['speed']}")
                print(f"     Use case: {model['use_case']}")
        
        print(f"\nRecommendations:")
        print(f"  Speed-focused: {cls.get_recommended_model('speed')}")
        print(f"  Quality-focused: {cls.get_recommended_model('quality')}")
        print(f"  Balanced: {cls.get_recommended_model('balanced')}")

if __name__ == "__main__":
    ModelConfig.print_recommendations()