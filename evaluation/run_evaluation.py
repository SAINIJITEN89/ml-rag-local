#!/usr/bin/env python3

import os
import sys
import subprocess
from datetime import datetime

def main():
    print("="*60)
    print("MODEL EVALUATION SYSTEM")
    print("="*60)
    print()
    
    print("This evaluation will:")
    print("1. Test all available models on comprehensive scenarios")
    print("2. Measure performance across multiple dimensions")
    print("3. Generate detailed recommendations")
    print("4. Create a structured report")
    print()
    
    # Check if required dependencies are available
    print("Checking system requirements...")
    
    try:
        import pandas as pd
        print("✅ Pandas available")
    except ImportError:
        print("❌ Pandas not available. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pandas"], check=True)
    
    # Check Ollama availability
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama service available")
        else:
            print("❌ Ollama service not responding correctly")
            return
    except Exception as e:
        print(f"❌ Cannot connect to Ollama: {e}")
        print("Please ensure Ollama is running on localhost:11434")
        return
    
    print()
    
    # Run the evaluation
    print("Starting model evaluation...")
    print("This may take several minutes depending on the number of models and tests.")
    print()
    
    # Change to evaluation directory
    evaluation_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(evaluation_dir)
    
    # Run the evaluator
    try:
        result = subprocess.run([sys.executable, "model_evaluator.py"], 
                              capture_output=True, text=True, timeout=3600)
        
        if result.returncode == 0:
            print("✅ Evaluation completed successfully")
            print(result.stdout)
        else:
            print("❌ Evaluation failed")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return
    
    except subprocess.TimeoutExpired:
        print("❌ Evaluation timed out (>1 hour)")
        return
    except Exception as e:
        print(f"❌ Error running evaluation: {e}")
        return
    
    # Find the latest results file
    results_dir = "results"
    if os.path.exists(results_dir):
        performance_files = [f for f in os.listdir(results_dir) if f.startswith("performance_summary_")]
        if performance_files:
            latest_file = max(performance_files, key=lambda x: os.path.getctime(os.path.join(results_dir, x)))
            
            print()
            print("Generating recommendation report...")
            
            # Generate recommendations
            try:
                result = subprocess.run([sys.executable, "recommendation_generator.py", latest_file],
                                      capture_output=True, text=True)
                
                if result.returncode == 0:
                    print("✅ Recommendation report generated")
                    print(result.stdout)
                else:
                    print("❌ Failed to generate recommendation report")
                    print("STDERR:", result.stderr)
            except Exception as e:
                print(f"❌ Error generating recommendations: {e}")
        else:
            print("❌ No performance results found")
    else:
        print("❌ Results directory not found")
    
    print()
    print("="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print()
    print("Check the evaluation/results/ directory for:")
    print("- detailed_results_<timestamp>.json")
    print("- performance_summary_<timestamp>.json")
    print("- results_<timestamp>.csv")
    print("- recommendation_report_<timestamp>.md")


if __name__ == "__main__":
    main()