#!/usr/bin/env python3

import os
import sys
import json
import time
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import pandas as pd
import psutil
import threading
from contextlib import contextmanager

# Add parent directory to path to import from main project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_system import RAGSystem
from model_config import ModelConfig
from resource_monitor import get_system_resources


@dataclass
class ResourceUsage:
    """Resource usage metrics"""
    memory_before_gb: float
    memory_after_gb: float
    memory_delta_gb: float
    memory_peak_gb: float
    cpu_avg_percent: float
    cpu_peak_percent: float
    

@dataclass
class TestResult:
    """Single test result"""
    test_name: str
    model: str
    query: str
    expected_info: List[str]
    response: str
    response_time: float
    success: bool
    relevance_score: float
    accuracy_score: float
    completeness_score: float
    reasoning_quality: float
    resource_usage: Optional[ResourceUsage] = None
    error: Optional[str] = None


@dataclass
class ModelPerformance:
    """Overall model performance metrics"""
    model: str
    total_tests: int
    successful_tests: int
    avg_response_time: float
    avg_relevance_score: float
    avg_accuracy_score: float
    avg_completeness_score: float
    avg_reasoning_quality: float
    model_size_gb: float
    avg_memory_usage_gb: float
    peak_memory_usage_gb: float
    avg_cpu_usage_percent: float
    peak_cpu_usage_percent: float
    can_run: bool
    failure_rate: float


class ResourceMonitor:
    """Monitor system resources during evaluation"""
    
    def __init__(self):
        self.monitoring = False
        self.memory_samples = []
        self.cpu_samples = []
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start resource monitoring"""
        self.monitoring = True
        self.memory_samples = []
        self.cpu_samples = []
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> ResourceUsage:
        """Stop monitoring and return usage stats"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        
        if not self.memory_samples or not self.cpu_samples:
            # Fallback to current readings
            current = get_system_resources()
            return ResourceUsage(
                memory_before_gb=current['memory_used_gb'],
                memory_after_gb=current['memory_used_gb'],
                memory_delta_gb=0.0,
                memory_peak_gb=current['memory_used_gb'],
                cpu_avg_percent=current['cpu_percent'],
                cpu_peak_percent=current['cpu_percent']
            )
        
        return ResourceUsage(
            memory_before_gb=self.memory_samples[0],
            memory_after_gb=self.memory_samples[-1],
            memory_delta_gb=max(self.memory_samples) - self.memory_samples[0],
            memory_peak_gb=max(self.memory_samples),
            cpu_avg_percent=sum(self.cpu_samples) / len(self.cpu_samples),
            cpu_peak_percent=max(self.cpu_samples)
        )
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            try:
                resources = get_system_resources()
                self.memory_samples.append(resources['memory_used_gb'])
                self.cpu_samples.append(resources['cpu_percent'])
                time.sleep(0.5)  # Sample every 500ms
            except Exception:
                pass


class ProgressTracker:
    """Track evaluation progress and provide user feedback"""
    
    def __init__(self, total_models: int, total_tests_per_model: int):
        self.total_models = total_models
        self.total_tests_per_model = total_tests_per_model
        self.total_tests = total_models * total_tests_per_model
        self.completed_models = 0
        self.completed_tests = 0
        self.start_time = time.time()
        self.current_model = None
    
    def start_model(self, model_name: str):
        """Start evaluation of a model"""
        self.current_model = model_name
        elapsed = time.time() - self.start_time
        progress = (self.completed_tests / self.total_tests) * 100 if self.total_tests > 0 else 0
        
        print(f"\\n📊 [{self.completed_models + 1}/{self.total_models}] Evaluating {model_name}")
        print(f"   Progress: {progress:.1f}% | Elapsed: {elapsed/60:.1f}min | Tests completed: {self.completed_tests}/{self.total_tests}")
    
    def complete_test(self, test_name: str, success: bool, response_time: float):
        """Mark a test as completed"""
        self.completed_tests += 1
        status = "✅" if success else "❌"
        print(f"   {status} {test_name}: {response_time:.2f}s")
    
    def complete_model(self, model_name: str, successful_tests: int, total_tests: int):
        """Mark a model evaluation as completed"""
        self.completed_models += 1
        success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
        print(f"   📈 {model_name} completed: {successful_tests}/{total_tests} tests passed ({success_rate:.1f}%)")
    
    def finish(self):
        """Complete the evaluation"""
        elapsed = time.time() - self.start_time
        print(f"\\n🎉 Evaluation completed in {elapsed/60:.1f} minutes")


class ModelEvaluator:
    """Comprehensive model evaluation framework"""
    
    def __init__(self, output_dir: str = "evaluation/results"):
        self.output_dir = output_dir
        self.results: List[TestResult] = []
        self.models_to_test = list(ModelConfig.MODEL_CONFIGS.keys())
        self.resource_monitor = ResourceMonitor()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize test scenarios
        self.test_scenarios = self._load_test_scenarios()
        
        # Calculate total tests for progress tracking
        self.total_test_scenarios = sum(len(tests) for tests in self.test_scenarios.values())
    
    def _load_test_scenarios(self) -> Dict[str, List[Dict]]:
        """Load all test scenarios"""
        return {
            'format_handling': self._get_format_handling_tests(),
            'scattered_info': self._get_scattered_info_tests(),
            'large_documents': self._get_large_document_tests(),
            'conflicting_info': self._get_conflicting_info_tests(),
            'reasoning_depth': self._get_reasoning_depth_tests(),
            'edge_cases': self._get_edge_case_tests()
        }
    
    def _get_format_handling_tests(self) -> List[Dict]:
        """Tests for handling multiple document formats"""
        return [
            {
                'name': 'PDF Processing',
                'documents': ['test_data/documents/pdf/technical_manual.pdf'],
                'query': 'What are the key installation steps mentioned in the document?',
                'expected_info': ['installation', 'steps', 'requirements'],
                'difficulty': 'medium'
            },
            {
                'name': 'DOCX Processing',
                'documents': ['test_data/documents/docx/project_spec.docx'],
                'query': 'What are the project deliverables and timeline?',
                'expected_info': ['deliverables', 'timeline', 'milestones'],
                'difficulty': 'medium'
            },
            {
                'name': 'JSON Data Processing',
                'documents': ['test_data/documents/json/api_config.json'],
                'query': 'What are the available API endpoints and their parameters?',
                'expected_info': ['endpoints', 'parameters', 'authentication'],
                'difficulty': 'easy'
            },
            {
                'name': 'XML Processing',
                'documents': ['test_data/documents/xml/system_config.xml'],
                'query': 'What are the system configuration parameters?',
                'expected_info': ['configuration', 'parameters', 'settings'],
                'difficulty': 'medium'
            }
        ]
    
    def _get_scattered_info_tests(self) -> List[Dict]:
        """Tests for gathering information scattered across multiple documents"""
        return [
            {
                'name': 'Cross-Document Synthesis',
                'documents': [
                    'test_data/documents/txt/requirements.txt',
                    'test_data/documents/txt/architecture.txt',
                    'test_data/documents/txt/deployment.txt'
                ],
                'query': 'Based on all documents, what is the complete system architecture and deployment strategy?',
                'expected_info': ['architecture', 'components', 'deployment', 'requirements'],
                'difficulty': 'hard'
            },
            {
                'name': 'Multi-Source Timeline',
                'documents': [
                    'test_data/documents/txt/project_phase1.txt',
                    'test_data/documents/txt/project_phase2.txt',
                    'test_data/documents/txt/project_phase3.txt'
                ],
                'query': 'Create a comprehensive timeline combining all project phases',
                'expected_info': ['timeline', 'phases', 'dependencies', 'deliverables'],
                'difficulty': 'hard'
            }
        ]
    
    def _get_large_document_tests(self) -> List[Dict]:
        """Tests for handling large documents"""
        return [
            {
                'name': 'Large Technical Document',
                'documents': ['test_data/documents/txt/large_technical_spec.txt'],
                'query': 'What are the main technical specifications and their relationships?',
                'expected_info': ['specifications', 'requirements', 'relationships'],
                'difficulty': 'hard'
            },
            {
                'name': 'Comprehensive Manual',
                'documents': ['test_data/documents/txt/user_manual_complete.txt'],
                'query': 'What are all the troubleshooting steps mentioned for common issues?',
                'expected_info': ['troubleshooting', 'steps', 'solutions'],
                'difficulty': 'medium'
            }
        ]
    
    def _get_conflicting_info_tests(self) -> List[Dict]:
        """Tests for reasoning with conflicting information"""
        return [
            {
                'name': 'Version Conflicts',
                'documents': [
                    'test_data/documents/txt/version_1_spec.txt',
                    'test_data/documents/txt/version_2_spec.txt',
                    'test_data/documents/txt/latest_updates.txt'
                ],
                'query': 'What are the current valid specifications, resolving any conflicts between versions?',
                'expected_info': ['current', 'specifications', 'conflicts', 'resolution'],
                'difficulty': 'very_hard'
            },
            {
                'name': 'Policy Contradictions',
                'documents': [
                    'test_data/documents/txt/old_policy.txt',
                    'test_data/documents/txt/new_policy.txt',
                    'test_data/documents/txt/exceptions.txt'
                ],
                'query': 'What is the current effective policy considering all documents?',
                'expected_info': ['policy', 'current', 'effective', 'exceptions'],
                'difficulty': 'very_hard'
            }
        ]
    
    def _get_reasoning_depth_tests(self) -> List[Dict]:
        """Tests for deep reasoning capabilities"""
        return [
            {
                'name': 'Causal Analysis',
                'documents': ['test_data/documents/txt/system_logs.txt'],
                'query': 'Analyze the root causes of system failures and propose preventive measures',
                'expected_info': ['root_causes', 'analysis', 'prevention', 'measures'],
                'difficulty': 'very_hard'
            },
            {
                'name': 'Strategic Planning',
                'documents': [
                    'test_data/documents/txt/market_analysis.txt',
                    'test_data/documents/txt/company_resources.txt'
                ],
                'query': 'Based on market analysis and available resources, what strategic recommendations would you make?',
                'expected_info': ['strategy', 'recommendations', 'analysis', 'resources'],
                'difficulty': 'very_hard'
            }
        ]
    
    def _get_edge_case_tests(self) -> List[Dict]:
        """Tests for edge cases and unusual scenarios"""
        return [
            {
                'name': 'Empty Document Handling',
                'documents': ['test_data/documents/txt/empty.txt'],
                'query': 'What information can you extract from this document?',
                'expected_info': ['empty', 'no_content'],
                'difficulty': 'easy'
            },
            {
                'name': 'Corrupted Data Handling',
                'documents': ['test_data/documents/txt/partially_corrupted.txt'],
                'query': 'Extract any useful information despite data corruption',
                'expected_info': ['partial', 'extraction', 'recovery'],
                'difficulty': 'hard'
            }
        ]
    
    def _can_model_run(self, model_name: str) -> Tuple[bool, str]:
        """Check if model can run on current system"""
        try:
            config = ModelConfig.MODEL_CONFIGS.get(model_name)
            if not config:
                return False, "Model not in configuration"
            
            available_memory = ModelConfig.get_available_memory_gb()
            required_memory = config['memory_gb']
            
            if available_memory < required_memory + 1.0:  # 1GB buffer
                return False, f"Insufficient memory: {available_memory:.1f}GB available, {required_memory:.1f}GB required"
            
            # Try to initialize RAG system with this model
            rag = RAGSystem(llm_model=model_name)
            test_response = rag.query("Test query")
            
            return True, "Model can run successfully"
            
        except Exception as e:
            return False, f"Model initialization failed: {str(e)}"
    
    def _score_response(self, response: str, expected_info: List[str], query: str) -> Dict[str, float]:
        """Score response quality (simplified scoring - could be enhanced with LLM-based evaluation)"""
        scores = {
            'relevance_score': 0.0,
            'accuracy_score': 0.0,
            'completeness_score': 0.0,
            'reasoning_quality': 0.0
        }
        
        if not response or len(response.strip()) < 10:
            return scores
        
        response_lower = response.lower()
        
        # Relevance: Check if response addresses the query
        query_words = set(query.lower().split())
        response_words = set(response_lower.split())
        relevance = len(query_words.intersection(response_words)) / len(query_words)
        scores['relevance_score'] = min(relevance * 1.5, 1.0)  # Scale up to 1.0
        
        # Completeness: Check if expected information is covered
        covered_info = sum(1 for info in expected_info if info.lower() in response_lower)
        scores['completeness_score'] = covered_info / len(expected_info) if expected_info else 0.0
        
        # Accuracy: Basic heuristics (length, structure, etc.)
        if len(response) > 50:
            scores['accuracy_score'] += 0.3
        if '.' in response and response.count('.') > 1:
            scores['accuracy_score'] += 0.3
        if any(word in response_lower for word in ['because', 'therefore', 'however', 'analysis']):
            scores['accuracy_score'] += 0.4
        
        # Reasoning quality: Check for reasoning indicators
        reasoning_indicators = ['analysis', 'because', 'therefore', 'consequently', 'evidence', 'suggests']
        reasoning_count = sum(1 for indicator in reasoning_indicators if indicator in response_lower)
        scores['reasoning_quality'] = min(reasoning_count / 3.0, 1.0)
        
        return scores
    
    def run_single_test(self, model_name: str, test_scenario: Dict, progress_tracker: ProgressTracker = None) -> TestResult:
        """Run a single test scenario"""
        start_time = time.time()
        
        # Start resource monitoring
        self.resource_monitor.start_monitoring()
        
        try:
            # Initialize RAG system
            rag = RAGSystem(llm_model=model_name)
            
            # Add documents to the system
            for doc_path in test_scenario['documents']:
                if os.path.exists(doc_path):
                    rag.add_document(doc_path)
            
            # Run query
            result = rag.query(test_scenario['query'])
            response = result['answer']
            response_time = time.time() - start_time
            
            # Stop monitoring and get resource usage
            resource_usage = self.resource_monitor.stop_monitoring()
            
            # Score response
            scores = self._score_response(response, test_scenario['expected_info'], test_scenario['query'])
            
            # Update progress tracker
            if progress_tracker:
                progress_tracker.complete_test(test_scenario['name'], True, response_time)
            
            return TestResult(
                test_name=test_scenario['name'],
                model=model_name,
                query=test_scenario['query'],
                expected_info=test_scenario['expected_info'],
                response=response,
                response_time=response_time,
                success=True,
                resource_usage=resource_usage,
                **scores
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            # Stop monitoring in case of error
            try:
                resource_usage = self.resource_monitor.stop_monitoring()
            except:
                resource_usage = None
            
            # Update progress tracker
            if progress_tracker:
                progress_tracker.complete_test(test_scenario['name'], False, response_time)
            
            return TestResult(
                test_name=test_scenario['name'],
                model=model_name,
                query=test_scenario['query'],
                expected_info=test_scenario['expected_info'],
                response="",
                response_time=response_time,
                success=False,
                relevance_score=0.0,
                accuracy_score=0.0,
                completeness_score=0.0,
                reasoning_quality=0.0,
                resource_usage=resource_usage,
                error=str(e)
            )
    
    def evaluate_model(self, model_name: str, progress_tracker: ProgressTracker = None) -> List[TestResult]:
        """Evaluate a single model on all test scenarios"""
        print(f"\nEvaluating model: {model_name}")
        
        # Check if model can run
        can_run, reason = self._can_model_run(model_name)
        if not can_run:
            print(f"  ❌ Model cannot run: {reason}")
            return []
        
        print(f"  ✅ Model can run")
        
        model_results = []
        
        # Run all test scenarios
        for category, tests in self.test_scenarios.items():
            print(f"  Category: {category}")
            for test in tests:
                result = self.run_single_test(model_name, test)
                model_results.append(result)
                status = "✅" if result.success else "❌"
                print(f"    {status} {test['name']}: {result.response_time:.2f}s")
        
        return model_results
    
    def run_full_evaluation(self) -> Dict[str, ModelPerformance]:
        """Run full evaluation on all models"""
        print("🚀 Starting comprehensive model evaluation...")
        print(f"📊 Will evaluate {len(self.models_to_test)} models with {self.total_test_scenarios} test scenarios each")
        
        # Initialize progress tracking
        progress_tracker = ProgressTracker(len(self.models_to_test), self.total_test_scenarios)
        
        all_results = []
        model_performances = {}
        
        for model in self.models_to_test:
            model_results = self.evaluate_model(model, progress_tracker)
            all_results.extend(model_results)
            
            if model_results:
                # Calculate performance metrics
                successful_tests = [r for r in model_results if r.success]
                total_tests = len(model_results)
                
                # Calculate resource usage metrics
                resource_data = [r.resource_usage for r in successful_tests if r.resource_usage is not None]
                avg_memory_usage = sum(r.memory_delta_gb for r in resource_data) / len(resource_data) if resource_data else 0
                peak_memory_usage = max((r.memory_peak_gb for r in resource_data), default=0)
                avg_cpu_usage = sum(r.cpu_avg_percent for r in resource_data) / len(resource_data) if resource_data else 0
                peak_cpu_usage = max((r.cpu_peak_percent for r in resource_data), default=0)
                
                performance = ModelPerformance(
                    model=model,
                    total_tests=total_tests,
                    successful_tests=len(successful_tests),
                    avg_response_time=sum(r.response_time for r in successful_tests) / len(successful_tests) if successful_tests else 0,
                    avg_relevance_score=sum(r.relevance_score for r in successful_tests) / len(successful_tests) if successful_tests else 0,
                    avg_accuracy_score=sum(r.accuracy_score for r in successful_tests) / len(successful_tests) if successful_tests else 0,
                    avg_completeness_score=sum(r.completeness_score for r in successful_tests) / len(successful_tests) if successful_tests else 0,
                    avg_reasoning_quality=sum(r.reasoning_quality for r in successful_tests) / len(successful_tests) if successful_tests else 0,
                    model_size_gb=ModelConfig.MODEL_CONFIGS[model]['memory_gb'],
                    avg_memory_usage_gb=avg_memory_usage,
                    peak_memory_usage_gb=peak_memory_usage,
                    avg_cpu_usage_percent=avg_cpu_usage,
                    peak_cpu_usage_percent=peak_cpu_usage,
                    can_run=len(successful_tests) > 0,
                    failure_rate=(total_tests - len(successful_tests)) / total_tests if total_tests > 0 else 1.0
                )
            else:
                performance = ModelPerformance(
                    model=model,
                    total_tests=0,
                    successful_tests=0,
                    avg_response_time=0,
                    avg_relevance_score=0,
                    avg_accuracy_score=0,
                    avg_completeness_score=0,
                    avg_reasoning_quality=0,
                    model_size_gb=ModelConfig.MODEL_CONFIGS[model]['memory_gb'],
                    avg_memory_usage_gb=0,
                    peak_memory_usage_gb=0,
                    avg_cpu_usage_percent=0,
                    peak_cpu_usage_percent=0,
                    can_run=False,
                    failure_rate=1.0
                )
            
            model_performances[model] = performance
        
        # Complete progress tracking
        progress_tracker.finish()
        
        # Save results
        self._save_results(all_results, model_performances)
        
        return model_performances
    
    def _save_results(self, results: List[TestResult], performances: Dict[str, ModelPerformance]):
        """Save evaluation results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_data = [asdict(r) for r in results]
        with open(f"{self.output_dir}/detailed_results_{timestamp}.json", 'w') as f:
            json.dump(results_data, f, indent=2)
        
        # Save performance summary
        performance_data = {k: asdict(v) for k, v in performances.items()}
        with open(f"{self.output_dir}/performance_summary_{timestamp}.json", 'w') as f:
            json.dump(performance_data, f, indent=2)
        
        # Save CSV for easy analysis
        if results:
            df = pd.DataFrame(results_data)
            df.to_csv(f"{self.output_dir}/results_{timestamp}.csv", index=False)
        
        print(f"\nResults saved to {self.output_dir}/")


if __name__ == "__main__":
    evaluator = ModelEvaluator()
    performances = evaluator.run_full_evaluation()
    
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    
    for model, perf in performances.items():
        print(f"\nModel: {model}")
        print(f"  Can Run: {'✅' if perf.can_run else '❌'}")
        if perf.can_run:
            print(f"  Success Rate: {(1-perf.failure_rate)*100:.1f}%")
            print(f"  Avg Response Time: {perf.avg_response_time:.2f}s")
            print(f"  Avg Relevance: {perf.avg_relevance_score:.2f}")
            print(f"  Avg Accuracy: {perf.avg_accuracy_score:.2f}")
            print(f"  Avg Completeness: {perf.avg_completeness_score:.2f}")
            print(f"  Avg Reasoning: {perf.avg_reasoning_quality:.2f}")