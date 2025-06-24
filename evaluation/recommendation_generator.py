#!/usr/bin/env python3

import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import pandas as pd

@dataclass
class ModelRecommendation:
    model: str
    rank: int
    overall_score: float
    strengths: List[str]
    weaknesses: List[str]
    use_cases: List[str]
    performance_summary: Dict[str, float]
    resource_requirements: Dict[str, Any]
    recommendation_confidence: float


class RecommendationGenerator:
    """Generate structured recommendations based on evaluation results"""
    
    def __init__(self, results_file: str):
        self.results_file = results_file
        self.performance_data = self._load_performance_data()
        self.detailed_results = self._load_detailed_results()
    
    def _load_performance_data(self) -> Dict[str, Any]:
        """Load performance summary data"""
        try:
            base_name = self.results_file.replace('performance_summary_', '').replace('.json', '')
            performance_file = f"evaluation/results/performance_summary_{base_name}.json"
            
            with open(performance_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: Performance file not found for {self.results_file}")
            return {}
    
    def _load_detailed_results(self) -> List[Dict]:
        """Load detailed test results"""
        try:
            base_name = self.results_file.replace('performance_summary_', '').replace('.json', '')
            detailed_file = f"evaluation/results/detailed_results_{base_name}.json"
            
            with open(detailed_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: Detailed results file not found for {self.results_file}")
            return []
    
    def _calculate_overall_score(self, model_data: Dict) -> float:
        """Calculate overall score for a model"""
        if not model_data['can_run']:
            return 0.0
        
        weights = {
            'success_rate': 0.25,
            'avg_relevance_score': 0.20,
            'avg_accuracy_score': 0.20,
            'avg_completeness_score': 0.15,
            'avg_reasoning_quality': 0.15,
            'speed_score': 0.05  # Derived from response time
        }
        
        success_rate = (model_data['successful_tests'] / model_data['total_tests']) if model_data['total_tests'] > 0 else 0
        
        # Speed score: inverse of response time (normalized)
        speed_score = 1.0 / (1.0 + model_data['avg_response_time']) if model_data['avg_response_time'] > 0 else 0
        
        overall_score = (
            weights['success_rate'] * success_rate +
            weights['avg_relevance_score'] * model_data['avg_relevance_score'] +
            weights['avg_accuracy_score'] * model_data['avg_accuracy_score'] +
            weights['avg_completeness_score'] * model_data['avg_completeness_score'] +
            weights['avg_reasoning_quality'] * model_data['avg_reasoning_quality'] +
            weights['speed_score'] * speed_score
        )
        
        return overall_score
    
    def _analyze_strengths_weaknesses(self, model: str, model_data: Dict) -> tuple:
        """Analyze model strengths and weaknesses"""
        strengths = []
        weaknesses = []
        
        # Analyze based on performance metrics
        if model_data['avg_response_time'] < 2.0:
            strengths.append("Fast response times")
        elif model_data['avg_response_time'] > 10.0:
            weaknesses.append("Slow response times")
        
        if model_data['avg_relevance_score'] > 0.8:
            strengths.append("High relevance in responses")
        elif model_data['avg_relevance_score'] < 0.5:
            weaknesses.append("Low relevance in responses")
        
        if model_data['avg_accuracy_score'] > 0.8:
            strengths.append("High accuracy")
        elif model_data['avg_accuracy_score'] < 0.5:
            weaknesses.append("Low accuracy")
        
        if model_data['avg_completeness_score'] > 0.8:
            strengths.append("Comprehensive responses")
        elif model_data['avg_completeness_score'] < 0.5:
            weaknesses.append("Incomplete responses")
        
        if model_data['avg_reasoning_quality'] > 0.8:
            strengths.append("Strong reasoning capabilities")
        elif model_data['avg_reasoning_quality'] < 0.5:
            weaknesses.append("Weak reasoning capabilities")
        
        if model_data['failure_rate'] < 0.1:
            strengths.append("High reliability")
        elif model_data['failure_rate'] > 0.3:
            weaknesses.append("High failure rate")
        
        if model_data['memory_usage_gb'] < 3.0:
            strengths.append("Low memory requirements")
        elif model_data['memory_usage_gb'] > 8.0:
            weaknesses.append("High memory requirements")
        
        # Analyze based on detailed test results
        model_results = [r for r in self.detailed_results if r['model'] == model]
        
        # Check performance on different test categories
        category_performance = {}
        for result in model_results:
            test_name = result['test_name']
            category = self._categorize_test(test_name)
            if category not in category_performance:
                category_performance[category] = []
            category_performance[category].append(result)
        
        for category, results in category_performance.items():
            success_rate = sum(1 for r in results if r['success']) / len(results)
            avg_quality = sum(r['reasoning_quality'] for r in results if r['success']) / len([r for r in results if r['success']]) if any(r['success'] for r in results) else 0
            
            if success_rate > 0.8 and avg_quality > 0.7:
                strengths.append(f"Excellent {category} handling")
            elif success_rate < 0.5 or avg_quality < 0.3:
                weaknesses.append(f"Poor {category} handling")
        
        return strengths, weaknesses
    
    def _categorize_test(self, test_name: str) -> str:
        """Categorize test based on test name"""
        test_name_lower = test_name.lower()
        
        if any(word in test_name_lower for word in ['pdf', 'docx', 'json', 'xml', 'format']):
            return "format handling"
        elif any(word in test_name_lower for word in ['cross', 'multi', 'scattered', 'synthesis']):
            return "information synthesis"
        elif any(word in test_name_lower for word in ['large', 'comprehensive', 'manual']):
            return "large document processing"
        elif any(word in test_name_lower for word in ['conflict', 'version', 'contradiction']):
            return "conflict resolution"
        elif any(word in test_name_lower for word in ['analysis', 'reasoning', 'causal', 'strategic']):
            return "reasoning and analysis"
        elif any(word in test_name_lower for word in ['empty', 'corrupted', 'edge']):
            return "edge case handling"
        else:
            return "general processing"
    
    def _determine_use_cases(self, model: str, model_data: Dict, strengths: List[str]) -> List[str]:
        """Determine best use cases for the model"""
        use_cases = []
        
        # Based on memory requirements
        if model_data['memory_usage_gb'] < 3.0:
            use_cases.append("Resource-constrained environments")
            use_cases.append("Development and testing")
        
        # Based on speed
        if model_data['avg_response_time'] < 2.0:
            use_cases.append("Real-time applications")
            use_cases.append("Interactive user interfaces")
        
        # Based on accuracy and reasoning
        if model_data['avg_accuracy_score'] > 0.8 and model_data['avg_reasoning_quality'] > 0.7:
            use_cases.append("Complex analytical tasks")
            use_cases.append("Research and analysis")
        
        # Based on completeness
        if model_data['avg_completeness_score'] > 0.8:
            use_cases.append("Comprehensive document processing")
            use_cases.append("Information extraction tasks")
        
        # Based on reliability
        if model_data['failure_rate'] < 0.1:
            use_cases.append("Production environments")
            use_cases.append("Mission-critical applications")
        
        # Model-specific recommendations
        if 'phi' in model.lower():
            use_cases.append("Instruction following tasks")
            use_cases.append("Reasoning-intensive applications")
        
        if 'codellama' in model.lower():
            use_cases.append("Code analysis and generation")
            use_cases.append("Technical documentation processing")
        
        if '1b' in model.lower():
            use_cases.append("Edge computing")
            use_cases.append("Batch processing with limited resources")
        
        return list(set(use_cases))  # Remove duplicates
    
    def _calculate_confidence(self, model_data: Dict) -> float:
        """Calculate recommendation confidence based on test coverage and results"""
        if not model_data['can_run']:
            return 0.0
        
        # Base confidence on number of successful tests
        test_coverage = model_data['successful_tests'] / model_data['total_tests'] if model_data['total_tests'] > 0 else 0
        
        # Adjust based on performance consistency
        metrics = [
            model_data['avg_relevance_score'],
            model_data['avg_accuracy_score'],
            model_data['avg_completeness_score'],
            model_data['avg_reasoning_quality']
        ]
        
        # Calculate coefficient of variation (lower is better for consistency)
        if len(metrics) > 1:
            mean_metric = sum(metrics) / len(metrics)
            variance = sum((x - mean_metric) ** 2 for x in metrics) / len(metrics)
            std_dev = variance ** 0.5
            cv = std_dev / mean_metric if mean_metric > 0 else 1.0
            consistency_score = 1.0 / (1.0 + cv)
        else:
            consistency_score = 0.5
        
        # Combine factors
        confidence = (test_coverage * 0.6 + consistency_score * 0.4)
        
        return min(confidence, 1.0)
    
    def generate_recommendations(self) -> List[ModelRecommendation]:
        """Generate recommendations for all models"""
        recommendations = []
        
        # Calculate scores and rank models
        model_scores = []
        for model, data in self.performance_data.items():
            score = self._calculate_overall_score(data)
            model_scores.append((model, score, data))
        
        # Sort by score (descending)
        model_scores.sort(key=lambda x: x[1], reverse=True)
        
        for rank, (model, score, data) in enumerate(model_scores, 1):
            strengths, weaknesses = self._analyze_strengths_weaknesses(model, data)
            use_cases = self._determine_use_cases(model, data, strengths)
            confidence = self._calculate_confidence(data)
            
            recommendation = ModelRecommendation(
                model=model,
                rank=rank,
                overall_score=score,
                strengths=strengths,
                weaknesses=weaknesses,
                use_cases=use_cases,
                performance_summary={
                    'success_rate': (data['successful_tests'] / data['total_tests']) if data['total_tests'] > 0 else 0,
                    'avg_response_time': data['avg_response_time'],
                    'avg_relevance_score': data['avg_relevance_score'],
                    'avg_accuracy_score': data['avg_accuracy_score'],
                    'avg_completeness_score': data['avg_completeness_score'],
                    'avg_reasoning_quality': data['avg_reasoning_quality'],
                    'failure_rate': data['failure_rate']
                },
                resource_requirements={
                    'memory_gb': data['memory_usage_gb'],
                    'can_run': data['can_run']
                },
                recommendation_confidence=confidence
            )
            
            recommendations.append(recommendation)
        
        return recommendations
    
    def generate_report(self, output_file: str) -> str:
        """Generate a comprehensive recommendation report"""
        recommendations = self.generate_recommendations()
        
        report = []
        report.append("# Model Evaluation and Recommendation Report")
        report.append("")
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Executive Summary
        report.append("## Executive Summary")
        report.append("")
        
        runnable_models = [r for r in recommendations if r.resource_requirements['can_run']]
        if runnable_models:
            best_model = runnable_models[0]
            report.append(f"**Recommended Model**: {best_model.model}")
            report.append(f"**Overall Score**: {best_model.overall_score:.3f}")
            report.append(f"**Confidence Level**: {best_model.recommendation_confidence:.1%}")
            report.append("")
            report.append("**Key Strengths**:")
            for strength in best_model.strengths[:3]:
                report.append(f"- {strength}")
            report.append("")
        else:
            report.append("⚠️ **No models can run on the current system**")
            report.append("")
        
        # System Requirements Analysis
        report.append("## System Compatibility Analysis")
        report.append("")
        report.append("| Model | Can Run | Memory Required | Status |")
        report.append("|-------|---------|----------------|---------|")
        
        for rec in recommendations:
            status = "✅ Compatible" if rec.resource_requirements['can_run'] else "❌ Insufficient Resources"
            report.append(f"| {rec.model} | {'Yes' if rec.resource_requirements['can_run'] else 'No'} | {rec.resource_requirements['memory_gb']:.1f}GB | {status} |")
        
        report.append("")
        
        # Detailed Model Rankings
        report.append("## Detailed Model Rankings")
        report.append("")
        
        for rec in recommendations:
            report.append(f"### {rec.rank}. {rec.model}")
            report.append("")
            report.append(f"**Overall Score**: {rec.overall_score:.3f}")
            report.append(f"**Confidence**: {rec.recommendation_confidence:.1%}")
            report.append("")
            
            if rec.resource_requirements['can_run']:
                # Performance metrics
                report.append("**Performance Metrics**:")
                report.append(f"- Success Rate: {rec.performance_summary['success_rate']:.1%}")
                report.append(f"- Average Response Time: {rec.performance_summary['avg_response_time']:.2f}s")
                report.append(f"- Relevance Score: {rec.performance_summary['avg_relevance_score']:.3f}")
                report.append(f"- Accuracy Score: {rec.performance_summary['avg_accuracy_score']:.3f}")
                report.append(f"- Completeness Score: {rec.performance_summary['avg_completeness_score']:.3f}")
                report.append(f"- Reasoning Quality: {rec.performance_summary['avg_reasoning_quality']:.3f}")
                report.append("")
                
                # Strengths
                if rec.strengths:
                    report.append("**Strengths**:")
                    for strength in rec.strengths:
                        report.append(f"- {strength}")
                    report.append("")
                
                # Weaknesses
                if rec.weaknesses:
                    report.append("**Weaknesses**:")
                    for weakness in rec.weaknesses:
                        report.append(f"- {weakness}")
                    report.append("")
                
                # Use cases
                if rec.use_cases:
                    report.append("**Recommended Use Cases**:")
                    for use_case in rec.use_cases:
                        report.append(f"- {use_case}")
                    report.append("")
            else:
                report.append("❌ **Cannot run on current system** - Insufficient memory")
                report.append("")
            
            report.append("---")
            report.append("")
        
        # Recommendations by Use Case
        report.append("## Recommendations by Use Case")
        report.append("")
        
        use_case_models = {}
        for rec in recommendations:
            if rec.resource_requirements['can_run']:
                for use_case in rec.use_cases:
                    if use_case not in use_case_models:
                        use_case_models[use_case] = []
                    use_case_models[use_case].append((rec.model, rec.overall_score))
        
        for use_case, models in use_case_models.items():
            report.append(f"### {use_case}")
            models.sort(key=lambda x: x[1], reverse=True)
            for i, (model, score) in enumerate(models[:3], 1):
                report.append(f"{i}. **{model}** (Score: {score:.3f})")
            report.append("")
        
        # Implementation Recommendations
        report.append("## Implementation Recommendations")
        report.append("")
        
        if runnable_models:
            best_model = runnable_models[0]
            report.append(f"### Primary Recommendation: {best_model.model}")
            report.append("")
            report.append("**Implementation Strategy**:")
            
            if best_model.overall_score > 0.8:
                report.append("- Deploy immediately for production use")
            elif best_model.overall_score > 0.6:
                report.append("- Suitable for production with monitoring")
                report.append("- Consider gradual rollout")
            else:
                report.append("- Use for development and testing first")
                report.append("- Monitor performance closely")
            
            if best_model.performance_summary['avg_response_time'] > 5.0:
                report.append("- Consider response time optimization")
            
            if best_model.performance_summary['failure_rate'] > 0.2:
                report.append("- Implement robust error handling")
            
            report.append("")
            
            # Fallback options
            if len(runnable_models) > 1:
                report.append("**Fallback Options**:")
                for backup in runnable_models[1:3]:
                    report.append(f"- {backup.model} (Score: {backup.overall_score:.3f})")
                report.append("")
        
        # System Upgrade Recommendations
        report.append("## System Upgrade Recommendations")
        report.append("")
        
        non_runnable = [r for r in recommendations if not r.resource_requirements['can_run']]
        if non_runnable:
            min_memory_needed = min(r.resource_requirements['memory_gb'] for r in non_runnable)
            report.append(f"To run additional models, consider upgrading system memory to at least {min_memory_needed:.1f}GB.")
            report.append("")
            report.append("**Models that would become available**:")
            for model in non_runnable:
                if model.resource_requirements['memory_gb'] <= min_memory_needed + 2:
                    report.append(f"- {model.model} (requires {model.resource_requirements['memory_gb']:.1f}GB)")
            report.append("")
        
        # Technical Notes
        report.append("## Technical Notes")
        report.append("")
        report.append("### Evaluation Methodology")
        report.append("- **Format Handling**: Tests ability to process PDF, DOCX, JSON, and XML documents")
        report.append("- **Information Synthesis**: Tests ability to combine information from multiple sources")
        report.append("- **Large Document Processing**: Tests performance on documents >10MB")
        report.append("- **Conflict Resolution**: Tests reasoning with contradictory information")
        report.append("- **Edge Case Handling**: Tests robustness with corrupted or empty data")
        report.append("")
        
        report.append("### Scoring Methodology")
        report.append("- **Relevance Score**: How well responses address the query")
        report.append("- **Accuracy Score**: Correctness and factual accuracy")
        report.append("- **Completeness Score**: Coverage of expected information")
        report.append("- **Reasoning Quality**: Depth and logic of analysis")
        report.append("- **Overall Score**: Weighted combination of all metrics")
        report.append("")
        
        # Save report
        report_content = "\n".join(report)
        with open(output_file, 'w') as f:
            f.write(report_content)
        
        return report_content


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python recommendation_generator.py <performance_summary_file>")
        sys.exit(1)
    
    results_file = sys.argv[1]
    generator = RecommendationGenerator(results_file)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"evaluation/results/recommendation_report_{timestamp}.md"
    
    report = generator.generate_report(output_file)
    print(f"Recommendation report generated: {output_file}")
    print("\n" + "="*50)
    print("REPORT PREVIEW")
    print("="*50)
    print(report[:2000] + "..." if len(report) > 2000 else report)