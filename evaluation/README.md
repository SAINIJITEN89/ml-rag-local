# Model Evaluation Framework

This comprehensive evaluation framework tests the performance of different language models on various RAG (Retrieval-Augmented Generation) tasks.

## Overview

The evaluation system assesses models across multiple dimensions:

1. **System Compatibility** - Can the model run on your hardware?
2. **Document Format Handling** - PDF, DOCX, JSON, XML processing
3. **Information Synthesis** - Combining data from multiple sources
4. **Large Document Processing** - Handling substantial text volumes
5. **Conflict Resolution** - Reasoning with contradictory information
6. **Edge Case Handling** - Robustness with corrupted/empty data
7. **Reasoning Quality** - Depth of analysis and logical thinking

## Quick Start

1. **Ensure Prerequisites**:
   ```bash
   # Ollama must be running
   ollama serve
   
   # Install required Python packages
   pip install pandas requests psutil
   ```

2. **Run Complete Evaluation**:
   ```bash
   cd evaluation
   python run_evaluation.py
   ```

3. **View Results**:
   - Check `results/recommendation_report_<timestamp>.md` for the main report
   - Review `results/performance_summary_<timestamp>.json` for detailed metrics

## Files and Structure

```
evaluation/
├── README.md                          # This file
├── run_evaluation.py                  # Main evaluation runner
├── model_evaluator.py                 # Core evaluation framework
├── recommendation_generator.py        # Report generation
├── test_data/                         # Test documents
│   └── documents/
│       ├── txt/                       # Text test files
│       ├── pdf/                       # PDF test files (future)
│       ├── docx/                      # DOCX test files (future)
│       ├── json/                      # JSON test files (future)
│       └── xml/                       # XML test files (future)
└── results/                           # Generated results
    ├── detailed_results_<timestamp>.json
    ├── performance_summary_<timestamp>.json
    ├── results_<timestamp>.csv
    └── recommendation_report_<timestamp>.md
```

## Test Categories

### 1. Format Handling Tests
- **PDF Processing**: Extract information from PDF documents
- **DOCX Processing**: Handle Microsoft Word documents
- **JSON Processing**: Parse structured JSON data
- **XML Processing**: Process XML configuration files

### 2. Information Synthesis Tests
- **Cross-Document Synthesis**: Combine information from multiple sources
- **Multi-Source Timeline**: Create comprehensive timelines from scattered data

### 3. Large Document Tests
- **Large Technical Document**: Process extensive technical specifications
- **Comprehensive Manual**: Extract information from lengthy manuals

### 4. Conflict Resolution Tests
- **Version Conflicts**: Resolve contradictions between document versions
- **Policy Contradictions**: Determine current effective policies

### 5. Reasoning Depth Tests
- **Causal Analysis**: Identify root causes and relationships
- **Strategic Planning**: Make recommendations based on complex data

### 6. Edge Case Tests
- **Empty Document Handling**: Handle documents with no content
- **Corrupted Data Handling**: Extract useful information despite corruption

## Evaluation Metrics

### Primary Metrics
- **Success Rate**: Percentage of tests completed without errors
- **Response Time**: Average time to generate responses
- **Relevance Score**: How well responses address queries (0-1)
- **Accuracy Score**: Correctness of information provided (0-1)
- **Completeness Score**: Coverage of expected information (0-1)
- **Reasoning Quality**: Depth and logic of analysis (0-1)

### Secondary Metrics
- **Memory Usage**: RAM requirements for model operation
- **Failure Rate**: Percentage of failed test attempts
- **Confidence Score**: Reliability of the recommendation

## Scoring Methodology

### Overall Score Calculation
```
Overall Score = 0.25 × Success Rate +
                0.20 × Relevance Score +
                0.20 × Accuracy Score +
                0.15 × Completeness Score +
                0.15 × Reasoning Quality +
                0.05 × Speed Score
```

### Individual Metric Scoring
- **Relevance**: Query-response word overlap analysis
- **Accuracy**: Heuristics based on response structure and length
- **Completeness**: Coverage of expected information keywords
- **Reasoning**: Presence of analytical language and logic indicators
- **Speed**: Inverse relationship with response time

## Customization

### Adding New Test Cases
1. Edit `model_evaluator.py`
2. Add test scenarios to the appropriate `_get_*_tests()` method
3. Place test documents in the `test_data/documents/` directory

### Modifying Scoring
1. Update the `_score_response()` method in `model_evaluator.py`
2. Adjust weights in `_calculate_overall_score()` method
3. Add new metrics as needed

### Adding New Models
1. Update `model_config.py` in the parent directory
2. Add model configuration with memory requirements
3. The evaluation will automatically include new models

## Output Files

### Recommendation Report (`recommendation_report_<timestamp>.md`)
- Executive summary with best model recommendation
- System compatibility analysis
- Detailed model rankings with strengths/weaknesses
- Use case recommendations
- Implementation guidance

### Performance Summary (`performance_summary_<timestamp>.json`)
- Aggregated metrics for each model
- Overall performance scores
- Resource requirements
- Success/failure rates

### Detailed Results (`detailed_results_<timestamp>.json`)
- Individual test results for each model
- Complete response texts
- Timing information
- Error details

### CSV Export (`results_<timestamp>.csv`)
- Tabular format for easy analysis
- Compatible with Excel and data analysis tools
- Suitable for creating custom visualizations

## System Requirements

### Minimum Requirements
- 4GB RAM (for smallest models)
- Python 3.8+
- Ollama service running
- 1GB free disk space

### Recommended Requirements
- 8GB+ RAM (for better model selection)
- SSD storage for faster document processing
- Stable internet connection for model downloads

## Troubleshooting

### Common Issues

1. **"Cannot connect to Ollama"**
   - Ensure Ollama is running: `ollama serve`
   - Check if service is accessible: `curl http://localhost:11434/api/tags`

2. **"Model not available"**
   - Pull required models: `ollama pull <model_name>`
   - Check available models: `ollama list`

3. **"Insufficient memory"**
   - Models will be skipped if they can't fit in available RAM
   - Close other applications or upgrade system memory

4. **"Evaluation takes too long"**
   - Large models on slower hardware may take 30+ minutes
   - Consider running evaluation on a subset of models first

### Performance Tips

1. **Faster Evaluation**:
   - Use smaller models for initial testing
   - Reduce test document sizes
   - Run on systems with more RAM/CPU

2. **More Comprehensive Testing**:
   - Add more test documents
   - Increase variety of test scenarios
   - Test with domain-specific documents

## Future Enhancements

- Support for additional document formats (EPUB, RTF, etc.)
- Integration with external evaluation metrics
- Automated model comparison reports
- Performance trending over time
- Custom evaluation scenarios for specific domains
- GPU acceleration support
- Distributed evaluation across multiple machines

## Contributing

To contribute new test cases or improvements:

1. Add test documents to the appropriate `test_data/documents/` subdirectory
2. Update the test scenario definitions in `model_evaluator.py`
3. Ensure new tests follow the existing pattern and include expected information lists
4. Test your changes with a small subset of models first

## License

This evaluation framework is part of the larger RAG system project and follows the same licensing terms.