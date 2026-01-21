# LLM Evaluation Harness

**Status**: 📋 Planned

A reusable evaluation framework for LLM-based systems including RAG, agents, and structured extraction.

## Planned Features

- Flexible dataset specification (prompt, expected, context, metadata)
- Multiple evaluator types:
  - Exact match
  - Regex patterns
  - Semantic similarity
  - LLM-as-judge rubric scoring
- Regression gates for CI/CD
- Full trace logging (prompts, responses, latency, cost)
- Confusion bucket analysis (formatting vs factual vs tool errors)

## Use Cases

- RAG pipeline evaluation
- Agent task completion scoring
- OCR post-processing quality
- Structured data extraction accuracy

## Coming Soon

This project will demonstrate:
- Building evaluation datasets
- Implementing custom evaluators
- Setting up regression gates
- Trace analysis and debugging
