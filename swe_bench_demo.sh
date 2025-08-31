#!/bin/bash

# SWE-bench Evaluation Demo
# This script demonstrates how to run SWE-bench evaluations with OpenBench

echo "=== SWE-bench Integration Demo ==="
echo

echo "1. Available SWE-bench benchmarks:"
bench list | grep swe
echo

echo "2. Describe SWE-bench Lite (smallest variant for testing):"
bench describe swe_bench_lite
echo

echo "3. Example command to run SWE-bench Lite with a model:"
echo "   bench eval swe_bench_lite --model groq/llama-3.1-70b-versatile --limit 5"
echo
echo "   This would evaluate the model on 5 SWE-bench instances for quick testing."
echo

echo "4. Example command to run full evaluation:"
echo "   bench eval swe_bench_verified --model openai/gpt-4"
echo
echo "   This would run the full verified SWE-bench dataset."
echo

echo "5. Commands for different SWE-bench variants:"
echo "   - SWE-bench Lite (300 instances):     bench eval swe_bench_lite --model <model>"
echo "   - SWE-bench Verified (subset):        bench eval swe_bench_verified --model <model>"  
echo "   - SWE-bench Full (2000+ instances):   bench eval swe_bench_full --model <model>"
echo

echo "6. Model providers supported:"
echo "   - OpenAI: openai/gpt-4, openai/gpt-3.5-turbo"
echo "   - Anthropic: anthropic/claude-3-5-sonnet-20241022"
echo "   - Groq: groq/llama-3.1-70b-versatile, groq/mixtral-8x7b-32768"
echo "   - And 30+ other providers via Inspect AI"
echo

echo "=== Demo Complete ==="
echo "Note: Actual evaluation requires API keys for the model providers."