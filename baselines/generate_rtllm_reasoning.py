#!/usr/bin/env python3
"""
Generate synthetic reasoning traces for RTLLM dataset.

This script takes RTLLM spec→code pairs and uses an LLM to generate
reasoning traces that explain how to arrive at the correct code.

Usage:
    # Using OpenAI
    export OPENAI_API_KEY=sk-...
    python generate_rtllm_reasoning.py --provider openai
    
    # Using Tinker (base Qwen3-8B)
    python generate_rtllm_reasoning.py --provider tinker
"""

import argparse
import asyncio
import json
import os
from pathlib import Path
from tqdm import tqdm
from typing import Optional

from config import RTLLM_PATH, DATASETS_DIR, create_directories


REASONING_PROMPT = """You are an expert Verilog designer. Given a specification and the correct Verilog implementation, explain your reasoning process for how you would arrive at this solution.

## Specification:
{spec}

## Correct Implementation:
```verilog
{code}
```

Explain your step-by-step reasoning for how to implement this specification. Focus on:
1. Understanding the requirements (inputs, outputs, behavior)
2. Choosing the right approach (combinational vs sequential, data structures)
3. Key implementation decisions
4. Edge cases to handle

Provide your reasoning in a clear, educational manner. Do NOT include the final code - just explain the thought process."""


def load_rtllm_dataset(rtllm_path: str = RTLLM_PATH):
    """Load RTLLM dataset from local directory."""
    from prepare_datasets import load_rtllm_dataset as _load
    return _load(rtllm_path)


async def generate_reasoning_openai(spec: str, code: str, client) -> Optional[str]:
    """Generate reasoning using OpenAI API."""
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",  # Use gpt-4o for better quality
            messages=[
                {"role": "system", "content": "You are an expert Verilog RTL designer who explains your reasoning clearly."},
                {"role": "user", "content": REASONING_PROMPT.format(spec=spec, code=code)}
            ],
            temperature=0.7,
            max_tokens=2000,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"OpenAI error: {e}")
        return None


async def generate_reasoning_tinker(spec: str, code: str, client, tokenizer, renderer) -> Optional[str]:
    """Generate reasoning using Tinker (base Qwen3-8B)."""
    import tinker
    
    messages = [
        {"role": "system", "content": "You are an expert Verilog RTL designer who explains your reasoning clearly."},
        {"role": "user", "content": REASONING_PROMPT.format(spec=spec, code=code)}
    ]
    
    try:
        model_input = renderer.build_generation_prompt(messages)
        result = await client.sample_async(
            prompt=model_input,
            num_samples=1,
            sampling_params=tinker.SamplingParams(
                max_tokens=2000,
                temperature=0.7,
            ),
        )
        
        if result and result.sequences:
            return tokenizer.decode(result.sequences[0].tokens, skip_special_tokens=True)
        return None
    except Exception as e:
        print(f"Tinker error: {e}")
        return None


async def main_async(args):
    """Main async function."""
    create_directories()
    
    # Load RTLLM dataset
    print(f"Loading RTLLM dataset from {RTLLM_PATH}...")
    rtllm_data = load_rtllm_dataset(RTLLM_PATH)
    print(f"Loaded {len(rtllm_data)} examples")
    
    # Limit if specified
    if args.limit:
        rtllm_data = rtllm_data[:args.limit]
        print(f"Limited to {len(rtllm_data)} examples")
    
    # Initialize client based on provider
    if args.provider == "openai":
        from openai import AsyncOpenAI
        client = AsyncOpenAI()
        tokenizer = None
        renderer = None
    else:  # tinker
        import tinker
        from tinker_cookbook.tokenizer_utils import get_tokenizer
        from tinker_cookbook.renderers import get_renderer
        from tinker_cookbook import model_info
        
        service_client = tinker.ServiceClient()
        client = service_client.create_sampling_client(base_model="Qwen/Qwen3-8B")
        tokenizer = get_tokenizer("Qwen/Qwen3-8B")
        renderer_name = model_info.get_recommended_renderer_name("Qwen/Qwen3-8B")
        renderer = get_renderer(renderer_name, tokenizer)
    
    # Generate reasoning for each example
    results = []
    for item in tqdm(rtllm_data, desc="Generating reasoning"):
        spec = item["description"]
        code = item["code"]
        
        if args.provider == "openai":
            reasoning = await generate_reasoning_openai(spec, code, client)
        else:
            reasoning = await generate_reasoning_tinker(spec, code, client, tokenizer, renderer)
        
        if reasoning:
            results.append({
                "description": spec,
                "code": code,
                "reasoning": reasoning,
                "design_name": item.get("design_name", ""),
                "category": item.get("category", ""),
                "subcategory": item.get("subcategory", ""),
            })
        
        # Rate limiting for OpenAI
        if args.provider == "openai":
            await asyncio.sleep(0.5)
    
    # Save results
    output_path = os.path.join(DATASETS_DIR, "rtllm_with_reasoning.jsonl")
    print(f"\nSaving {len(results)} examples with reasoning to {output_path}...")
    with open(output_path, 'w') as f:
        for item in results:
            f.write(json.dumps(item) + "\n")
    
    print(f"✓ Done! Generated reasoning for {len(results)}/{len(rtllm_data)} examples")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic reasoning traces for RTLLM")
    parser.add_argument(
        "--provider",
        choices=["openai", "tinker"],
        default="openai",
        help="LLM provider to use for generating reasoning"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of examples to process"
    )
    
    args = parser.parse_args()
    
    if args.provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set")
        print("Set it with: export OPENAI_API_KEY=sk-...")
        return
    
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()



