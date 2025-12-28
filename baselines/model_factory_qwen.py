#!/usr/bin/env python3
"""
Custom model factory for Qwen3-8B with Tinker integration.

This factory can be used with the CVDP benchmark to evaluate models
that are hosted on Tinker (either base or fine-tuned).

IMPORTANT: This handles code extraction from VeriThoughts-style output:
- Reasoning: <think>...</think> followed by [BEGIN]...[DONE]
- Instruction: [BEGIN]...[DONE]

Usage with CVDP benchmark:
    cd cvdp_benchmark
    python run_benchmark.py -f dataset.jsonl -l -m tinker-qwen3-8b-base \
        -c /path/to/model_factory_qwen.py
"""

import asyncio
import logging
import os
import re
import sys
from typing import Optional, Any, Tuple, Dict

logger = logging.getLogger(__name__)


class TinkerQwenInstance:
    """
    Model instance that uses Tinker's sampling API for inference.
    
    This can be used with both the base Qwen3-8B model and fine-tuned versions.
    Matches the interface expected by CVDP (similar to OpenAI_Instance).
    """
    
    def __init__(
        self,
        context: Any = "You are a helpful assistant.",
        key: Optional[str] = None,
        model: str = "tinker-qwen3-8b",
        model_path: Optional[str] = None,
        base_model: str = "Qwen/Qwen3-8B",
        temperature: float = 0.6,
        max_tokens: int = 16384,
    ):
        self.context = context
        self.model = model
        self.model_path = model_path  # Tinker checkpoint path (e.g., tinker://...)
        self.base_model = base_model
        self.temperature = temperature
        # Allow large max_tokens to accommodate reasoning + code generation
        # Qwen3-8B supports up to 32K context, we use 16K for output to leave room for input
        self.max_tokens = max_tokens if max_tokens > 0 else 16384
        self.debug = False
        self.save_prompts = True  # Save prompts to disk for debugging
        self._sampling_client = None
        self._tokenizer = None
        self._renderer = None
        
        # Import tinker here to avoid import errors when module is loaded
        try:
            import tinker
            self._tinker = tinker
            logger.info(f"Created TinkerQwenInstance: model={model}, base={base_model}")
        except ImportError as e:
            logger.error(f"Failed to import tinker: {e}")
            raise
    
    @property
    def requires_evaluation(self) -> bool:
        """Whether this model requires harness evaluation."""
        return True
    
    def set_debug(self, debug: bool = True) -> None:
        """Enable or disable debug mode."""
        self.debug = debug
        
    def _get_sampling_client(self):
        """Lazily initialize the sampling client."""
        if self._sampling_client is None:
            service_client = self._tinker.ServiceClient()
            if self.model_path:
                # Load a fine-tuned model from checkpoint
                logger.info(f"Loading model from checkpoint: {self.model_path}")
                self._sampling_client = service_client.create_sampling_client(
                    model_path=self.model_path,
                    base_model=self.base_model
                )
            else:
                # Use base model
                logger.info(f"Using base model: {self.base_model}")
                self._sampling_client = service_client.create_sampling_client(
                    base_model=self.base_model
                )
        return self._sampling_client
    
    def _get_tokenizer_and_renderer(self):
        """Lazily initialize tokenizer and renderer (cached for performance)."""
        if self._tokenizer is None or self._renderer is None:
            from tinker_cookbook.tokenizer_utils import get_tokenizer
            from tinker_cookbook.renderers import get_renderer
            from tinker_cookbook import model_info
            
            self._tokenizer = get_tokenizer(self.base_model)
            renderer_name = model_info.get_recommended_renderer_name(self.base_model)
            self._renderer = get_renderer(renderer_name, self._tokenizer)
        
        return self._tokenizer, self._renderer
    
    def prompt(
        self,
        prompt: str,
        schema: Optional[Any] = None,
        prompt_log: str = "",
        files: list = [],
        timeout: int = 60,
        category: Optional[int] = None,
    ) -> Tuple[Dict, bool]:
        """
        Send a prompt to the model and return the response.
        
        Args:
            prompt: The input prompt (user message)
            schema: JSON schema for structured output (not used)
            prompt_log: Path for logging prompts
            files: List of expected output files
            timeout: Request timeout in seconds
            category: Problem category
            
        Returns:
            Tuple of (output_dict, success_bool) matching CVDP's expected format
        """
        # Get sampling client
        sampling_client = self._get_sampling_client()
        
        # Build messages - use context as system prompt if it's a string
        system_prompt = self.context if isinstance(self.context, str) else "You are a helpful assistant."
        
        # Add instruction to wrap final code in markdown code blocks for reliable extraction
        # Keep thinking enabled but ensure code is properly marked
        user_prompt = prompt + "\n\nAfter your reasoning, provide the final Verilog/SystemVerilog code wrapped in ```systemverilog and ``` markers."
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Log the prompt if requested or save_prompts is enabled
        if prompt_log and (self.debug or self.save_prompts):
            try:
                prompt_dir = os.path.dirname(prompt_log)
                if prompt_dir:
                    os.makedirs(prompt_dir, exist_ok=True)
                with open(prompt_log, "w") as f:
                    f.write(f"=== SYSTEM PROMPT ===\n{system_prompt}\n\n")
                    f.write(f"=== USER PROMPT ===\n{user_prompt}\n")
                logger.info(f"Saved prompt to {prompt_log}")
            except Exception as e:
                logger.warning(f"Failed to write prompt log: {e}")
        
        # Sample from the model
        try:
            response = asyncio.run(self._sample_async(sampling_client, messages))
            logger.info(f"Got response of length {len(response)}")
        except Exception as e:
            logger.error(f"Error sampling from model: {e}")
            return {}, False
        
        # Extract code from response (strips reasoning traces)
        code = self._extract_code(response)
        
        # Save full response to prompt log if enabled
        if prompt_log and self.save_prompts:
            try:
                with open(prompt_log, "a") as f:
                    f.write(f"\n\n=== FULL MODEL RESPONSE ===\n{response}\n")
                    f.write(f"\n\n=== EXTRACTED CODE ===\n{code}\n")
            except Exception as e:
                logger.warning(f"Failed to append response to prompt log: {e}")
        
        if not code:
            logger.warning("No code extracted from response")
            return {}, False
        
        # Return in CVDP's expected format
        return {"direct_text": code}, True
    
    async def _sample_async(self, client, messages: list) -> str:
        """Sample from the model asynchronously."""
        # Use cached tokenizer and renderer for better performance
        tokenizer, renderer = self._get_tokenizer_and_renderer()
        
        # Build the prompt - already returns ModelInput
        model_input = renderer.build_generation_prompt(messages)
        
        # Sample using the correct API
        result = await client.sample_async(
            prompt=model_input,
            num_samples=1,
            sampling_params=self._tinker.SamplingParams(
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            ),
        )
        
        # Decode the response - result.sequences is a list of SampledSequence
        if result and result.sequences and len(result.sequences) > 0:
            output_tokens = result.sequences[0].tokens
            response_text = tokenizer.decode(output_tokens, skip_special_tokens=True)
            return response_text
        
        return ""
    
    def _extract_code(self, response: str) -> str:
        """
        Extract code from the model response.
        
        Handles:
        - [BEGIN]...[DONE] markers (VeriThoughts format)
        - CODE BEGIN...CODE END markers
        - Markdown code blocks
        - module...endmodule patterns
        """
        # Method 1: [BEGIN]...[DONE] markers (our training format)
        begin_match = re.search(
            r'\[BEGIN\]\s*(.*?)\s*\[DONE\]',
            response,
            re.DOTALL | re.IGNORECASE
        )
        if begin_match:
            return begin_match.group(1).strip()
        
        # Method 2: CODE BEGIN...CODE END (original VeriThoughts)
        code_match = re.search(
            r'CODE\s*BEGIN\s*(.*?)\s*CODE\s*END',
            response,
            re.DOTALL | re.IGNORECASE
        )
        if code_match:
            return code_match.group(1).strip()
        
        # Method 3: Markdown code blocks
        md_match = re.search(
            r'```(?:verilog|systemverilog|sv)?\s*(.*?)\s*```',
            response,
            re.DOTALL | re.IGNORECASE
        )
        if md_match:
            return md_match.group(1).strip()
        
        # Method 4: Find module...endmodule block(s)
        modules = re.findall(
            r'(module\s+\w+.*?endmodule)',
            response,
            re.DOTALL | re.IGNORECASE
        )
        if modules:
            return '\n\n'.join(modules)
        
        # Method 5: Strip <think>...</think> tags
        stripped = re.sub(
            r'<think>.*?</think>',
            '',
            response,
            flags=re.DOTALL | re.IGNORECASE
        ).strip()
        if stripped and stripped != response:
            return stripped
        
        # Fallback: Return full response
        return response.strip()


class CustomModelFactory:
    """
    Custom model factory that adds Tinker-based Qwen3-8B support.
    
    This extends the default CVDP ModelFactory with Tinker model types.
    
    IMPORTANT: When using fine-tuned models, ensure you use the `sampler_weights` path
    from checkpoints.jsonl, NOT the `state_path` (which uses `weights`).
    
    Correct:   tinker://xxx:train:0/sampler_weights/final
    Wrong:     tinker://xxx:train:0/weights/final
    """
    
    def __init__(self):
        # Register model types
        self.model_types = {
            "tinker-qwen3-8b": self._create_tinker_qwen_instance,
            "tinker-qwen3-8b-base": self._create_tinker_qwen_base,
            "tinker-qwen3-8b-reasoning": self._create_tinker_qwen_reasoning,
            "tinker-qwen3-8b-instruction": self._create_tinker_qwen_instruction,
            # sbj_score is used by CVDP for subjective code quality scoring
            "sbj_score": self._create_subjective_scorer,
        }
        logger.info(f"CustomModelFactory initialized with models: {list(self.model_types.keys())}")
    
    def create_model(self, model_name: str, context: Any = None, key: Optional[str] = None, **kwargs) -> Any:
        """Create a model instance based on the model name."""
        if model_name in self.model_types:
            return self.model_types[model_name](model_name, context, key, **kwargs)
        else:
            raise ValueError(f"Unsupported model type: {model_name}. Available: {list(self.model_types.keys())}")
    
    def _create_tinker_qwen_instance(self, model_name: str, context: Any, key: Optional[str], **kwargs) -> TinkerQwenInstance:
        """Create a generic Tinker Qwen instance."""
        return TinkerQwenInstance(context=context, key=key, model=model_name, **kwargs)
    
    def _create_tinker_qwen_base(self, model_name: str, context: Any, key: Optional[str], **kwargs) -> TinkerQwenInstance:
        """Create a base Qwen3-8B instance (no fine-tuning)."""
        return TinkerQwenInstance(
            context=context,
            key=key,
            model=model_name,
            model_path=None,  # Use base model
            **kwargs
        )
    
    def _create_tinker_qwen_reasoning(self, model_name: str, context: Any, key: Optional[str], **kwargs) -> TinkerQwenInstance:
        """Create a Qwen instance fine-tuned on reasoning dataset."""
        model_path = os.environ.get("TINKER_REASONING_MODEL_PATH")
        if not model_path:
            logger.warning("TINKER_REASONING_MODEL_PATH not set, using base model")
        else:
            # Validate that it's a sampler_weights path
            if "/weights/" in model_path and "/sampler_weights/" not in model_path:
                logger.error(
                    f"TINKER_REASONING_MODEL_PATH uses 'weights' but should use 'sampler_weights'!\n"
                    f"  Current: {model_path}\n"
                    f"  Should be: {model_path.replace('/weights/', '/sampler_weights/')}"
                )
        return TinkerQwenInstance(
            context=context,
            key=key,
            model=model_name,
            model_path=model_path,
            **kwargs
        )
    
    def _create_tinker_qwen_instruction(self, model_name: str, context: Any, key: Optional[str], **kwargs) -> TinkerQwenInstance:
        """Create a Qwen instance fine-tuned on instruction dataset."""
        model_path = os.environ.get("TINKER_INSTRUCTION_MODEL_PATH")
        if not model_path:
            logger.warning("TINKER_INSTRUCTION_MODEL_PATH not set, using base model")
        else:
            # Validate that it's a sampler_weights path
            if "/weights/" in model_path and "/sampler_weights/" not in model_path:
                logger.error(
                    f"TINKER_INSTRUCTION_MODEL_PATH uses 'weights' but should use 'sampler_weights'!\n"
                    f"  Current: {model_path}\n"
                    f"  Should be: {model_path.replace('/weights/', '/sampler_weights/')}"
                )
        return TinkerQwenInstance(
            context=context,
            key=key,
            model=model_name,
            model_path=model_path,
            **kwargs
        )
    
    def _create_subjective_scorer(self, model_name: str, context: Any, key: Optional[str], **kwargs) -> Any:
        """
        Create a model for subjective scoring of code quality.
        
        CVDP uses this for LLM-based evaluation of generated code.
        We try OpenAI first (if available), then fall back to Tinker Qwen.
        """
        # Try to use OpenAI if API key is available
        openai_key = key or os.environ.get("OPENAI_API_KEY")
        if openai_key:
            try:
                # Try to import and use CVDP's OpenAI integration
                from src.llm_lib.openai_llm import OpenAI_Instance
                logger.info("Using OpenAI for subjective scoring")
                return OpenAI_Instance(
                    context=context or "You are an expert code reviewer.",
                    key=openai_key,
                    model="gpt-4o-mini",  # Use cheaper model for scoring
                )
            except ImportError:
                logger.warning("Could not import OpenAI_Instance, falling back to Tinker")
            except Exception as e:
                logger.warning(f"Failed to create OpenAI instance: {e}, falling back to Tinker")
        
        # Fall back to Tinker Qwen base model for scoring
        logger.info("Using Tinker Qwen3-8B for subjective scoring")
        return TinkerQwenInstance(
            context=context or "You are an expert code reviewer. Evaluate the following code.",
            key=key,
            model="sbj_score",
            model_path=None,  # Use base model
            temperature=0.3,  # Lower temperature for more consistent scoring
            **kwargs
        )
    
    def register_model_type(self, model_identifier: str, factory_method):
        """Register a new model type."""
        self.model_types[model_identifier] = factory_method


if __name__ == "__main__":
    # Test the factory
    factory = CustomModelFactory()
    print(f"Available model types: {list(factory.model_types.keys())}")
    
    # Test code extraction
    test_instance = TinkerQwenInstance.__new__(TinkerQwenInstance)
    test_instance.debug = False
    
    # Test VeriThoughts reasoning format
    reasoning_output = """<think>
Let me think about this problem step by step.
First, I need to understand the requirements...
</think>
[BEGIN]
module test_module(
    input wire clk,
    input wire rst,
    output reg [7:0] data
);
    always @(posedge clk) begin
        if (rst) data <= 8'b0;
        else data <= data + 1;
    end
endmodule
[DONE]"""
    
    extracted = test_instance._extract_code(reasoning_output)
    print(f"\nTest reasoning format extraction:")
    print(f"Extracted code:\n{extracted[:200]}...")
    
    # Test instruction format
    instruction_output = """[BEGIN]
module simple(input a, output b);
    assign b = a;
endmodule
[DONE]"""
    
    extracted2 = test_instance._extract_code(instruction_output)
    print(f"\nTest instruction format extraction:")
    print(f"Extracted code:\n{extracted2}")
