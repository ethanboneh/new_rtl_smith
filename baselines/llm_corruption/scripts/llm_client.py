#!/usr/bin/env python3
"""
LLM client for querying reasoning models like o3-mini.

Supports OpenAI API-compatible models.
"""

import os
import json
import re
import yaml
from typing import Dict, Optional, List
from openai import OpenAI


class LLMClient:
    """Client for querying LLMs via OpenAI-compatible API."""
    
    def __init__(
        self,
        model: str = "o3-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        """
        Initialize LLM client.
        
        Args:
            model: Model name (e.g., "o3-mini", "gpt-4")
            api_key: API key (defaults to OPENAI_API_KEY env var)
            base_url: Base URL for API (defaults to OpenAI)
        """
        self.model = model
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            raise ValueError(
                "API key not provided. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
    
    def load_prompt_template(self, template_path: str) -> Dict:
        """Load a YAML prompt template."""
        with open(template_path, 'r') as f:
            return yaml.safe_load(f)
    
    def format_prompt(
        self,
        template: Dict,
        **kwargs
    ) -> tuple[str, str]:
        """
        Format a prompt template with variables.
        
        Returns:
            (system_prompt, user_prompt) tuple
        """
        system = template.get('system', '')
        instance = template.get('instance', '')
        
        # Simple template substitution
        for key, value in kwargs.items():
            placeholder = f"{{{{{key}}}}}"
            system = system.replace(placeholder, str(value))
            instance = instance.replace(placeholder, str(value))
        
        return system, instance
    
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            system_prompt: System prompt
            user_prompt: User prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional API parameters
            
        Returns:
            Generated text
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # o3/o1 models have different API requirements:
        # - Use max_completion_tokens instead of max_tokens
        # - Don't support temperature parameter
        is_o_model = self.model.startswith("o3") or self.model.startswith("o1")
        
        api_params = {
            "model": self.model,
            "messages": messages,
            **kwargs
        }
        
        # Only add temperature for non-o models
        if not is_o_model:
            api_params["temperature"] = temperature
        
        if max_tokens is not None:
            if is_o_model:
                api_params["max_completion_tokens"] = max_tokens
            else:
                api_params["max_tokens"] = max_tokens
        
        response = self.client.chat.completions.create(**api_params)
        
        return response.choices[0].message.content
    
    def remove_bug_comments(self, code: str) -> str:
        """
        Remove all comments from the code.
        The prompt explicitly instructs not to include comments, so we remove
        all single-line (//) and multi-line (/* */) comments.
        """
        if not code:
            return code
        
        lines = code.split('\n')
        cleaned_lines = []
        in_multiline_comment = False
        
        for line in lines:
            # Handle multi-line comments
            if in_multiline_comment:
                # Check if this line closes the multi-line comment
                if '*/' in line:
                    # Extract everything after */
                    line = line.split('*/', 1)[1]
                    in_multiline_comment = False
                else:
                    # Still inside multi-line comment, skip this line
                    continue
            
            # Check for multi-line comment start
            if '/*' in line:
                parts = line.split('/*', 1)
                before_comment = parts[0]
                after_start = parts[1] if len(parts) > 1 else ''
                
                # Check if it closes on the same line
                if '*/' in after_start:
                    # Extract everything after */
                    line = before_comment + after_start.split('*/', 1)[1]
                    in_multiline_comment = False
                else:
                    # Multi-line comment continues, keep only what's before /*
                    line = before_comment
                    in_multiline_comment = True
            
            # Remove single-line comments (//)
            # But preserve // in strings (though SystemVerilog doesn't use // in strings like C)
            # Remove everything after // on the line
            if '//' in line:
                line = line.split('//')[0].rstrip()
            
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def corrupt_code(
        self,
        clean_code: str,
        template_path: Optional[str] = None,
        feedback: str = ""
    ) -> Dict[str, str]:
        """
        Generate corrupted code from clean code.
        
        Args:
            clean_code: Clean SystemVerilog code to corrupt
            template_path: Path to prompt template
            feedback: Optional feedback from previous attempts
        
        Returns:
            Dictionary with 'explanation' and 'corrupted_code'
        """
        if template_path is None:
            template_path = os.path.join(
                os.path.dirname(__file__),
                "..",
                "prompts",
                "corruption_prompt.yaml"
            )
        
        template = self.load_prompt_template(template_path)
        system, user = self.format_prompt(template, src_code=clean_code)
        
        # Add feedback to user prompt if provided
        if feedback:
            user = user + feedback
        
        response = self.generate(
            system_prompt=system,
            user_prompt=user,
            temperature=0.8,
            max_tokens=4096
        )
        
        # Parse response
        explanation = ""
        corrupted_code = ""
        
        # Extract explanation
        if "Explanation:" in response:
            parts = response.split("Explanation:", 1)
            if len(parts) > 1:
                explanation = parts[1].split("Bugged Code:")[0].strip()
        
        # Extract code block
        code_block_pattern = r'```(?:\w+)?\n(.*?)```'
        matches = re.findall(code_block_pattern, response, re.DOTALL)
        if matches:
            corrupted_code = matches[-1].strip()
        else:
            # Fallback: try to extract code after "Bugged Code:"
            if "Bugged Code:" in response:
                code_part = response.split("Bugged Code:", 1)[1]
                # Try to find code block
                matches = re.findall(code_block_pattern, code_part, re.DOTALL)
                if matches:
                    corrupted_code = matches[0].strip()
        
        # Remove bug-related comments from the corrupted code
        if corrupted_code:
            corrupted_code = self.remove_bug_comments(corrupted_code)
        
        return {
            'explanation': explanation,
            'corrupted_code': corrupted_code,
            'raw_response': response
        }
    
    def generate_issue_description(
        self,
        clean_code: str,
        corrupted_code: str,
        template_path: Optional[str] = None
    ) -> str:
        """Generate issue description from clean and corrupted code."""
        if template_path is None:
            template_path = os.path.join(
                os.path.dirname(__file__),
                "..",
                "prompts",
                "issue_description_prompt.yaml"
            )
        
        template = self.load_prompt_template(template_path)
        system, user = self.format_prompt(
            template,
            clean_code=clean_code,
            corrupted_code=corrupted_code
        )
        
        response = self.generate(
            system_prompt=system,
            user_prompt=user,
            temperature=0.5,
            max_tokens=2048
        )
        
        # Extract issue description
        if "Issue Description:" in response:
            return response.split("Issue Description:", 1)[1].strip()
        return response.strip()
    
    def generate_reasoning_trace(
        self,
        clean_code: str,
        corrupted_code: str,
        issue_description: str,
        template_path: Optional[str] = None
    ) -> str:
        """Generate reasoning trace from code and issue description."""
        if template_path is None:
            template_path = os.path.join(
                os.path.dirname(__file__),
                "..",
                "prompts",
                "reasoning_trace_prompt.yaml"
            )
        
        template = self.load_prompt_template(template_path)
        system, user = self.format_prompt(
            template,
            clean_code=clean_code,
            corrupted_code=corrupted_code,
            issue_description=issue_description
        )
        
        response = self.generate(
            system_prompt=system,
            user_prompt=user,
            temperature=0.3,  # Lower temperature for more consistent reasoning
            max_tokens=4096
        )
        
        # Extract reasoning trace
        if "Reasoning Trace:" in response:
            return response.split("Reasoning Trace:", 1)[1].strip()
        return response.strip()

