#!/usr/bin/env python3
"""
Evaluate a LoRA fine-tuned model on VerilogEval.
This script extends run_local_eval.py to support loading LoRA adapters.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
except ModuleNotFoundError as exc:
    raise SystemExit(
        "ERROR: Required packages (torch, transformers, peft) are missing. "
        "Install them, e.g. `pip install torch transformers peft accelerate`."
    ) from exc

# Import prompt utilities from the evaluation script
PROMPT_SYSTEM_SPEC_TO_RTL = (
    "You are a Verilog RTL designer that only writes code using correct "
    "Verilog syntax."
)

PROMPT_RULES_SUFFIX = """\
Here are some additional rules and coding conventions.

 - Declare all ports and signals as logic; do not to use wire or reg.

 - For combinational logic with an always block do not explicitly specify
   the sensitivity list; instead use always @(*).

 - All sized numeric constants must have a size greater than zero
   (e.g, 0'b0 is not a valid expression).

 - An always block must read at least one signal otherwise it will never
   be executed; use an assign statement instead of an always block in
   situations where there is no need to read any signals.

 - if the module uses a synchronous reset signal, this means the reset
   signal is sampled with respect to the clock. When implementing a
   synchronous reset signal, do not include posedge reset in the
   sensitivity list of any sequential always block.
"""

PROMPT_NO_EXPLAIN_SUFFIX = (
    "Enclose your code with [BEGIN] and [DONE]. Only output the code snippet "
    "and do NOT output anything else."
)


def load_examples(task: str, examples: int, scripts_dir: Path) -> str:
    """Load few-shot examples from the repository if requested."""
    if task != "spec-to-rtl" or examples <= 0:
        return ""

    example_filename = (
        f"verilog-example-prefix_{task}_{examples}-shot.txt"
        if examples != 4
        else f"verilog-example-prefix_{task}_{examples}-shot default.txt"
    )
    candidate = scripts_dir / example_filename
    if not candidate.exists():
        raise FileNotFoundError(
            f"Unable to locate few-shot example file: {candidate}"
        )
    return candidate.read_text()


def build_prompt(
    *,
    problem_text: str,
    include_rules: bool,
    include_examples: str,
) -> Tuple[str, str]:
    """Construct system and user prompts mirroring sv-generate logic."""
    system_msg = PROMPT_SYSTEM_SPEC_TO_RTL

    user_parts: List[str] = []
    if include_examples:
        user_parts.append(include_examples.rstrip("\n"))

    user_parts.append("Question:")
    user_parts.append(problem_text.strip())

    if include_rules:
        user_parts.append(PROMPT_RULES_SUFFIX.rstrip("\n"))

    user_parts.append(PROMPT_NO_EXPLAIN_SUFFIX)
    user_parts.append("Answer:")

    user_prompt = "\n".join(user_parts).strip() + "\n"
    return system_msg.strip(), user_prompt


@dataclass
class ProblemResult:
    problem: str
    status: str
    compile_returncode: int
    runtime_returncode: Optional[int]
    duration_s: float
    prompt_tokens: int
    response_tokens: int
    verilog_path: Path
    generate_log: Path
    compile_log: Path
    raw_response_path: Path
    metadata: Dict[str, object] = field(default_factory=dict)


def sanitize_name(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    slug = re.sub(r"_+", "_", slug)
    return slug.strip("_").lower() or "run"


def read_problem_list(dataset_dir: Path, selected: Sequence[str]) -> List[str]:
    if selected:
        return [p.strip() for p in selected if p.strip()]

    problems_txt = dataset_dir / "problems.txt"
    if not problems_txt.exists():
        raise FileNotFoundError(f"Missing problems.txt in {dataset_dir}")

    problems: List[str] = []
    for line in problems_txt.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        problems.append(line)
    if not problems:
        raise RuntimeError(f"No problems found in {problems_txt}")
    return problems


def ensure_tools_available() -> None:
    if shutil.which("iverilog") is None:
        raise SystemExit(
            "ERROR: iverilog not found in PATH. Install Icarus Verilog v12."
        )


def extract_code_from_response(response: str) -> Tuple[str, Dict[str, object]]:
    """Extract Verilog code from model output."""
    metadata: Dict[str, object] = {}
    cleaned = response

    # Prefer [BEGIN]/[DONE]-style markers
    # Use rfind to find the LAST occurrence of [BEGIN] to avoid matching it in the prompt
    if "[BEGIN]" in response:
        end_markers = ["[DONE]", "[END]", "[FINISH]", "[STOP]"]
        try:
            # Find the last occurrence of [BEGIN] to avoid matching it in the prompt
            start = response.rfind("[BEGIN]") + len("[BEGIN]")
            end = None
            for marker in end_markers:
                # Search for the marker after the [BEGIN] we found
                marker_idx = response.find(marker, start)
                if marker_idx != -1:
                    end = marker_idx
                    metadata["end_marker"] = marker
                    break
            if end is not None:
                cleaned = response[start:end]
                metadata["extraction"] = "markers"
        except ValueError:
            pass
    else:
        # Fallback to first code block fenced by ```
        fence_matches = list(re.finditer(r"```(?:verilog)?\s*", response))
        if len(fence_matches) >= 2:
            start = fence_matches[0].end()
            end = fence_matches[1].start()
            cleaned = response[start:end]
            metadata["extraction"] = "triple_backtick"
        else:
            metadata["extraction"] = "raw"

    lines = cleaned.splitlines()
    stripped_lines = [line.rstrip() for line in lines]
    verilog = "\n".join(stripped_lines).strip()
    return verilog + ("\n" if not verilog.endswith("\n") else ""), metadata


def write_generate_log(
    path: Path,
    *,
    problem: str,
    system_msg: str,
    user_prompt: str,
    response_text: str,
    prompt_tokens: int,
    response_tokens: int,
    generation_seconds: float,
) -> None:
    with path.open("w", encoding="utf-8") as log:
        log.write(f"problem         = {problem}\n")
        log.write(f"prompt_tokens   = {prompt_tokens}\n")
        log.write(f"resp_tokens     = {response_tokens}\n")
        log.write(f"total_tokens    = {prompt_tokens + response_tokens}\n")
        log.write("total_cost      = 0.0\n")
        log.write(f"generation_time = {generation_seconds:.2f} s\n\n")

        log.write("System Message\n")
        log.write("-" * 74 + "\n")
        log.write(system_msg.strip() + "\n\n")

        log.write("Prompt\n")
        log.write("-" * 74 + "\n")
        log.write(user_prompt.rstrip() + "\n\n")

        log.write("Response\n")
        log.write("-" * 74 + "\n")
        log.write(response_text.strip() + "\n")


def run_iverilog_flow(
    *,
    sample_sv: Path,
    test_sv: Path,
    ref_sv: Path,
    binary_path: Path,
    log_path: Path,
    timeout_s: int,
) -> Tuple[int, Optional[int]]:
    """Compile and simulate a generated Verilog sample."""
    compile_cmd = [
        "iverilog",
        "-Wall",
        "-Winfloop",
        "-Wno-timescale",
        "-g2012",
        "-s",
        "tb",
        "-o",
        str(binary_path),
        str(sample_sv),
        str(test_sv),
        str(ref_sv),
    ]

    with log_path.open("w", encoding="utf-8") as log_file:
        compile_proc = subprocess.run(
            compile_cmd,
            stdout=log_file,
            stderr=log_file,
            check=False,
        )

    runtime_rc: Optional[int] = None
    if compile_proc.returncode == 0:
        with log_path.open("a", encoding="utf-8") as log_file:
            try:
                runtime_proc = subprocess.run(
                    [str(binary_path)],
                    stdout=log_file,
                    stderr=log_file,
                    check=False,
                    timeout=timeout_s,
                )
                runtime_rc = runtime_proc.returncode
            except subprocess.TimeoutExpired:
                log_file.write("TIMEOUT\n")
                runtime_rc = 124
    return compile_proc.returncode, runtime_rc


def determine_pass_status(log_path: Path) -> bool:
    """Return True when the simulation log indicates success."""
    if not log_path.exists():
        return False
    text = log_path.read_text()
    if "TIMEOUT" in text:
        return False
    mismatch_match = re.search(r"Mismatches:\s+0\b", text)
    return mismatch_match is not None


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a LoRA fine-tuned model on VerilogEval problems.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--base-model-id",
        default="Qwen/Qwen2.5-Coder-3B",
        help="Base HuggingFace model identifier (e.g., Qwen/Qwen2.5-Coder-3B).",
    )
    parser.add_argument(
        "--lora-adapter-path",
        type=Path,
        default=None,
        help="Path to the LoRA adapter directory. If not provided, uses base model without fine-tuning.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable trust_remote_code=True when loading the model/tokenizer.",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="device_map argument passed to AutoModelForCausalLM (e.g. 'cpu', 'auto').",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "float32", "bfloat16", "float16"],
        default="auto",
        help="Preferred model precision.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Maximum number of new tokens to generate per problem.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature; 0 disables sampling.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p nucleus sampling parameter when sampling is enabled.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Path to the dataset directory (containing *_prompt.txt files). "
        "Defaults to verilog-eval/dataset_spec-to-rtl relative to script.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where outputs/logs will be written. "
        "Defaults to /matx/u/ethanboneh/runs/<sanitized-model>_<timestamp>.",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=0,
        help="Number of in-context examples to prepend (0-4 supported).",
    )
    parser.add_argument(
        "--include-rules",
        action="store_true",
        help="Include the Verilog coding rules suffix in the prompt.",
    )
    parser.add_argument(
        "--problems",
        nargs="*",
        default=(),
        help="Subset of problems to evaluate (default: all problems).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of problems (processed in listed order).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete any existing output directory before running.",
    )
    parser.add_argument(
        "--sim-timeout",
        type=int,
        default=30,
        help="Simulation timeout in seconds for each problem.",
    )
    parser.add_argument(
        "--save-raw-response",
        action="store_true",
        help="Save the raw LLM response alongside the extracted Verilog.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for sampling (set on torch generator).",
    )

    return parser.parse_args(argv)


def resolve_output_dir(base: Optional[Path], model_id: str, adapter_path: Optional[Path]) -> Path:
    if base is not None:
        return base.resolve()
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    if adapter_path is not None:
        # Ensure adapter_path is a Path object
        adapter_path = Path(adapter_path) if not isinstance(adapter_path, Path) else adapter_path
        adapter_name = adapter_path.name
        run_name = f"{sanitize_name(model_id)}_{adapter_name}_{timestamp}"
    else:
        run_name = f"{sanitize_name(model_id)}_base_{timestamp}"
    # Use /matx/u/ethanboneh/runs as default location
    runs_dir = Path("/matx/u/ethanboneh/runs")
    runs_dir.mkdir(parents=True, exist_ok=True)
    return (runs_dir / run_name).resolve()


def dtype_from_string(torch_module, dtype_str: str):
    if dtype_str == "auto":
        return None
    return getattr(torch_module, dtype_str)


def maybe_apply_chat_template(
    tokenizer,
    *,
    system_msg: str,
    user_prompt: str,
    use_chat_template: bool = True,
):
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        conversation = []
        if system_msg:
            conversation.append({"role": "system", "content": system_msg})
        conversation.append({"role": "user", "content": user_prompt})
        chat_text = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )
        return tokenizer(chat_text, return_tensors="pt")
    # fallback to concatenation
    prompt = system_msg.strip() + "\n\n" + user_prompt if system_msg else user_prompt
    return tokenizer(prompt, return_tensors="pt")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    ensure_tools_available()

    # Resolve dataset directory
    if args.dataset_dir is None:
        # Try to find verilog-eval relative to this script
        script_dir = Path(__file__).resolve().parent
        verilog_eval_dir = script_dir.parent / "verilog-eval"
        if verilog_eval_dir.exists():
            dataset_dir = verilog_eval_dir / "dataset_spec-to-rtl"
        else:
            # Try absolute path
            dataset_dir = Path("/afs/cs.stanford.edu/u/ethanboneh/rtl_smith/verilog-eval/dataset_spec-to-rtl")
    else:
        dataset_dir = args.dataset_dir.resolve()

    if not dataset_dir.exists():
        raise SystemExit(f"ERROR: dataset directory not found: {dataset_dir}")

    scripts_dir = dataset_dir.parent / "scripts"
    if not scripts_dir.exists():
        scripts_dir = Path(__file__).resolve().parent.parent / "verilog-eval" / "scripts"

    problems = read_problem_list(dataset_dir, args.problems)
    if args.limit is not None:
        problems = problems[: args.limit]

    output_dir = resolve_output_dir(args.output_dir, args.base_model_id, args.lora_adapter_path)
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    examples_prefix = load_examples(
        task="spec-to-rtl",
        examples=args.shots,
        scripts_dir=scripts_dir,
    )

    dtype = dtype_from_string(torch, args.dtype)
    tokenizer_kwargs = {"trust_remote_code": args.trust_remote_code}
    model_kwargs = {
        "device_map": args.device_map,
        "trust_remote_code": args.trust_remote_code,
    }
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype

    print(f"Loading base model: {args.base_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_id,
        **tokenizer_kwargs,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_id,
        **model_kwargs,
    )

    if args.lora_adapter_path is not None:
        print(f"Loading LoRA adapter from: {args.lora_adapter_path}")
        model = PeftModel.from_pretrained(base_model, args.lora_adapter_path)
    else:
        print("Using base model without fine-tuning (no LoRA adapter)")
        model = base_model
    model.eval()

    torch.manual_seed(args.seed)

    summary: List[ProblemResult] = []
    successes: List[str] = []

    for index, problem in enumerate(problems, start=1):
        problem_dir = output_dir / problem
        problem_dir.mkdir(parents=True, exist_ok=True)

        prompt_path = dataset_dir / f"{problem}_prompt.txt"
        test_path = dataset_dir / f"{problem}_test.sv"
        ref_path = dataset_dir / f"{problem}_ref.sv"

        missing_files = [
            str(path)
            for path in (prompt_path, test_path, ref_path)
            if not path.exists()
        ]
        if missing_files:
            print(f"[{index}/{len(problems)}] {problem}: missing files {missing_files}")
            continue

        problem_text = prompt_path.read_text()
        system_msg, user_prompt = build_prompt(
            problem_text=problem_text,
            include_rules=args.include_rules,
            include_examples=examples_prefix,
        )

        start_time = time.time()
        encoded_inputs = maybe_apply_chat_template(
            tokenizer,
            system_msg=system_msg,
            user_prompt=user_prompt,
            use_chat_template=True,
        )
        encoded_inputs = {k: v.to(model.device) for k, v in encoded_inputs.items()}
        prompt_tokens = int(encoded_inputs["input_ids"].shape[-1])

        generation_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "pad_token_id": getattr(tokenizer, "pad_token_id", tokenizer.eos_token_id),
        }
        if args.temperature > 0.0:
            generation_kwargs.update(
                {
                    "do_sample": True,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                }
            )
        else:
            generation_kwargs["do_sample"] = False

        with torch.inference_mode():
            generated = model.generate(
                **encoded_inputs,
                **generation_kwargs,
            )

        total_tokens = int(generated.shape[-1])
        response_tokens = total_tokens - prompt_tokens
        generated_ids = generated[:, encoded_inputs["input_ids"].shape[-1]:]
        response_text = tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True,
        )
        generation_time = time.time() - start_time

        sample_idx = 1
        sample_suffix = f"_sample{sample_idx:02d}"
        verilog_path = problem_dir / f"{problem}{sample_suffix}.sv"
        generate_log = problem_dir / f"{problem}{sample_suffix}-sv-generate.log"
        compile_log = problem_dir / f"{problem}{sample_suffix}-sv-iv-test.log"
        raw_response_path = problem_dir / f"{problem}{sample_suffix}_raw.txt"

        verilog_code, extraction_meta = extract_code_from_response(response_text)
        verilog_path.write_text(verilog_code, encoding="utf-8")

        if args.save_raw_response:
            raw_response_path.write_text(response_text, encoding="utf-8")
        else:
            raw_response_path = raw_response_path  # still store path in result for parity

        write_generate_log(
            generate_log,
            problem=problem,
            system_msg=system_msg,
            user_prompt=user_prompt,
            response_text=response_text,
            prompt_tokens=prompt_tokens,
            response_tokens=response_tokens,
            generation_seconds=generation_time,
        )

        binary_path = problem_dir / f"{problem}{sample_suffix}"
        compile_rc, runtime_rc = run_iverilog_flow(
            sample_sv=verilog_path,
            test_sv=test_path,
            ref_sv=ref_path,
            binary_path=binary_path,
            log_path=compile_log,
            timeout_s=args.sim_timeout,
        )
        if binary_path.exists():
            binary_path.unlink()

        passed = (
            compile_rc == 0
            and runtime_rc == 0
            and determine_pass_status(compile_log)
        )

        status = "pass" if passed else "fail"
        if passed:
            successes.append(problem)

        metadata = {
            "extraction_strategy": extraction_meta.get("extraction"),
            "generation_time_s": generation_time,
            "compile_returncode": compile_rc,
            "runtime_returncode": runtime_rc,
        }

        summary.append(
            ProblemResult(
                problem=problem,
                status=status,
                compile_returncode=compile_rc,
                runtime_returncode=runtime_rc,
                duration_s=generation_time,
                prompt_tokens=prompt_tokens,
                response_tokens=response_tokens,
                verilog_path=verilog_path,
                generate_log=generate_log,
                compile_log=compile_log,
                raw_response_path=raw_response_path,
                metadata=metadata,
            )
        )

        print(
            f"[{index}/{len(problems)}] {problem}: {status.upper()} "
            f"(tokens: {prompt_tokens}->{response_tokens}, "
            f"time: {generation_time:.1f}s)"
        )

    summary_data = {
        "base_model_id": args.base_model_id,
        "lora_adapter_path": str(args.lora_adapter_path) if args.lora_adapter_path else None,
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "shots": args.shots,
        "include_rules": args.include_rules,
        "problems_total": len(summary),
        "problems_passed": len(successes),
        "passed_problems": successes,
        "results": [
            {
                "problem": result.problem,
                "status": result.status,
                "compile_returncode": result.compile_returncode,
                "runtime_returncode": result.runtime_returncode,
                "prompt_tokens": result.prompt_tokens,
                "response_tokens": result.response_tokens,
                "generation_time_s": result.duration_s,
                "verilog_path": str(result.verilog_path),
                "generate_log": str(result.generate_log),
                "compile_log": str(result.compile_log),
                "raw_response_path": str(result.raw_response_path),
                "metadata": result.metadata,
            }
            for result in summary
        ],
    }

    summary_json_path = output_dir / "summary.json"
    summary_json_path.write_text(
        json.dumps(summary_data, indent=2),
        encoding="utf-8",
    )

    print("")
    print("Evaluation complete.")
    print(f"Total problems evaluated : {len(summary)}")
    print(f"Problems passed          : {len(successes)}")
    print(f"Pass rate                : {len(successes)/len(summary)*100:.1f}%" if summary else "N/A")
    print(f"Results directory        : {output_dir}")
    if successes:
        joined = ", ".join(successes)
        print(f"Passed problems          : {joined}")
    else:
        print("Passed problems          : (none)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

