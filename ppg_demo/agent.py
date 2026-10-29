"""Optional zero-paid-API AI agent using local Ollama function calls.

Runs with a locally installed Ollama model that supports tool calling.
The deterministic fallback prints an explanation of computed results; it is
explicitly NOT an LLM agent.
"""
from __future__ import annotations

import json
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any

from .analysis import analyze_segment, validate_subject

SYSTEM = """You are a research signal-analysis assistant, not a medical expert.
For questions about recordings, call analyze_segment first. Report only numbers
from its tool result. Separate observations from hypotheses and uncertainty.
Do not diagnose health conditions or suggest an individual has a disease.
The reference values are evaluation labels and are never estimator inputs.
Explain that motion-frequency overlap and spectral concentration are heuristics,
not calibrated confidence scores. If something is unavailable, say so.
Do not claim that a single segment establishes population-level performance.
"""

TOOL = {
    "type": "function",
    "function": {
        "name": "analyze_segment",
        "description": "Analyze a PPG-DaLiA wrist-recording segment and compare PPG estimates to ECG-derived reference labels (comparison only).",
        "parameters": {
            "type": "object",
            "properties": {
                "subject": {"type": "string", "description": "Participant ID S1 to S15"},
                "start_seconds": {"type": "number", "description": "Start of segment in seconds"},
                "duration_seconds": {"type": "number", "description": "Duration of segment, between 8 and 300 seconds"},
            },
            "required": ["subject", "start_seconds", "duration_seconds"],
        },
    },
}


def template_explanation(result: dict[str, Any]) -> str:
    """Evidence-based, deterministic response when no LLM API is available."""
    raw = result["raw_periodogram"]["mae_bpm"]
    filt = result["filtered_welch"]["mae_bpm"]
    comparison = "Evaluation unavailable for the selected windows."
    if raw is not None and filt is not None:
        if filt < raw:
            comparison = f"Filtered spectral estimation had lower MAE ({filt:.2f} vs {raw:.2f} BPM)."
        elif filt > raw:
            comparison = f"Filtering did not improve MAE ({filt:.2f} vs {raw:.2f} BPM)."
        else:
            comparison = f"The methods tied at {filt:.2f} BPM MAE."
    overlap = result["motion_frequency_overlap_percent"]
    overlap_text = (f"ACC and PPG dominant frequencies overlapped in {overlap:.1f}% of assessable windows."
                    if overlap is not None else "Motion-frequency overlap could not be assessed.")
    return (
        f"Observations: Analyzed {result['complete_windows']} complete 8-second windows for "
        f"{result['subject']}, from {result['selected_start_seconds']:g} to "
        f"{result['selected_end_seconds']:g} seconds. "
        f"Mean PPG-derived estimate: {result['mean_estimate_bpm']} BPM; "
        f"mean reference: {result['mean_reference_bpm']} BPM. "
        f"{comparison} {overlap_text}\n\n"
        "Uncertainty: This spectral estimator can lock onto movement or harmonics "
        "instead of pulse. The overlap statistic is a heuristic rather than proof "
        "of motion contamination. Results are for one selected segment, with "
        "overlapping windows, not a population-level accuracy claim. "
        "This is not a medical diagnosis."
    )


def _post_ollama(messages: list[dict], model: str,
                 *, tools: list[dict] | None = None, endpoint: str) -> dict:
    payload: dict[str, Any] = {"model": model, "messages": messages,
                               "stream": False, "options": {"temperature": 0}}
    if tools:
        payload["tools"] = tools
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(endpoint, data=data,
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=180) as res:
        return json.load(res)


def ask_agent(question: str, data_root: str | Path, *, subject: str = "S1",
              start_seconds: float = 0, duration_seconds: float = 120,
              model: str = "qwen3:4b",
              endpoint: str = "http://127.0.0.1:11434/api/chat") -> tuple[str, dict]:
    """Real LLM tool-call loop: user -> model -> function -> model.

    Runs one allowlisted tool call; LLM never chooses a file path or runs code.
    Returns both generated answer and exact tool result for independent checking.
    """
    subject = validate_subject(subject)
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content":
         f"Segment defaults: {subject}, {start_seconds} seconds start, "
         f"{duration_seconds} seconds duration. {question} "
         "Use the analysis tool before explaining the results."},
    ]
    first = _post_ollama(messages, model, tools=[TOOL], endpoint=endpoint)
    message = first.get("message", {})
    calls = message.get("tool_calls") or []
    if not calls:
        raise RuntimeError("Model did not invoke the analysis tool; choose a tool-calling model")
    call = calls[0]
    if call.get("function", {}).get("name") != "analyze_segment":
        raise ValueError("Unrecognized tool requested")
    args = call["function"].get("arguments", {})
    if isinstance(args, str):
        args = json.loads(args)
    # Never allow arbitrary paths, non-allowlisted subjects, or huge workloads.
    args_subject = validate_subject(str(args.get("subject", subject)))
    args_start = float(args.get("start_seconds", start_seconds))
    args_duration = float(args.get("duration_seconds", duration_seconds))
    if args_duration > 300 or args_duration < 8 or args_start < 0:
        raise ValueError("Refused out-of-range tool arguments")
    result = analyze_segment(data_root, args_subject, args_start, args_duration)
    messages += [message, {"role": "tool", "tool_name": "analyze_segment",
                           "content": json.dumps(result, allow_nan=False)}]
    answer = _post_ollama(messages, model, endpoint=endpoint)
    reply = answer.get("message", {}).get("content", "").strip()
    if not reply:
        raise RuntimeError("Model returned an empty explanation")
    return reply, result
