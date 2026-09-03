"""
Provider abstraction for MILLICOUNT.

Each provider exposes:
  - a display name
  - the env var / secrets key it looks for
  - a default model string
  - an `analyze(image_bytes, api_key, model) -> MillipedeAnalysis` function

Add a new provider by writing an `analyze_*` function with that signature
and registering it in PROVIDERS at the bottom of this file.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Callable

from pydantic import BaseModel


class MillipedeAnalysis(BaseModel):
    millipede_detected: bool
    segment_count: int
    leg_count: int
    confidence: float
    visible_segments: int
    explanation: str


ANALYSIS_PROMPT = """
You are the visual analysis engine for an application called MILLICOUNT.

Your task is to analyze an image of a millipede and estimate:

1. The number of body segments.
2. The total number of legs.

IMPORTANT INSTRUCTIONS:

- First determine whether a millipede is actually visible.
- Count the body segments from the head toward the posterior end.
- A body segment is a repeated structural unit along the main body.
- Do NOT count individual ridges, shadows, highlights, or texture lines as separate segments.
- Do NOT count legs as body segments.
- Do NOT count antennae as body segments.
- If the body is curved, mentally follow the body from head to tail.
- If some segments are partially hidden, use the visible structure and make the best biological estimate.
- Count visible legs when possible, but do NOT simply multiply blindly.
- The leg count should be a biological estimate based on the millipede's segment structure.
- If the image is too blurry, cropped, occluded, or ambiguous, lower the confidence.
- Never invent certainty. If you are unsure, report a lower confidence value.

For segment counting, carefully inspect the entire body rather than estimating from the apparent body length.

Return:
- whether a millipede was detected
- estimated total body segment count
- estimated total leg count
- confidence from 0 to 1
- number of clearly visible segments
- a brief explanation of how the estimate was obtained
"""

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "millipede_detected": {"type": "boolean"},
        "segment_count": {"type": "integer"},
        "leg_count": {"type": "integer"},
        "confidence": {"type": "number"},
        "visible_segments": {"type": "integer"},
        "explanation": {"type": "string"},
    },
    "required": [
        "millipede_detected",
        "segment_count",
        "leg_count",
        "confidence",
        "visible_segments",
        "explanation",
    ],
    "additionalProperties": False,
}


# ============================================================
# GEMINI (Google)
# ============================================================

def analyze_gemini(image_bytes: bytes, api_key: str, model: str) -> MillipedeAnalysis:
    try:
        from google import genai
        from google.genai import types
    except ImportError as e:
        raise RuntimeError(
            "The 'google-genai' package is not installed. Run: "
            "pip install google-genai"
        ) from e

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_text(text=ANALYSIS_PROMPT),
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=MillipedeAnalysis,
        ),
    )

    if not response.text:
        raise RuntimeError("Gemini returned an empty response.")

    try:
        data = json.loads(response.text)
        return MillipedeAnalysis(**data)
    except Exception as e:
        raise RuntimeError(f"Could not parse Gemini response:\n{response.text}") from e


# ============================================================
# OPENAI
# ============================================================

def analyze_openai(image_bytes: bytes, api_key: str, model: str) -> MillipedeAnalysis:
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError(
            "The 'openai' package is not installed. Run: pip install openai"
        ) from e

    client = OpenAI(api_key=api_key)
    b64 = base64.b64encode(image_bytes).decode("utf-8")

    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": ANALYSIS_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    },
                ],
            }
        ],
        response_format=MillipedeAnalysis,
    )

    parsed = completion.choices[0].message.parsed
    if parsed is None:
        refusal = completion.choices[0].message.refusal
        raise RuntimeError(f"OpenAI did not return structured output: {refusal}")

    return parsed


# ============================================================
# ANTHROPIC (Claude)
# ============================================================

def analyze_claude(image_bytes: bytes, api_key: str, model: str) -> MillipedeAnalysis:
    try:
        import anthropic
    except ImportError as e:
        raise RuntimeError(
            "The 'anthropic' package is not installed. Run: pip install anthropic"
        ) from e

    client = anthropic.Anthropic(api_key=api_key)
    b64 = base64.b64encode(image_bytes).decode("utf-8")

    tool = {
        "name": "report_millipede_analysis",
        "description": "Report the millipede segment/leg analysis results.",
        "input_schema": JSON_SCHEMA,
    }

    response = client.messages.create(
        model=model,
        max_tokens=1024,
        tools=[tool],
        tool_choice={"type": "tool", "name": "report_millipede_analysis"},
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": b64,
                        },
                    },
                    {"type": "text", "text": ANALYSIS_PROMPT},
                ],
            }
        ],
    )

    tool_use_block = next(
        (b for b in response.content if b.type == "tool_use"), None
    )
    if tool_use_block is None:
        raise RuntimeError("Claude did not return a structured tool call.")

    try:
        return MillipedeAnalysis(**tool_use_block.input)
    except Exception as e:
        raise RuntimeError(
            f"Could not parse Claude response:\n{tool_use_block.input}"
        ) from e


# ============================================================
# PROVIDER REGISTRY
# ============================================================

@dataclass(frozen=True)
class Provider:
    key: str                 # internal id, also used for session/secrets/env lookups
    label: str                # shown in the UI
    default_model: str
    secrets_key: str          # key looked up in st.secrets / env vars
    env_fallback: str | None  # extra env var name to also check (e.g. GOOGLE_API_KEY)
    analyze: Callable[[bytes, str, str], MillipedeAnalysis]
    signup_url: str


PROVIDERS: dict[str, Provider] = {
    "gemini": Provider(
        key="gemini",
        label="Google Gemini",
        default_model="gemini-3.7-flash",
        secrets_key="GEMINI_API_KEY",
        env_fallback="GOOGLE_API_KEY",
        analyze=analyze_gemini,
        signup_url="https://aistudio.google.com/apikey",
    ),
    "openai": Provider(
        key="openai",
        label="OpenAI",
        default_model="gpt-4o",
        secrets_key="OPENAI_API_KEY",
        env_fallback=None,
        analyze=analyze_openai,
        signup_url="https://platform.openai.com/api-keys",
    ),
    "claude": Provider(
        key="claude",
        label="Anthropic Claude",
        default_model="claude-sonnet-5",
        secrets_key="ANTHROPIC_API_KEY",
        env_fallback=None,
        analyze=analyze_claude,
        signup_url="https://console.anthropic.com/settings/keys",
    ),
}
