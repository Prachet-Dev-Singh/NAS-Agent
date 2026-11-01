"""
gateway.py
----------
LLM Gateway for NAS-Agent (Blueprint Section 7 & 11 Phase 6).

Key Capabilities:
1. Multi-provider support: Mock (tests/offline), OpenRouter, OpenAI, and Gemini.
2. Disk response cache keyed on SHA-256 of (system_prompt, user_prompt, model).
3. Rate-limiting resilience with exponential backoff, jitter, and pause-and-resume on 429.
4. Strict Pydantic schema validation for Proposal (MacroProposal | MicroProposal).
5. Self-healing repair retry loop (up to 3 attempts with validation error feedback).
6. Deterministic fallback proposal generation on exhaustion.
7. Audit logging to SQLite llm_calls table.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import os
import random
import re
import time
import uuid
from typing import Any, Dict, Optional, Tuple, Union

import requests
from pydantic import ValidationError

from nas_agent.calibration.budget import sample_random_macro_proposal
from nas_agent.memory.store import CandidateStore
from nas_agent.state import MacroProposal, MicroProposal, Proposal


logger = logging.getLogger(__name__)


def extract_json_from_text(text: str) -> Dict[str, Any]:
    """Extract JSON object from LLM output, stripping markdown code fences."""
    text = text.strip()
    # Check for markdown fenced code blocks
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if fence_match:
        text = fence_match.group(1).strip()
    else:
        # Fallback: find first '{' and last '}'
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]

    return json.loads(text)


class LLMGateway:
    """Robust gateway for LLM proposal queries with cache, repair, and rate-limit backoff."""

    def __init__(
        self,
        provider: str = "mock",
        model: str = "mock-model",
        api_key: Optional[str] = None,
        cache_dir: str = "experiments/llm_cache",
        max_repair_attempts: int = 3,
        temperature: float = 0.7,
        max_retries: int = 5,
        base_backoff_sec: float = 2.0,
    ) -> None:
        self.provider = provider.lower()
        self.model = model
        self.api_key = api_key or os.environ.get(f"{self.provider.upper()}_API_KEY", "")
        self.cache_dir = cache_dir
        self.max_repair_attempts = max_repair_attempts
        self.temperature = temperature
        self.max_retries = max_retries
        self.base_backoff_sec = base_backoff_sec
        self._mock_rng = random.Random(42)

        os.makedirs(self.cache_dir, exist_ok=True)

    def _compute_cache_key(self, system_prompt: str, user_prompt: str) -> str:
        payload = f"{self.provider}||{self.model}||{system_prompt}||{user_prompt}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _check_cache(self, cache_key: str) -> Optional[Tuple[str, int, int]]:
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return data["response"], data.get("tokens_in", 0), data.get("tokens_out", 0)
            except Exception as e:
                logger.warning(f"Error reading LLM cache {cache_file}: {e}")
        return None

    def _save_cache(self, cache_key: str, response: str, tokens_in: int, tokens_out: int) -> None:
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "response": response,
                        "tokens_in": tokens_in,
                        "tokens_out": tokens_out,
                        "model": self.model,
                        "provider": self.provider,
                        "ts": datetime.datetime.utcnow().isoformat(),
                    },
                    f,
                    indent=2,
                )
        except Exception as e:
            logger.warning(f"Error writing LLM cache {cache_file}: {e}")

    def _call_provider_raw(self, system_prompt: str, user_prompt: str) -> Tuple[str, int, int]:
        """Execute raw HTTP request to provider with exponential backoff on rate limits."""
        if self.provider == "mock":
            return self._mock_response(user_prompt), 100, 150

        headers: Dict[str, str] = {"Content-Type": "application/json"}
        payload: Dict[str, Any] = {}
        endpoint = ""

        if self.provider in ("openrouter", "openai", "groq"):
            headers["Authorization"] = f"Bearer {self.api_key}"
            if self.provider == "openrouter":
                endpoint = "https://openrouter.ai/api/v1/chat/completions"
                headers["HTTP-Referer"] = "https://github.com/nas-agent"
                headers["X-Title"] = "NAS-Agent"
            elif self.provider == "groq":
                endpoint = "https://api.groq.com/openai/v1/chat/completions"
            else:
                endpoint = "https://api.openai.com/v1/chat/completions"

            payload = {
                "model": self.model,
                "temperature": self.temperature,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            }

        elif self.provider == "gemini":
            endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
            payload = {
                "systemInstruction": {"parts": [{"text": system_prompt}]},
                "contents": [{"parts": [{"text": user_prompt}]}],
                "generationConfig": {"temperature": self.temperature},
            }
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

        # Retry loop with exponential backoff
        for attempt in range(self.max_retries + 1):
            try:
                resp = requests.post(endpoint, headers=headers, json=payload, timeout=60.0)
                if resp.status_code == 200:
                    data = resp.json()
                    if self.provider in ("openrouter", "openai", "groq"):
                        content = data["choices"][0]["message"]["content"]
                        usage = data.get("usage", {})
                        t_in = usage.get("prompt_tokens", 0)
                        t_out = usage.get("completion_tokens", 0)
                        return content, t_in, t_out
                    elif self.provider == "gemini":
                        content = data["candidates"][0]["content"]["parts"][0]["text"]
                        usage = data.get("usageMetadata", {})
                        t_in = usage.get("promptTokenCount", 0)
                        t_out = usage.get("candidatesTokenCount", 0)
                        return content, t_in, t_out

                elif resp.status_code == 429:
                    retry_after = resp.headers.get("Retry-After")
                    sleep_time = float(retry_after) if retry_after else (
                        min(60.0, self.base_backoff_sec * (2 ** attempt) + random.uniform(0.5, 2.0))
                    )
                    logger.warning(f"Rate limited (429). Pausing for {sleep_time:.1f}s before retry {attempt + 1}...")
                    time.sleep(sleep_time)
                else:
                    logger.warning(f"Provider {self.provider} error {resp.status_code}: {resp.text}")
                    if attempt < self.max_retries:
                        time.sleep(min(30.0, self.base_backoff_sec * (2 ** attempt)))
                    else:
                        resp.raise_for_status()

            except (requests.exceptions.RequestException, TimeoutError) as e:
                logger.warning(f"Request exception on attempt {attempt}: {e}")
                if attempt < self.max_retries:
                    time.sleep(min(30.0, self.base_backoff_sec * (2 ** attempt)))
                else:
                    raise

        raise RuntimeError(f"Failed to query {self.provider} after {self.max_retries} attempts.")

    def _mock_response(self, user_prompt: str) -> str:
        """Deterministic mock response for testing and offline execution."""
        # If user prompt has near-feasible parent mentioned, propose micro-surgery
        if "Available for Micro Surgery" in user_prompt:
            match = re.search(r"\|\s*([a-zA-Z0-9_\-]+)\s*\|\s*[0-9\.]+\s*\|\s*[0-9\.]+\s*\|", user_prompt)
            if match and "No checkpointed candidates" not in user_prompt:
                parent_id = match.group(1)
                return json.dumps({
                    "kind": "micro",
                    "parent_id": parent_id,
                    "op": {
                        "technique": "svd_lowrank",
                        "rank_ratio": 0.5,
                        "which": "both"
                    },
                    "target_stages": [0],
                    "distill": {"enabled": False},
                    "rationale": f"Mock micro surgery low-rank on parent {parent_id}"
                })

        # Default mock macro proposal
        macro = sample_random_macro_proposal(self._mock_rng)
        return macro.model_dump_json()

    def query_proposal(
        self,
        system_prompt: str,
        user_prompt: str,
        run_id: str = "run_default",
        store: Optional[CandidateStore] = None,
    ) -> Tuple[Proposal, Dict[str, Any]]:
        """Query LLM with caching, validation, and self-healing repair attempts.

        Returns
        -------
        Tuple[Proposal, audit_dict]
        """
        call_id = f"call_{uuid.uuid4().hex[:10]}"
        cache_key = self._compute_cache_key(system_prompt, user_prompt)
        cached = self._check_cache(cache_key)

        cache_hit = False
        tokens_in, tokens_out = 0, 0
        raw_text = ""

        if cached is not None:
            raw_text, tokens_in, tokens_out = cached
            cache_hit = True
        else:
            raw_text, tokens_in, tokens_out = self._call_provider_raw(system_prompt, user_prompt)
            self._save_cache(cache_key, raw_text, tokens_in, tokens_out)

        repair_count = 0
        current_text = raw_text

        # Validation & repair loop
        for attempt in range(self.max_repair_attempts + 1):
            try:
                data = extract_json_from_text(current_text)
                kind = data.get("kind", "macro")
                if kind == "micro":
                    proposal: Proposal = MicroProposal.model_validate(data)
                else:
                    proposal = MacroProposal.model_validate(data)

                # Successfully validated proposal
                audit = {
                    "call_id": call_id,
                    "run_id": run_id,
                    "provider": self.provider,
                    "model_id": self.model,
                    "tokens_in": tokens_in,
                    "tokens_out": tokens_out,
                    "cache_hit": cache_hit,
                    "repair_count": repair_count,
                    "fallback": False,
                    "ts": datetime.datetime.utcnow().isoformat(),
                }
                if store is not None:
                    store.log_llm_call(
                        call_id=call_id,
                        run_id=run_id,
                        provider=self.provider,
                        model_id=self.model,
                        tokens_in=tokens_in,
                        tokens_out=tokens_out,
                        cache_hit=cache_hit,
                        repair_count=repair_count,
                        ts=audit["ts"],
                    )
                return proposal, audit

            except (json.JSONDecodeError, ValidationError, ValueError, KeyError) as e:
                logger.warning(f"Proposal validation failed on attempt {attempt}: {e}")
                if attempt < self.max_repair_attempts:
                    repair_count += 1
                    repair_prompt = (
                        f"{user_prompt}\n\n"
                        f"CRITICAL ERROR: Your previous response failed schema validation with error:\n{str(e)}\n"
                        "Please correct the error and output valid JSON conforming strictly to the Proposal schema."
                    )
                    try:
                        current_text, t_in, t_out = self._call_provider_raw(system_prompt, repair_prompt)
                        tokens_in += t_in
                        tokens_out += t_out
                    except Exception as call_err:
                        logger.error(f"Failed repair query: {call_err}")
                        break
                else:
                    break

        # Fallback to random valid macro proposal if all repair attempts exhausted
        logger.error(f"Exhausted {self.max_repair_attempts} repair attempts. Emitting fallback proposal.")
        fallback_proposal = sample_random_macro_proposal(random.Random())
        fallback_proposal.rationale = "Fallback proposal generated after LLM schema validation failure."

        audit = {
            "call_id": call_id,
            "run_id": run_id,
            "provider": self.provider,
            "model_id": self.model,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "cache_hit": cache_hit,
            "repair_count": repair_count,
            "fallback": True,
            "ts": datetime.datetime.utcnow().isoformat(),
        }
        if store is not None:
            store.log_llm_call(
                call_id=call_id,
                run_id=run_id,
                provider=self.provider,
                model_id=self.model,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cache_hit=cache_hit,
                repair_count=repair_count,
                ts=audit["ts"],
            )

        return fallback_proposal, audit
