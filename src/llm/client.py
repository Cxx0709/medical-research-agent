"""
OpenAI-Compatible LLM Client
Supports OpenAI, DeepSeek, Gemini (OpenAI-compatible endpoints), Qwen, etc.
Provides graceful fallback when API key is missing or calls fail.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class LLMClient:
    """Lightweight, resilient client for OpenAI-compatible Chat Completion APIs."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 15.0,
    ):
        self.api_key = api_key or os.getenv("LLM_API_KEY", "").strip()
        raw_base_url = base_url or os.getenv("LLM_BASE_URL", "https://api.openai.com/v1").strip()
        self.base_url = raw_base_url.rstrip("/")
        self.model = model or os.getenv("LLM_MODEL", "gpt-4o-mini").strip()
        self.timeout = timeout

    @property
    def is_available(self) -> bool:
        """Checks if an API key is provided and non-empty."""
        return bool(self.api_key)

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 1500,
    ) -> Optional[str]:
        """Sends chat completion request to OpenAI-compatible endpoint.

        Returns string response on success, or None if unavailable or on error.
        """
        if not self.is_available:
            return None

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            choices = data.get("choices", [])
            if choices and "message" in choices[0]:
                content = choices[0]["message"].get("content", "")
                return content.strip()
            return None
        except Exception as e:
            logger.warning(f"LLM API request to {url} failed: {e}. Falling back to rule-based engine.")
            return None
