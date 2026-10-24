"""The model call, made twice per question: once to read the question, once to
write the answer.

Both go to the same model. A clinic box holds one set of weights because the
second set is a memory decision rather than a modelling one, and the target box
has 16GB. The endpoint is OpenAI-compatible, which is what Ollama
exposes on a laptop and what vLLM exposes on the clinic GPUs, so deployment is
a URL and not a code path.

Nothing configured, or the server not answering, is not a licence to guess. The
caller gets None and the orchestrator escalates to a person.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request

# Defaults point at a local Ollama, so pulling a model is the only step needed
# to make this live. Override for vLLM or any hosted OpenAI-compatible server.
BASE_URL = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:11434/v1")
MODEL = os.environ.get("LLM_MODEL", "qwen2.5:3b-instruct")
API_KEY = os.environ.get("LLM_API_KEY", "")
TIMEOUT = float(os.environ.get("LLM_TIMEOUT", "30"))

# Small instruct models wrap JSON in prose often enough to plan for it.
_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


class Llm:
    """One model, two prompts. No retries: a clinic waiting on a second attempt
    is worse off than a clinic told to ask the supervisor."""

    def __init__(self, base_url: str = BASE_URL, model: str = MODEL,
                 api_key: str = API_KEY, timeout: float = TIMEOUT):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self._reachable: bool | None = None

    @property
    def name(self) -> str:
        return f"{self.model} at {self.base_url}"

    def reachable(self) -> bool:
        """Whether this model is actually being served, probed once per process.
        The answer changes a startup banner and a health field, and it separates
        an outage from a confused model in the escalation reason; it is never a
        decision inside a request."""
        if self._reachable is None:
            self._reachable = self._probe()
        return self._reachable

    def chat(self, system: str, user: str, max_tokens: int = 300) -> str | None:
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": 0,          # the same question gets the same answer
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }).encode()
        raw = self._post("/chat/completions", body)
        if raw is None:
            return None
        try:
            return raw["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            return None

    def json(self, system: str, user: str, max_tokens: int = 300) -> dict | None:
        """A parsed object, or None. A model that cannot produce one is treated
        the same as a model that is down."""
        text = self.chat(system, user, max_tokens)
        if not text:
            return None
        found = _OBJECT.search(text)
        if found is None:
            return None
        try:
            parsed = json.loads(found.group(0))
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None

    # ---- transport -------------------------------------------------------

    def _post(self, path: str, body: bytes) -> dict | None:
        request = urllib.request.Request(
            self.base_url + path, data=body, method="POST",
            headers=self._headers() | {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read())
        except (urllib.error.URLError, OSError, json.JSONDecodeError, TimeoutError):
            self._reachable = False
            return None
        # A reply is proof it is served. Without this, one timeout while the
        # model was busy left /health reporting it down for the whole process.
        self._reachable = True
        return payload

    def _probe(self) -> bool:
        """An endpoint that answers is not the same as a model that is loaded.
        Ollama serves /v1/models as `{"object": "list", "data": null}` before
        anything is pulled, so a 200 proves nothing and the configured model
        has to appear in the list."""
        request = urllib.request.Request(self.base_url + "/models",
                                         headers=self._headers())
        try:
            with urllib.request.urlopen(request, timeout=min(self.timeout, 5)) as r:
                payload = json.loads(r.read())
        except (urllib.error.URLError, OSError, TimeoutError,
                json.JSONDecodeError):
            return False
        if not isinstance(payload, dict) or "data" not in payload:
            return True          # a server that does not list; assume it serves
        listed = payload["data"] or []
        served = {str(m.get("id", "")) for m in listed if isinstance(m, dict)}
        return any(name == self.model or name.startswith(self.model)
                   or self.model.startswith(name) for name in served)

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
