import io
import json
import urllib.error
import urllib.request

from clinic_retrieval.llm import Llm


class _Reply(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _answers(payload):
    return lambda request, timeout=None: _Reply(json.dumps(payload).encode())


def _times_out(request, timeout=None):
    raise TimeoutError()


def test_a_reply_after_a_timeout_marks_the_model_reachable_again(monkeypatch):
    # Seen live: one request timed out while the model was busy, and /health
    # reported it unreachable for the rest of the process.
    llm = Llm(base_url="http://model.test/v1")
    monkeypatch.setattr(urllib.request, "urlopen", _times_out)
    assert llm.chat("system", "user") is None
    assert llm.reachable() is False

    monkeypatch.setattr(urllib.request, "urlopen", _answers(
        {"choices": [{"message": {"content": "{}"}}]}))
    assert llm.chat("system", "user") == "{}"
    assert llm.reachable() is True


def test_a_refused_connection_marks_the_model_unreachable(monkeypatch):
    llm = Llm(base_url="http://model.test/v1")
    llm._reachable = True

    def refused(request, timeout=None):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", refused)
    assert llm.chat("system", "user") is None
    assert llm.reachable() is False
