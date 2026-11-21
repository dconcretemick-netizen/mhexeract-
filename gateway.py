#!/usr/bin/env python3
"""
MHexeract Multi-Model AI Gateway
=============================

One entry point for OpenAI, Anthropic, Google Vertex AI and DeepSeek models,
built on LiteLLM (direct provider SDK calls) with OpenRouter as a second
transport for every model.

Features
--------
* Automatic fallback routing   - ordered chain of models x routes (direct, then
                                 OpenRouter), retries with backoff for transient
                                 errors, circuit breakers, quarantine.
* Token usage tracking         - every attempt is written to a SQLite ledger
                                 (tokens, latency, status, errors).
* Efficiency                   - task-aware tier selection (lighter models for
                                 simple work), response cache, compression.
* Balanced self-learning       - Thompson-sampling router that learns each
                                 model's quality, reliability, latency and effort
                                 per task type, balancing exploitation with a
                                 controlled amount of exploration.
* Automated process improvement- an optimizer that periodically reviews the
                                 ledger and tunes the routing policy (tier per
                                 task, route order, output-token estimates,
                                 retries, quarantines) with guardrails, a change
                                 log, and automatic rollback when quality drops.

Usage
-----
    python gateway.py chat "Explain CAP theorem" [--strategy balanced|efficient|quality]
    python gateway.py feedback <request_id> 0.9
    python gateway.py improve [--dry-run]
    python gateway.py report [--days 7]
    python gateway.py models
    python gateway.py simulate --n 300          # offline demo, no API keys needed

Set GATEWAY_MOCK=1 to run without API keys (LiteLLM mock responses).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import random
import re
import sqlite3
import statistics
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger("mhexeract.gateway")

# --------------------------------------------------------------------------- #
# LiteLLM, or a stdlib-only shim for constrained devices (Android / Termux)
# --------------------------------------------------------------------------- #
# LiteLLM pulls in native wheels (tokenizers, tiktoken, pydantic-core) that can
# be slow or impossible to build on Android. In LITE mode the gateway uses only
# the Python standard library and sends every request through OpenRouter's
# OpenAI-compatible endpoint, which still reaches OpenAI, Anthropic, Google and
# DeepSeek models. Force it with GATEWAY_LITE=1.
LITE = os.getenv("GATEWAY_LITE", "0") == "1"
if not LITE:
    try:
        import litellm  # type: ignore

        litellm.suppress_debug_info = True
    except Exception:  # noqa: BLE001
        LITE = True
if LITE:
    import types
    import urllib.error
    import urllib.request

    class _LLMError(Exception):
        def __init__(self, message: str = "", llm_provider: str = "", model: str = "", **_: Any):
            super().__init__(message)
            self.message, self.llm_provider, self.model = message, llm_provider, model

    _names = ["RateLimitError", "Timeout", "APIConnectionError", "ServiceUnavailableError",
              "InternalServerError", "AuthenticationError", "NotFoundError", "ContextWindowExceededError",
              "ContentPolicyViolationError", "BadRequestError", "PermissionDeniedError"]
    litellm = types.SimpleNamespace(**{n: type(n, (_LLMError,), {}) for n in _names})  # type: ignore
    litellm.model_cost = {}  # context-window lookups only

    class _Obj(dict):
        __getattr__ = dict.get

    def _wrap(d: Any) -> Any:
        if isinstance(d, dict):
            return _Obj({k: _wrap(v) for k, v in d.items()})
        if isinstance(d, list):
            return [_wrap(x) for x in d]
        return d

    def _model_response(model: str = "", choices: Any = None, usage: Any = None, **_: Any) -> Any:
        for c in choices or []:
            c.setdefault("message", {}).setdefault("content", "")
        return _wrap({"model": model, "choices": choices or [], "usage": usage or {}, "_hidden_params": {}})

    def _token_counter(model: str = "", messages: Any = None, **_: Any) -> int:
        chars = sum(len(str(m.get("content", ""))) for m in messages or [])
        return chars // 4 + 4 * len(messages or [])

    def _lite_target(model: str) -> Optional[tuple[str, str, str, str]]:
        """(provider, base_url, env_key, wire_model) for routes LITE mode can call over plain HTTPS."""
        if model.startswith("openrouter/"):
            return "openrouter", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", model.split("/", 1)[1]
        if model.startswith("deepseek/"):
            return "deepseek", "https://api.deepseek.com", "DEEPSEEK_API_KEY", model.split("/", 1)[1]
        if model.startswith("openai/") or "/" not in model:
            return "openai", "https://api.openai.com/v1", "OPENAI_API_KEY", model.split("/", 1)[-1]
        return None

    def _completion(model: str, messages: list, timeout: float = 60, **kw: Any) -> Any:
        target = _lite_target(model)
        if target is None:
            raise litellm.AuthenticationError(f"LITE mode cannot call {model} directly; use its OpenRouter route",
                                              "lite", model)
        provider, base, env_key, wire_model = target
        body = {"model": wire_model, "messages": messages,
                **{k: v for k, v in kw.items() if k in ("max_tokens", "temperature", "top_p", "response_format",
                                                        "stop", "seed", "tools", "tool_choice")}}
        if provider == "openai" and "max_tokens" in body:  # current OpenAI models use max_completion_tokens
            body["max_completion_tokens"] = body.pop("max_tokens")
        headers = {"Authorization": f"Bearer {os.getenv(env_key, '')}", "Content-Type": "application/json"}
        if provider == "openrouter":
            headers["X-Title"] = "MHexeract Gateway"
        req = urllib.request.Request(f"{base}/chat/completions", data=json.dumps(body).encode(), headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.loads(r.read())
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")[:500]
            cls = {400: "BadRequestError", 401: "AuthenticationError", 402: "RateLimitError",
                   403: "PermissionDeniedError", 404: "NotFoundError", 408: "Timeout", 413: "ContextWindowExceededError",
                   429: "RateLimitError", 500: "InternalServerError", 502: "ServiceUnavailableError",
                   503: "ServiceUnavailableError"}.get(e.code, "APIConnectionError")
            if e.code == 400 and "context" in msg.lower():
                cls = "ContextWindowExceededError"
            raise getattr(litellm, cls)(msg, provider, model) from None
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise litellm.APIConnectionError(str(e), provider, model) from None
        if "error" in data:
            raise litellm.ServiceUnavailableError(str(data["error"])[:500], provider, model)
        return _wrap(data)

    litellm.ModelResponse = _model_response
    litellm.token_counter = _token_counter
    litellm.completion = _completion

TIERS = ["economy", "standard", "premium"]
TIER_RANK = {t: i for i, t in enumerate(TIERS)}
TIER_EFFORT = {"economy": 1.0, "standard": 3.0, "premium": 8.0}  # relative compute weight, not money
TASK_TYPES = ["simple", "extraction", "summarization", "creative", "code", "reasoning"]

PROVIDER_ENV = {
    "openai": ["OPENAI_API_KEY"],
    "anthropic": ["ANTHROPIC_API_KEY"],
    "vertex_ai": ["VERTEXAI_PROJECT"],  # plus ADC / GOOGLE_APPLICATION_CREDENTIALS
    "deepseek": ["DEEPSEEK_API_KEY"],
    "openrouter": ["OPENROUTER_API_KEY"],
}

TRANSIENT_ERRORS = (
    litellm.RateLimitError,
    litellm.Timeout,
    litellm.APIConnectionError,
    litellm.ServiceUnavailableError,
    litellm.InternalServerError,
)
# Errors where retrying the same route is pointless; move to the next route.
SKIP_ERRORS = (
    litellm.AuthenticationError,
    litellm.NotFoundError,
    litellm.ContextWindowExceededError,
    litellm.ContentPolicyViolationError,
    litellm.BadRequestError,
    litellm.PermissionDeniedError,
)


# --------------------------------------------------------------------------- #
# Exceptions and data classes
# --------------------------------------------------------------------------- #
class GatewayError(Exception):
    pass


class AllRoutesFailed(GatewayError):
    def __init__(self, attempts: list[dict]):
        self.attempts = attempts
        summary = "; ".join(f"{a['model']}: {a['error']}" for a in attempts[-5:])
        super().__init__(f"All routes failed ({len(attempts)} attempts). Last: {summary}")


@dataclass
class Route:
    model: str  # LiteLLM model id
    via: str  # "direct" | "openrouter"
    max_input_tokens: int

    @property
    def transport_provider(self) -> str:
        return "openrouter" if self.via == "openrouter" else ""


@dataclass
class ModelSpec:
    alias: str
    provider: str
    tier: str
    quality: int
    routes: list[Route]

    @property
    def effort(self) -> float:
        """Relative compute weight per token (unitless): lighter tiers do the same work with less."""
        return TIER_EFFORT[self.tier]


@dataclass
class GatewayResponse:
    request_id: str
    text: str
    model_alias: str
    model: str
    via: str
    task_type: str
    tier_policy: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int
    attempts: int
    cached: bool = False
    exploratory: bool = False
    auto_quality: float = 0.0
    finish_reason: str = ""
    compression: Optional[dict] = None
    council: Optional[dict] = None
    raw: Any = field(default=None, repr=False)

    def summary(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k not in ("raw", "text")}


# --------------------------------------------------------------------------- #
# Route helpers
# --------------------------------------------------------------------------- #
def _context_window(model: str) -> Optional[int]:
    """Max input tokens from LiteLLM's model map (tries with and without provider prefix)."""
    table = getattr(litellm, "model_cost", {}) or {}
    for key in (model, model.split("/", 1)[-1]):
        info = table.get(key) or {}
        if info.get("max_input_tokens"):
            return int(info["max_input_tokens"])
    return None


def build_route(cfg: dict) -> Route:
    return Route(
        model=cfg["model"],
        via=cfg.get("via", "direct"),
        max_input_tokens=int(cfg.get("max_input_tokens") or _context_window(cfg["model"]) or 128_000),
    )


# --------------------------------------------------------------------------- #
# Task classification and automatic quality evaluation
# --------------------------------------------------------------------------- #
_TASK_PATTERNS = {
    "code": r"\b(code|function|bug|debug|python|javascript|typescript|sql|regex|refactor|api|class|compile|stack ?trace|unit test)\b|```",
    "reasoning": r"\b(prove|proof|derive|step[- ]by[- ]step|analy[sz]e|trade-?offs?|strategy|why does|evaluate|plan|architecture|optimi[sz]e|math|calculate)\b",
    "extraction": r"\b(extract|parse|json|fields?|table|classify|label|categori[sz]e|entities)\b",
    "summarization": r"\b(summari[sz]e|summary|tl;?dr|key points|condense|recap)\b",
    "creative": r"\b(story|poem|slogan|creative|brainstorm|tagline|write a (blog|post|song)|marketing copy)\b",
}
_REFUSAL = re.compile(r"^(i'?m sorry|i cannot|i can'?t help|as an ai)", re.I)


def classify_task(messages: list[dict]) -> str:
    text = " ".join(str(m.get("content", "")) for m in messages if m.get("role") != "system").lower()
    scores = {t: len(re.findall(p, text)) for t, p in _TASK_PATTERNS.items()}
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return "reasoning" if len(text) > 4000 else "simple"
    return best


def auto_evaluate(text: str, finish_reason: str, task_type: str, wants_json: bool) -> float:
    """Cheap, model-free quality signal in [0, 1]. Explicit feedback overrides it."""
    if not text or not text.strip():
        return 0.0
    if _REFUSAL.match(text.strip()):
        return 0.3
    score = 0.75
    if finish_reason == "length":
        score = 0.4
    if wants_json or task_type == "extraction" and text.strip().startswith(("{", "[")):
        try:
            json.loads(text.strip().strip("`").removeprefix("json"))
            score = max(score, 0.85)
        except Exception:
            score = min(score, 0.35) if wants_json else score
    if task_type in ("code", "reasoning") and len(text) < 40:
        score = min(score, 0.5)
    return score


# --------------------------------------------------------------------------- #
# Storage: usage ledger, feedback, learned policy, improvement log
# --------------------------------------------------------------------------- #
SCHEMA = """
CREATE TABLE IF NOT EXISTS attempts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL, day TEXT, request_id TEXT, alias TEXT, provider TEXT, tier TEXT,
    model TEXT, via TEXT, task_type TEXT, status TEXT, error_type TEXT, error TEXT,
    prompt_tokens INTEGER DEFAULT 0, completion_tokens INTEGER DEFAULT 0,
    cost_usd REAL DEFAULT 0, baseline_cost_usd REAL DEFAULT 0, latency_ms INTEGER DEFAULT 0,
    retry_index INTEGER DEFAULT 0, exploratory INTEGER DEFAULT 0, cached INTEGER DEFAULT 0,
    auto_quality REAL, finish_reason TEXT,
    original_tokens INTEGER DEFAULT 0, compressed INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS attempts_daily (
    day TEXT, ts REAL, alias TEXT, provider TEXT, tier TEXT, via TEXT, task_type TEXT, status TEXT,
    n INTEGER, cached_n INTEGER, prompt_tokens INTEGER, completion_tokens INTEGER, cost_usd REAL,
    baseline_cost_usd REAL, latency_sum REAL, quality_sum REAL, quality_n INTEGER,
    original_tokens INTEGER, compressed_n INTEGER,
    PRIMARY KEY (day, alias, via, task_type, status)
);
CREATE INDEX IF NOT EXISTS ix_attempts_alias ON attempts(alias, task_type, ts);
CREATE INDEX IF NOT EXISTS ix_attempts_req ON attempts(request_id);
CREATE INDEX IF NOT EXISTS ix_attempts_day ON attempts(day);
CREATE TABLE IF NOT EXISTS feedback (
    request_id TEXT PRIMARY KEY, score REAL, source TEXT, note TEXT, ts REAL
);
CREATE TABLE IF NOT EXISTS policy (
    key TEXT PRIMARY KEY, value TEXT, updated REAL
);
CREATE TABLE IF NOT EXISTS improvements (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, kind TEXT, target TEXT,
    old_value TEXT, new_value TEXT, reason TEXT, baseline_quality REAL,
    baseline_cost REAL, baseline_n INTEGER, status TEXT DEFAULT 'active'
);
"""


USAGE_VIEW = """
CREATE TEMP VIEW IF NOT EXISTS usage_all AS
  SELECT day, ts, alias, provider, tier, via, task_type, status, 1 AS n, cached AS cached_n,
         prompt_tokens, completion_tokens, cost_usd, baseline_cost_usd, latency_ms AS latency_sum,
         original_tokens, compressed AS compressed_n
  FROM attempts
  UNION ALL
  SELECT day, ts, alias, provider, tier, via, task_type, status, n, cached_n, prompt_tokens,
         completion_tokens, cost_usd, baseline_cost_usd, latency_sum, original_tokens, compressed_n
  FROM attempts_daily;
"""


class Store:
    def __init__(self, path: str, storage_cfg: Optional[dict] = None):
        self.path = path
        self.cfg = storage_cfg or {}
        self._lock = threading.Lock()
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        # forward-compatible migration for databases created by older versions
        cols = {r[1] for r in self.conn.execute("PRAGMA table_info(attempts)")}
        for col, ddl in (("original_tokens", "INTEGER DEFAULT 0"), ("compressed", "INTEGER DEFAULT 0")):
            if col not in cols:
                self.conn.execute(f"ALTER TABLE attempts ADD COLUMN {col} {ddl}")
        self.conn.executescript(USAGE_VIEW)
        try:  # WAL: readers (GUI) never block the writer (gateway); fewer fsyncs on phones
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA synchronous=NORMAL")
        except sqlite3.DatabaseError:
            pass
        self.conn.commit()
        self._policy_cache: dict[str, Any] = {}
        self._policy_ts = 0.0

    # -- self data compression (storage) ------------------------------------ #
    def size_mb(self) -> float:
        try:
            return os.path.getsize(self.path) / 1e6
        except OSError:
            return 0.0

    def storage_stats(self) -> dict:
        raw = self.query("SELECT COUNT(*) c, MIN(ts) t FROM attempts")[0]
        agg = self.query("SELECT COUNT(*) c, COALESCE(SUM(n),0) s FROM attempts_daily")[0]
        return {"db_mb": round(self.size_mb(), 3), "raw_rows": raw["c"], "aggregated_rows": agg["c"],
                "attempts_in_aggregates": agg["s"], "oldest_raw_ts": raw["t"],
                "max_db_mb": self.cfg.get("max_db_mb", 50), "max_raw_rows": self.cfg.get("max_raw_rows", 200_000),
                "retain_raw_days": self.cfg.get("retain_raw_days", 30)}

    def compaction_needed(self) -> bool:
        st = self.storage_stats()
        return st["db_mb"] > st["max_db_mb"] or st["raw_rows"] > st["max_raw_rows"]

    def compact(self, retain_days: Optional[float] = None, force: bool = False) -> dict:
        """
        Roll raw attempt rows older than `retain_days` into per-day aggregates, delete
        them, and VACUUM. Runs automatically only when the DB exceeds its size/row
        limits; if still over the limit it halves the retention window (floor 1 day).
        Totals, costs and token counts are preserved exactly; only per-request detail
        for old rows is dropped (the learner only needs the recent window anyway).
        """
        if not force and not self.compaction_needed():
            return {"compacted": False, "reason": "within limits", **self.storage_stats()}
        before = self.storage_stats()
        days = float(retain_days if retain_days is not None else self.cfg.get("retain_raw_days", 30))
        moved_total = 0
        while True:
            cutoff = time.time() - days * 86400
            with self._lock:
                self.conn.execute(
                    """INSERT INTO attempts_daily
                       SELECT a.day, MIN(a.ts), a.alias, MAX(a.provider), MAX(a.tier), a.via, a.task_type, a.status,
                              COUNT(*), SUM(a.cached), SUM(a.prompt_tokens), SUM(a.completion_tokens), SUM(a.cost_usd),
                              SUM(a.baseline_cost_usd), SUM(a.latency_ms),
                              SUM(COALESCE(f.score, a.auto_quality)), COUNT(COALESCE(f.score, a.auto_quality)),
                              SUM(a.original_tokens), SUM(a.compressed)
                       FROM attempts a LEFT JOIN feedback f ON f.request_id=a.request_id
                       WHERE a.ts < ? GROUP BY a.day, a.alias, a.via, a.task_type, a.status
                       ON CONFLICT(day, alias, via, task_type, status) DO UPDATE SET
                         n=n+excluded.n, cached_n=cached_n+excluded.cached_n,
                         prompt_tokens=prompt_tokens+excluded.prompt_tokens,
                         completion_tokens=completion_tokens+excluded.completion_tokens,
                         cost_usd=cost_usd+excluded.cost_usd, baseline_cost_usd=baseline_cost_usd+excluded.baseline_cost_usd,
                         latency_sum=latency_sum+excluded.latency_sum, quality_sum=quality_sum+excluded.quality_sum,
                         quality_n=quality_n+excluded.quality_n, original_tokens=original_tokens+excluded.original_tokens,
                         compressed_n=compressed_n+excluded.compressed_n""", (cutoff,))
                moved = self.conn.execute("DELETE FROM attempts WHERE ts < ?", (cutoff,)).rowcount
                self.conn.execute("DELETE FROM feedback WHERE request_id NOT IN (SELECT DISTINCT request_id FROM attempts)")
                self.conn.commit()
                self.conn.execute("VACUUM")
            moved_total += moved
            st = self.storage_stats()
            over = st["db_mb"] > st["max_db_mb"] or st["raw_rows"] > st["max_raw_rows"]
            if not over or days <= 1:
                break
            days = max(1.0, days / 2)
        after = self.storage_stats()
        return {"compacted": True, "rows_aggregated": moved_total, "retain_days_used": days,
                "db_mb_before": before["db_mb"], "db_mb_after": after["db_mb"], **after}

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self.conn.execute(sql, params)
            self.conn.commit()
            return cur

    def query(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self.conn.execute(sql, params).fetchall()

    def log_attempt(self, **row) -> None:
        now = time.time()
        row.setdefault("ts", now)
        row.setdefault("day", datetime.now(timezone.utc).strftime("%Y-%m-%d"))
        cols = ",".join(row)
        self.execute(f"INSERT INTO attempts ({cols}) VALUES ({','.join('?' * len(row))})", tuple(row.values()))

    def get_policy(self, key: str, default: Any = None) -> Any:
        # cached for 3s: policy changes rarely, and other processes (GUI/CLI) pick changes up quickly
        if time.time() - self._policy_ts > 3.0:
            rows = self.query("SELECT key, value FROM policy")
            self._policy_cache = {r["key"]: json.loads(r["value"]) for r in rows}
            self._policy_ts = time.time()
        v = self._policy_cache.get(key, default)
        return json.loads(json.dumps(v)) if isinstance(v, (dict, list)) else v

    def set_policy(self, key: str, value: Any) -> None:
        self.execute(
            "INSERT INTO policy(key,value,updated) VALUES(?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated=excluded.updated",
            (key, json.dumps(value), time.time()),
        )
        self._policy_cache[key] = json.loads(json.dumps(value))

    def request_count(self) -> int:
        raw = int(self.query("SELECT COUNT(DISTINCT request_id) c FROM attempts")[0]["c"])
        agg = int(self.query("SELECT COALESCE(SUM(n),0) c FROM attempts_daily WHERE status='ok'")[0]["c"])
        return raw + agg


# --------------------------------------------------------------------------- #
# Balanced self-learning router
# --------------------------------------------------------------------------- #
@dataclass
class ArmStats:
    n: int = 0  # successful responses observed
    q_sum: float = 0.0  # sum of quality scores
    ok: int = 0  # successful attempts
    fail: int = 0  # failed attempts
    latency_ms: float = 0.0
    out_tokens: float = 0.0


class Learner:
    """
    Learns per (model alias, task type):
      quality     ~ Beta(prior + sum q, prior + sum (1-q))   (feedback > judge > auto-eval)
      reliability ~ Beta(1 + ok, 1 + fail)
      latency, output tokens -> recency-windowed means
    The 'balanced' strategy draws Thompson samples and maximises
      utility = w_q * quality * reliability - w_e * effort_norm - w_l * latency_norm
    which naturally balances exploitation and exploration; an extra
    explore_rate lets the router probe one tier below the current policy so the
    optimizer has evidence for safe downgrades.
    """

    def __init__(self, store: Store, cfg: dict):
        self.store = store
        self.cfg = cfg
        self._cache: dict[tuple[str, str], ArmStats] = {}
        self._cache_ts = 0.0
        self.rng = random.Random()

    def invalidate(self) -> None:
        self._cache_ts = 0.0

    def _refresh(self) -> None:
        if time.time() - self._cache_ts < 5:
            return
        window = int(self.cfg.get("recency_window", 200))
        rows = self.store.query(
            """
            SELECT alias, task_type, status, latency_ms, completion_tokens, quality FROM (
              SELECT a.alias, a.task_type, a.status, a.latency_ms, a.completion_tokens,
                     COALESCE(f.score, a.auto_quality) AS quality,
                     ROW_NUMBER() OVER (PARTITION BY a.alias, a.task_type ORDER BY a.ts DESC) rn
              FROM attempts a LEFT JOIN feedback f ON f.request_id = a.request_id
              WHERE a.cached = 0
            ) WHERE rn <= ?
            """,
            (window,),
        )
        stats: dict[tuple[str, str], ArmStats] = {}
        lat: dict[tuple, list] = {}
        out: dict[tuple, list] = {}
        for r in rows:
            k = (r["alias"], r["task_type"])
            s = stats.setdefault(k, ArmStats())
            if r["status"] == "ok":
                s.ok += 1
                s.n += 1
                s.q_sum += float(r["quality"] if r["quality"] is not None else 0.75)
                lat.setdefault(k, []).append(r["latency_ms"] or 0)
                out.setdefault(k, []).append(r["completion_tokens"] or 0)
            else:
                s.fail += 1
        for k, s in stats.items():
            s.latency_ms = statistics.fmean(lat[k]) if lat.get(k) else 0.0
            s.out_tokens = statistics.fmean(out[k]) if out.get(k) else 0.0
        self._cache = stats
        self._cache_ts = time.time()

    def stats(self, alias: str, task_type: str) -> ArmStats:
        self._refresh()
        return self._cache.get((alias, task_type), ArmStats())

    def all_stats(self) -> dict[tuple[str, str], ArmStats]:
        self._refresh()
        return dict(self._cache)

    def expected_quality(self, spec: ModelSpec, task_type: str) -> float:
        s = self.stats(spec.alias, task_type)
        k = float(self.cfg.get("prior_strength", 4))
        prior = spec.quality / 10.0
        return (prior * k + s.q_sum) / (k + s.n)

    def sample_utility(self, spec: ModelSpec, task_type: str, est_effort: float,
                       max_effort: float, max_latency: float, target_quality: float = 0.72) -> float:
        """
        Unitless utility:
            w_q * q_eff * reliability  -  w_e * effort_norm  -  w_l * latency_norm
        effort is tokens x tier weight (lighter tiers do the same work with less compute).
        q_eff counts quality fully up to the 'good enough' target and at 25% above it,
        so the router uses heavy models where quality matters and light ones where it does not.
        q and reliability are Thompson samples, so uncertain models still get explored.
        """
        s = self.stats(spec.alias, task_type)
        k = float(self.cfg.get("prior_strength", 4))
        prior = spec.quality / 10.0
        a = 1 + prior * k + s.q_sum
        b = 1 + (1 - prior) * k + (s.n - s.q_sum)
        q = self.rng.betavariate(a, b)
        rel = self.rng.betavariate(1 + s.ok + 3, 1 + s.fail)  # optimistic prior
        w = self.cfg.get("weights", {})
        q_eff = min(q, target_quality) + 0.25 * max(0.0, q - target_quality)
        lat_norm = (s.latency_ms / max_latency) if max_latency > 0 and s.latency_ms else 0.3
        eff_norm = est_effort / max_effort if max_effort > 0 else 0.0
        return (w.get("quality", 1.0) * q_eff * rel
                - w.get("effort", 0.35) * eff_norm
                - w.get("latency", 0.15) * lat_norm)


# --------------------------------------------------------------------------- #
# Self data compression (context)
# --------------------------------------------------------------------------- #
class Compressor:
    """
    Shrinks a conversation only when needed: when it would overflow every
    eligible model's context window or pass
    the learned trigger threshold. Stages run lightest-first and stop as soon
    as the target is met:

      1. normalize  - collapse whitespace, blank-line runs, duplicate lines
      2. summarize  - fold older turns into one summary note (economy model;
                      skipped in mock mode) or drop them with a marker
      3. truncate   - head/tail trim the largest remaining messages

    System messages and the most recent turns are always preserved.
    """

    def __init__(self, gw: "Gateway"):
        self.gw = gw

    @property
    def cfg(self) -> dict:
        return self.gw.defaults.get("compression", {})

    def tokens(self, messages: list[dict]) -> int:
        return self.gw._count_tokens(messages)

    @staticmethod
    def _normalize(text: str) -> str:
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r" *\n *", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        out, prev = [], None
        for line in text.split("\n"):
            if line and line == prev:
                continue
            out.append(line)
            prev = line
        return "\n".join(out).strip()

    @staticmethod
    def _trim(text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        head = int(max_chars * 0.6)
        tail = max_chars - head
        cut = len(text) - head - tail
        return f"{text[:head]}\n[... {cut // 4} tokens trimmed by gateway ...]\n{text[-tail:]}"

    def _summarize(self, older: list[dict]) -> Optional[str]:
        if self.gw.mock or not self.cfg.get("summarize_with_model", True):
            return None
        transcript = "\n".join(f"{m['role']}: {m.get('content', '')}" for m in older)[:400_000]
        try:
            r = self.gw.chat(
                [{"role": "system", "content": "Summarize this conversation history for another AI. Keep facts, "
                                               "decisions, names, numbers, code identifiers and open questions. "
                                               "Be terse."},
                 {"role": "user", "content": transcript}],
                tier="economy", strategy="efficient", use_cache=False, task_type="summarization",
                _allow_compress=False, max_tokens=int(self.cfg.get("summary_max_tokens", 800)))
            return r.text
        except Exception as e:  # noqa: BLE001
            log.warning("summary compression failed, falling back to drop: %s", e)
            return None

    def compress(self, messages: list[dict], target_tokens: int) -> tuple[list[dict], dict]:
        original = self.tokens(messages)
        stages: list[str] = []
        msgs = [dict(m) for m in messages]

        # 1. normalize
        for m in msgs:
            if isinstance(m.get("content"), str):
                m["content"] = self._normalize(m["content"])
        stages.append("normalize")

        # 2. summarize / drop older turns
        keep = int(self.gw.compression_keep_turns())
        if self.tokens(msgs) > target_tokens:
            system = [m for m in msgs if m.get("role") == "system"]
            convo = [m for m in msgs if m.get("role") != "system"]
            if len(convo) > keep:
                older, recent = convo[:-keep], convo[-keep:]
                summary = self._summarize(older)
                note = (f"Summary of {len(older)} earlier messages:\n{summary}" if summary
                        else f"[{len(older)} earlier messages omitted by gateway to fit context]")
                msgs = system + [{"role": "system", "content": note}] + recent
                stages.append("summarize" if summary else "drop_older")

        # 2b. map-reduce: summarize a single oversized message chunk by chunk (real calls only)
        if self.tokens(msgs) > target_tokens and not self.gw.mock and self.cfg.get("summarize_with_model", True):
            idx = max(range(len(msgs)), key=lambda i: len(str(msgs[i].get("content", ""))))
            content = str(msgs[idx].get("content", ""))
            others = self.tokens(msgs) - len(content) // 4
            char_limit = max(4000, (target_tokens - others) * 4)
            if len(content) > char_limit:
                chunk_chars = int(self.cfg.get("map_chunk_tokens", 60_000)) * 4
                chunks = [content[i:i + chunk_chars] for i in range(0, len(content), chunk_chars)]
                max_chunks = int(self.cfg.get("map_max_chunks", 24))
                if len(chunks) > max_chunks:  # sample evenly so work stays bounded
                    step = len(chunks) / max_chunks
                    chunks = [chunks[int(i * step)] for i in range(max_chunks)]
                parts = []
                for n, ch in enumerate(chunks):
                    summ = self._summarize([{"role": "user", "content": ch}])
                    if summ is None:
                        parts = []
                        break
                    parts.append(f"[part {n + 1}/{len(chunks)}] {summ}")
                if parts:
                    msgs[idx]["content"] = ("[gateway map-reduce summary of a long input]\n" + "\n".join(parts))
                    stages.append("map_reduce")

        # 3. head/tail truncate the largest messages until under target
        guard = 0
        while self.tokens(msgs) > target_tokens and guard < 20:
            guard += 1
            idx = max(range(len(msgs)), key=lambda i: len(str(msgs[i].get("content", ""))))
            content = str(msgs[idx].get("content", ""))
            excess_chars = (self.tokens(msgs) - target_tokens) * 4 + 200
            new_len = max(400, len(content) - excess_chars)
            if new_len >= len(content):
                break
            msgs[idx]["content"] = self._trim(content, new_len)
            if "truncate" not in stages:
                stages.append("truncate")

        final = self.tokens(msgs)
        return msgs, {"original_tokens": original, "compressed_tokens": final,
                      "ratio": round(final / original, 3) if original else 1.0, "stages": stages}


# --------------------------------------------------------------------------- #
# Circuit breaker
# --------------------------------------------------------------------------- #
class CircuitBreaker:
    def __init__(self, threshold: int, cooldown_s: float):
        self.threshold = threshold
        self.cooldown = cooldown_s
        self.failures: dict[str, int] = {}
        self.opened_at: dict[str, float] = {}

    def available(self, key: str) -> bool:
        t = self.opened_at.get(key)
        if t is None:
            return True
        if time.time() - t > self.cooldown:  # half-open: allow one probe
            self.opened_at.pop(key, None)
            self.failures[key] = self.threshold - 1
            return True
        return False

    def success(self, key: str) -> None:
        self.failures[key] = 0
        self.opened_at.pop(key, None)

    def failure(self, key: str) -> None:
        self.failures[key] = self.failures.get(key, 0) + 1
        if self.failures[key] >= self.threshold:
            self.opened_at[key] = time.time()
            log.info("circuit opened for %s", key)


# --------------------------------------------------------------------------- #
# Automated process improvement
# --------------------------------------------------------------------------- #
class Optimizer:
    """
    Reviews the ledger and tunes the live routing policy. Every change is
    logged to `improvements` with a baseline; later runs evaluate active changes
    and roll them back automatically if quality dropped.

    Rules (all require min_samples of evidence and move one step at a time):
      1. tier_policy      lower a task's tier when the lighter tier meets the
                          quality target; raise it when the current tier misses.
      2. route_order      put OpenRouter first for a model when it is clearly
                          more reliable than the direct provider.
      3. output_tokens    calibrate expected output tokens per task (p75) so
                          token estimates and compression checks are accurate.
      4. retries          cut per-route retries when retries rarely succeed.
      5. quarantine       temporarily pull models whose reliability collapsed.
      6. rollback         revert active changes whose follow-up quality fell.
    """

    def __init__(self, gw: "Gateway"):
        self.gw = gw
        self.store = gw.store
        self.cfg = gw.defaults.get("improvement", {})

    # -- metrics helpers ---------------------------------------------------- #
    def _task_metrics(self, task_type: str, since: float = 0.0, tier: Optional[str] = None) -> tuple[float, float, int]:
        """(avg quality, avg tokens per response, n)"""
        sql = """SELECT COALESCE(f.score, a.auto_quality) q, a.prompt_tokens + a.completion_tokens c FROM attempts a
                 LEFT JOIN feedback f ON f.request_id=a.request_id
                 WHERE a.status='ok' AND a.cached=0 AND a.task_type=? AND a.ts>=?"""
        params: list = [task_type, since]
        if tier:
            sql += " AND a.tier=?"
            params.append(tier)
        rows = self.store.query(sql, tuple(params))
        if not rows:
            return 0.0, 0.0, 0
        qs = [float(r["q"] if r["q"] is not None else 0.75) for r in rows]
        cs = [float(r["c"] or 0) for r in rows]
        return statistics.fmean(qs), statistics.fmean(cs), len(rows)

    def _record(self, kind: str, target: str, old: Any, new: Any, reason: str,
                task_type: Optional[str], apply: bool, changes: list) -> None:
        bq, bc, bn = self._task_metrics(task_type) if task_type else (0.0, 0.0, 0)
        changes.append({"kind": kind, "target": target, "old": old, "new": new, "reason": reason})
        if apply:
            self.store.execute(
                "INSERT INTO improvements(ts,kind,target,old_value,new_value,reason,baseline_quality,baseline_cost,"
                "baseline_n,status) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (time.time(), kind, target, json.dumps(old), json.dumps(new), reason, bq, bc, bn,
                 "active" if kind == "tier_policy" else "applied"),
            )

    # -- main entry --------------------------------------------------------- #
    def run(self, apply: bool = True) -> list[dict]:
        changes: list[dict] = []
        min_n = int(self.cfg.get("min_samples", 20))
        target = float(self.cfg.get("target_quality", 0.72))
        margin = float(self.cfg.get("upgrade_margin", 0.15))

        # 6. rollback first, so we evaluate before stacking new changes
        self._rollbacks(apply, changes, min_n)

        # 1. tier policy per task type
        policy = self.gw.task_policy()
        recently_changed = {
            r["target"] for r in self.store.query(
                "SELECT target FROM improvements WHERE kind='tier_policy' AND status='active'")
        }
        for task in TASK_TYPES:
            if task in recently_changed:
                continue  # wait until the active change is evaluated
            cur = policy.get(task, "standard")
            rank = TIER_RANK[cur]
            q_cur, c_cur, n_cur = self._task_metrics(task, tier=cur)
            if rank > 0:
                lower = TIERS[rank - 1]
                q_low, c_low, n_low = self._task_metrics(task, tier=lower)
                if n_low >= min_n and q_low >= target and (n_cur == 0 or q_low >= q_cur - 0.05):
                    new = dict(policy, **{task: lower})
                    reason = (f"{lower} tier quality {q_low:.2f} (n={n_low}) meets target {target}; "
                              f"avg tokens {c_low:.0f} vs {c_cur:.0f}")
                    self._record("tier_policy", task, cur, lower, reason, task, apply, changes)
                    if apply:
                        self.store.set_policy("task_policy", new)
                    policy = new
                    continue
            if rank < len(TIERS) - 1 and n_cur >= min_n and q_cur < target - margin:
                higher = TIERS[rank + 1]
                new = dict(policy, **{task: higher})
                reason = f"{cur} tier quality {q_cur:.2f} (n={n_cur}) below target {target}-{margin}"
                self._record("tier_policy", task, cur, higher, reason, task, apply, changes)
                if apply:
                    self.store.set_policy("task_policy", new)
                policy = new

        # 2. route order per model
        order = self.store.get_policy("route_order", {})
        for spec in self.gw.models.values():
            rows = self.store.query(
                "SELECT via, SUM(status='ok') ok, COUNT(*) n FROM attempts WHERE alias=? AND cached=0 GROUP BY via",
                (spec.alias,))
            rel = {r["via"]: (r["ok"] / r["n"], r["n"]) for r in rows}
            if "direct" in rel and "openrouter" in rel:
                (rd, nd), (ro, no) = rel["direct"], rel["openrouter"]
                cur_first = order.get(spec.alias, "direct")
                if nd >= 10 and no >= 5:
                    want = "openrouter" if ro - rd > 0.2 else ("direct" if rd >= ro else cur_first)
                    if want != cur_first:
                        self._record("route_order", spec.alias, cur_first, want,
                                     f"reliability direct={rd:.2f} (n={nd}) openrouter={ro:.2f} (n={no})",
                                     None, apply, changes)
                        order[spec.alias] = want
        if apply:
            self.store.set_policy("route_order", order)

        # 3. expected output tokens per task (p75 of observed)
        est = self.store.get_policy("expected_output_tokens", {})
        for task in TASK_TYPES:
            rows = self.store.query(
                "SELECT completion_tokens t FROM attempts WHERE status='ok' AND cached=0 AND task_type=? "
                "ORDER BY ts DESC LIMIT 200", (task,))
            toks = sorted(int(r["t"] or 0) for r in rows)
            if len(toks) >= 10:
                p75 = toks[int(0.75 * (len(toks) - 1))]
                old = est.get(task, self.gw.defaults.get("expected_output_tokens", 600))
                if old and abs(p75 - old) / old > 0.25:
                    self._record("output_tokens", task, old, p75, f"p75 of last {len(toks)} responses", None, apply, changes)
                    est[task] = p75
        if apply:
            self.store.set_policy("expected_output_tokens", est)

        # 4. retries: if retries almost never rescue a route, stop wasting time on them
        r = self.store.query("SELECT SUM(status='ok') ok, COUNT(*) n FROM attempts WHERE retry_index>0")[0]
        retry_n, retry_ok = int(r["n"] or 0), int(r["ok"] or 0)
        cur_retries = int(self.store.get_policy("max_retries_per_route", self.gw.defaults.get("max_retries_per_route", 2)))
        if retry_n >= min_n:
            rate = retry_ok / retry_n
            new_r = cur_retries
            if rate < 0.2 and cur_retries > 0:
                new_r = cur_retries - 1
            elif rate > 0.6 and cur_retries < 3:
                new_r = cur_retries + 1
            if new_r != cur_retries:
                self._record("retries", "max_retries_per_route", cur_retries, new_r,
                             f"retry success rate {rate:.2f} over {retry_n} retries", None, apply, changes)
                if apply:
                    self.store.set_policy("max_retries_per_route", new_r)

        # 7. compression tuning: compress less if it hurts quality, more if it is harmless
        rows = self.store.query(
            """SELECT a.compressed c, AVG(COALESCE(f.score, a.auto_quality)) q, COUNT(*) n FROM attempts a
               LEFT JOIN feedback f ON f.request_id=a.request_id
               WHERE a.status='ok' AND a.cached=0 GROUP BY a.compressed""")
        cq = {r["c"]: (float(r["q"] or 0), int(r["n"])) for r in rows}
        if 1 in cq and 0 in cq and cq[1][1] >= min_n:
            (q_c, n_c), (q_u, _) = cq[1], cq[0]
            trig = self.gw.compression_trigger()
            keep = self.gw.compression_keep_turns()
            if q_c < q_u - 0.10:
                new_t, new_k = int(trig * 1.25), min(keep + 2, 20)
                why = f"compressed quality {q_c:.2f} < uncompressed {q_u:.2f} (n={n_c}); compress less"
            elif q_c >= q_u - 0.03:
                new_t, new_k = max(8_000, int(trig * 0.85)), keep
                why = f"compressed quality {q_c:.2f} ~ uncompressed {q_u:.2f} (n={n_c}); compress earlier to save tokens"
            else:
                new_t, new_k = trig, keep
                why = ""
            if new_t != trig or new_k != keep:
                self._record("compression", "trigger_tokens/keep_turns", [trig, keep], [new_t, new_k], why,
                             None, apply, changes)
                if apply:
                    self.store.set_policy("compression_trigger_tokens", new_t)
                    self.store.set_policy("compression_keep_turns", new_k)

        # 5. quarantine models whose recent reliability collapsed
        q_rel = float(self.cfg.get("quarantine_reliability", 0.5))
        q_secs = float(self.cfg.get("quarantine_s", 3600))
        quarantine = {k: v for k, v in self.store.get_policy("quarantine", {}).items() if v > time.time()}
        for spec in self.gw.models.values():
            rows = self.store.query(
                "SELECT status FROM attempts WHERE alias=? AND cached=0 ORDER BY ts DESC LIMIT 20", (spec.alias,))
            if len(rows) >= 20:
                rel = sum(r["status"] == "ok" for r in rows) / len(rows)
                if rel < q_rel and spec.alias not in quarantine:
                    self._record("quarantine", spec.alias, None, int(q_secs),
                                 f"reliability {rel:.2f} over last 20 attempts", None, apply, changes)
                    quarantine[spec.alias] = time.time() + q_secs
        if apply:
            self.store.set_policy("quarantine", quarantine)
            self.store.set_policy("last_improve_at_count", self.store.request_count())
            self.gw.learner.invalidate()
        return changes

    def _rollbacks(self, apply: bool, changes: list, min_n: int) -> None:
        drop = float(self.cfg.get("rollback_drop", 0.10))
        active = self.store.query("SELECT * FROM improvements WHERE kind='tier_policy' AND status='active'")
        for imp in active:
            task = imp["target"]
            q_after, c_after, n_after = self._task_metrics(task, since=imp["ts"])
            if n_after < min_n:
                continue
            if imp["baseline_n"] and q_after < imp["baseline_quality"] - drop:
                policy = self.gw.task_policy()
                old = json.loads(imp["old_value"])
                changes.append({"kind": "rollback", "target": task, "old": policy.get(task), "new": old,
                                "reason": f"quality fell {imp['baseline_quality']:.2f} -> {q_after:.2f}"})
                if apply:
                    policy[task] = old
                    self.store.set_policy("task_policy", policy)
                    self.store.execute("UPDATE improvements SET status='rolled_back' WHERE id=?", (imp["id"],))
            elif apply:
                self.store.execute("UPDATE improvements SET status='kept' WHERE id=?", (imp["id"],))
                changes.append({"kind": "confirm", "target": task, "old": None, "new": None,
                                "reason": f"kept: quality {q_after:.2f} vs baseline {imp['baseline_quality']:.2f}, "
                                          f"tokens {c_after:.0f} vs {imp['baseline_cost']:.0f}"})


# --------------------------------------------------------------------------- #
# Gateway
# --------------------------------------------------------------------------- #
class Gateway:
    def __init__(self, config_path: str = "models.json", db_path: str = "gateway_usage.db",
                 mock: Optional[bool] = None):
        cfg = json.loads(Path(config_path).read_text())
        self.defaults: dict = cfg.get("defaults", {})
        self._default_task_policy: dict = cfg.get("task_policy", {})
        self.models: dict[str, ModelSpec] = {}
        for m in cfg["models"]:
            self.models[m["alias"]] = ModelSpec(
                alias=m["alias"], provider=m["provider"], tier=m["tier"], quality=int(m.get("quality", 5)),
                routes=[build_route(r) for r in m["routes"]],
            )
        self.mock = bool(int(os.getenv("GATEWAY_MOCK", "0"))) if mock is None else mock
        self.mock_fail: set[str] = set(filter(None, os.getenv("GATEWAY_MOCK_FAIL", "").split(",")))
        self.mock_hook = None  # optional callable(route, messages) -> (text, finish_reason) for simulations
        self.store = Store(db_path, self.defaults.get("storage", {}))
        self.repo_registry = str(Path(db_path).resolve().parent / "repos.json")
        self.learner = Learner(self.store, self.defaults.get("learning", {}))
        self.breaker = CircuitBreaker(int(self.defaults.get("circuit_breaker_failures", 3)),
                                      float(self.defaults.get("circuit_breaker_cooldown_s", 120)))
        self.optimizer = Optimizer(self)
        self.compressor = Compressor(self)
        self._cache: dict[str, tuple[float, GatewayResponse]] = {}
        self.nodes_path = str(Path(db_path).resolve().parent / "nodes.json")
        self._node_info: dict[str, dict] = {}
        self.load_nodes()

    # -- discovered AI nodes -------------------------------------------------- #
    def load_nodes(self) -> int:
        """Add chat models from trusted, online nodes in nodes.json to the pool (replacing older node entries)."""
        d = load_nodes(self.nodes_path)
        models = {a: s for a, s in self.models.items() if not s.provider.startswith("node:")}
        info: dict[str, dict] = {}
        for n in d.get("nodes", []):
            if n.get("status") != "online" or not n.get("trusted"):
                continue
            info[n["id"]] = n
            for m in n.get("models", []):
                alias = f"{n['id']}:{m['id']}"
                if alias in models:
                    continue
                models[alias] = ModelSpec(alias=alias, provider=f"node:{n['id']}", tier=m.get("tier", "standard"),
                                          quality=int(m.get("quality", 5)),
                                          routes=[Route(model=f"node/{n['id']}/{m['id']}", via="direct",
                                                        max_input_tokens=int(m.get("ctx") or 32_000))])
        self.models, self._node_info = models, info
        return sum(1 for s in models.values() if s.provider.startswith("node:"))

    # -- policy accessors (learned values override config) ------------------ #
    def task_policy(self) -> dict:
        return dict(self._default_task_policy, **self.store.get_policy("task_policy", {}))

    def expected_output_tokens(self, task_type: str) -> int:
        learned = self.store.get_policy("expected_output_tokens", {})
        return int(learned.get(task_type, self.defaults.get("expected_output_tokens", 600)))

    def compression_trigger(self) -> int:
        return int(self.store.get_policy("compression_trigger_tokens",
                                         self.defaults.get("compression", {}).get("trigger_tokens", 120_000)))

    def compression_keep_turns(self) -> int:
        return int(self.store.get_policy("compression_keep_turns",
                                         self.defaults.get("compression", {}).get("keep_last_turns", 6)))

    def _compression_target(self, p_tok: int, task_type: str, tier: Optional[str],
                            models: Optional[list[str]]) -> Optional[int]:
        """Return a token target if compression is needed for this request, else None."""
        if not self.defaults.get("compression", {}).get("enabled", True):
            return None
        out_tok = self.expected_output_tokens(task_type)
        min_rank = TIER_RANK[tier or self.task_policy().get(task_type, "standard")]
        routes = [(s, r) for s in self.models.values() if (not models or s.alias in models)
                  and TIER_RANK[s.tier] >= min_rank for r in s.routes if self._route_available(s, r)]
        if not routes:
            return None
        targets = []
        trigger = self.compression_trigger()
        if p_tok > trigger:
            targets.append(int(trigger * 0.75))
        max_ctx = max(r.max_input_tokens for _, r in routes)
        if p_tok + out_tok > max_ctx * 0.95:
            targets.append(int(max_ctx * 0.9) - out_tok)
        targets = [t for t in targets if t > 0]
        return min(targets) if targets else None

    def max_retries(self) -> int:
        return int(self.store.get_policy("max_retries_per_route", self.defaults.get("max_retries_per_route", 2)))

    # -- availability ------------------------------------------------------- #
    def allow_paid(self) -> bool:
        """MHexeract is free by default: paid AI is used only if the owner turns it on."""
        env = os.getenv("MHEXERACT_ALLOW_PAID")
        if env in ("0", "1"):
            return env == "1"
        return bool(self.store.get_policy("allow_paid", not (self.defaults.get("cost") or {}).get("free_only", True)))

    def route_is_free(self, spec: ModelSpec, route: Route) -> bool:
        if route.model.startswith("node/"):
            n = self._node_info.get(route.model.split("/")[1], {})
            return n.get("kind", "local") != "cloud" or n.get("id") in FREE_NODE_IDS
        return route.via == "openrouter" and route.model.endswith(":free")

    def _route_available(self, spec: ModelSpec, route: Route) -> bool:
        if not self.mock and not self.route_is_free(spec, route) and not self.allow_paid():
            return False  # free-only mode (the default): never call anything that could cost money
        if route.model.startswith("node/"):
            n = self._node_info.get(route.model.split("/")[1])
            return bool(n) and (self.mock or not n.get("env") or bool(node_key(n["env"])))
        if LITE and route.via != "openrouter" and spec.provider not in ("openai", "deepseek"):
            return False  # LITE (Android) mode: only OpenAI-compatible HTTPS APIs can be called directly
        if self.mock:
            return True
        prov = "openrouter" if route.via == "openrouter" else spec.provider
        return all(os.getenv(k) for k in PROVIDER_ENV.get(prov, []))

    def _ordered_routes(self, spec: ModelSpec) -> list[Route]:
        first = self.store.get_policy("route_order", {}).get(spec.alias, "direct")
        return sorted(spec.routes, key=lambda r: 0 if r.via == first else 1)

    # -- token / effort estimation ---------------------------------------------------- #
    @staticmethod
    def _count_tokens(messages: list[dict]) -> int:
        try:
            return int(litellm.token_counter(model="gpt-4o", messages=messages))
        except Exception:
            return sum(len(str(m.get("content", ""))) for m in messages) // 4 + 4 * len(messages)

    @staticmethod
    def _effort(spec: ModelSpec, p_tok: int, c_tok: int) -> float:
        """Unitless compute effort: tokens weighted by model tier."""
        return (p_tok + c_tok) * spec.effort

    # -- candidate selection ------------------------------------------------ #
    def plan(self, messages: list[dict], strategy: str, task_type: str, tier: Optional[str],
             models: Optional[list[str]] = None) -> tuple[list[tuple[ModelSpec, Route, bool]], str, int, int]:
        p_tok = self._count_tokens(messages)
        out_tok = self.expected_output_tokens(task_type)
        policy_tier = tier or self.task_policy().get(task_type, "standard")
        min_rank = TIER_RANK[policy_tier]
        quarantine = {k for k, v in self.store.get_policy("quarantine", {}).items() if v > time.time()}

        lcfg = self.defaults.get("learning", {})
        explore_down = (strategy == "balanced" and tier is None and min_rank > 0
                        and self.learner.rng.random() < float(lcfg.get("explore_rate", 0.1)))

        pool: list[tuple[ModelSpec, float, bool]] = []
        for spec in self.models.values():
            if models and spec.alias not in models:
                continue
            if spec.alias in quarantine and not models:
                continue
            rank = TIER_RANK[spec.tier]
            exploratory = explore_down and rank == min_rank - 1
            if rank < min_rank and not exploratory:
                continue
            routes = [r for r in spec.routes if self._route_available(spec, r) and p_tok < r.max_input_tokens]
            if not routes:
                continue
            est = self._effort(spec, p_tok, out_tok)
            pool.append((spec, est, exploratory))

        if not pool:
            if not self.mock and not self.allow_paid():
                raise GatewayError(
                    "No free AI is set up yet. Add a free key (Google Gemini, Groq, Mistral, OpenRouter, SambaNova or "
                    "NVIDIA) in AI keys, or run a local model with Ollama or LM Studio. Paid AI is off.")
            raise GatewayError(
                f"No eligible models (tier>={policy_tier}). "
                "Check API keys in the AI keys menu or .env, or pass --tier / --models.")

        if strategy in ("efficient", "cost"):  # "cost" kept as a legacy alias
            pool.sort(key=lambda x: (x[1], -self.learner.expected_quality(x[0], task_type)))
        elif strategy == "quality":
            pool.sort(key=lambda x: (-self.learner.expected_quality(x[0], task_type), x[1]))
        else:  # balanced: Thompson sampling over learned quality, reliability, effort, latency
            max_e = max(e for _, e, _ in pool) or 1e-9
            max_l = max((self.learner.stats(s.alias, task_type).latency_ms for s, _, _ in pool), default=0) or 1.0
            tq = float(self.defaults.get("improvement", {}).get("target_quality", 0.72))
            scored = [(self.learner.sample_utility(s, task_type, e, max_e, max_l, tq), s, e, x) for s, e, x in pool]
            scored.sort(key=lambda t: -t[0])
            # an exploratory probe goes first so it actually gets traffic
            if explore_down:
                scored.sort(key=lambda t: 0 if t[3] else 1)
            pool = [(s, e, x) for _, s, e, x in scored]

        chain: list[tuple[ModelSpec, Route, bool]] = []
        for spec, _, exploratory in pool:
            for r in self._ordered_routes(spec):
                if self._route_available(spec, r) and p_tok < r.max_input_tokens:
                    chain.append((spec, r, exploratory))
        return chain, policy_tier, p_tok, out_tok

    # -- low-level call ----------------------------------------------------- #
    def _call(self, route: Route, messages: list[dict], **kw) -> Any:
        if self.mock:
            if route.model in self.mock_fail or (route.via == "direct" and route.model.split("/")[-1] in self.mock_fail):
                raise litellm.RateLimitError(message="mock rate limit", llm_provider=route.via, model=route.model)
            if self.mock_hook:
                text, finish = self.mock_hook(route, messages)
            else:
                text, finish = f"[mock:{route.model}] OK", "stop"
            p_tok = self._count_tokens(messages)
            c_tok = max(1, len(text) // 4)
            return litellm.ModelResponse(
                model=route.model,
                choices=[{"index": 0, "finish_reason": finish, "message": {"role": "assistant", "content": text}}],
                usage={"prompt_tokens": p_tok, "completion_tokens": c_tok, "total_tokens": p_tok + c_tok})
        if route.model.startswith("node/"):
            _, nid, wire = route.model.split("/", 2)
            n = self._node_info[nid]
            extra = {"X-Title": "MHexeract Gateway"} if nid == "openrouter" else None
            return node_completion(n["base_url"], node_key(n.get("env", "")), wire, messages,
                                   float(self.defaults.get("request_timeout_s", 60)) * (3 if n["kind"] != "cloud" else 1),
                                   nid, route.model, extra, **kw)
        params = dict(model=route.model, messages=messages,
                      timeout=float(self.defaults.get("request_timeout_s", 60)), **kw)
        if route.model.startswith("vertex_ai/"):
            params.setdefault("vertex_project", os.getenv("VERTEXAI_PROJECT"))
            params.setdefault("vertex_location", os.getenv("VERTEXAI_LOCATION", "global"))
        return litellm.completion(**params)

    # -- public API --------------------------------------------------------- #
    def chat(self, messages: list[dict] | str, *, strategy: Optional[str] = None, tier: Optional[str] = None,
             models: Optional[list[str]] = None,
             use_cache: bool = True, task_type: Optional[str] = None, _allow_compress: bool = True,
             **completion_kwargs) -> GatewayResponse:
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        strategy = strategy or self.defaults.get("strategy", "balanced")
        task_type = task_type or classify_task(messages)
        wants_json = (completion_kwargs.get("response_format") or {}).get("type") == "json_object"

        # cache: identical prompts are answered instantly
        key = hashlib.sha256(json.dumps([messages, tier, models, completion_kwargs], sort_keys=True,
                                        default=str).encode()).hexdigest()
        ttl = float(self.defaults.get("cache_ttl_s", 900))
        if use_cache and key in self._cache and time.time() - self._cache[key][0] < ttl:
            cached = self._cache[key][1]
            rid = uuid.uuid4().hex[:12]
            self.store.log_attempt(request_id=rid, alias=cached.model_alias, provider=self.models[cached.model_alias].provider,
                                   tier=self.models[cached.model_alias].tier, model=cached.model, via=cached.via,
                                   task_type=task_type, status="ok", cached=1)
            return GatewayResponse(**{**cached.__dict__, "request_id": rid, "cached": True, "latency_ms": 0})

        # self data compression: only when the request would not fit context/trigger
        comp_info: Optional[dict] = None
        if _allow_compress:
            target = self._compression_target(self._count_tokens(messages), task_type, tier, models)
            if target:
                messages, comp_info = self.compressor.compress(messages, target)
                log.info("compressed request %s -> %s tokens (%s)", comp_info["original_tokens"],
                         comp_info["compressed_tokens"], "+".join(comp_info["stages"]))
        orig_tok = comp_info["original_tokens"] if comp_info else 0

        try:
            chain, policy_tier, p_est, out_est = self.plan(messages, strategy, task_type, tier, models)
        except GatewayError:
            # availability-aware degradation: step down tiers rather than fail, unless a tier was forced
            chain = None
            if tier is None:
                start = TIER_RANK[self.task_policy().get(task_type, "standard")]
                for lower in reversed(TIERS[:start]):
                    try:
                        chain, policy_tier, p_est, out_est = self.plan(messages, strategy, task_type, lower,
                                                                       models)
                        log.info("degraded %s request to %s tier (no %s model available)", task_type, lower,
                                 policy_tier if tier else "higher-tier")
                        break
                    except GatewayError:
                        continue
            if chain is None:
                raise
        rid = uuid.uuid4().hex[:12]
        attempts: list[dict] = []
        retries = self.max_retries()

        dead_transports: set[str] = set()  # e.g. a bad OpenRouter key fails every OpenRouter route
        max_attempts = int(self.defaults.get("max_attempts_per_request", 8))
        deadline = time.time() + float(self.defaults.get("request_deadline_s", 120))
        for ci, (spec, route, exploratory) in enumerate(chain):
            # same model reachable via another transport later in the chain? then fail over instantly
            has_sibling = any(s2.alias == spec.alias for s2, _, _ in chain[ci + 1:])
            if len(attempts) >= max_attempts or time.time() > deadline:
                log.warning("request %s stopped after %d attempts (cap/deadline)", rid, len(attempts))
                break
            bkey = f"{spec.alias}|{route.via}"
            tkey = "openrouter" if route.via == "openrouter" else spec.provider
            if tkey in dead_transports or not self.breaker.available(bkey):
                continue
            for i in range(retries + 1):
                t0 = time.time()
                try:
                    resp = self._call(route, messages, **completion_kwargs)
                except Exception as e:  # noqa: BLE001
                    ms = int((time.time() - t0) * 1000)
                    etype = type(e).__name__
                    attempts.append({"model": route.model, "error": f"{etype}: {str(e)[:160]}"})
                    self.store.log_attempt(request_id=rid, alias=spec.alias, provider=spec.provider, tier=spec.tier,
                                           model=route.model, via=route.via, task_type=task_type, status="error",
                                           error_type=etype, error=str(e)[:500], latency_ms=ms, retry_index=i,
                                           exploratory=int(exploratory))
                    self.breaker.failure(bkey)
                    if isinstance(e, (litellm.AuthenticationError, litellm.PermissionDeniedError)):
                        dead_transports.add(tkey)
                        break
                    if isinstance(e, litellm.ContextWindowExceededError) and _allow_compress and comp_info is None:
                        messages, comp_info = self.compressor.compress(
                            messages, int(min(route.max_input_tokens, self._count_tokens(messages)) * 0.7))
                        orig_tok = comp_info["original_tokens"]
                        continue  # retry same route with the compressed request
                    if len(attempts) >= max_attempts or time.time() > deadline:
                        break
                    if isinstance(e, TRANSIENT_ERRORS) and not has_sibling and i < retries \
                            and self.breaker.available(bkey):
                        time.sleep(0 if self.mock else min(8, 0.5 * 2 ** i + random.random() * 0.25))
                        continue
                    break  # next route
                ms = int((time.time() - t0) * 1000)
                usage = getattr(resp, "usage", None)
                p_tok = int(getattr(usage, "prompt_tokens", 0) or p_est)
                c_tok = int(getattr(usage, "completion_tokens", 0) or 0)
                text = resp.choices[0].message.content or ""
                finish = resp.choices[0].finish_reason or ""
                aq = auto_evaluate(text, finish, task_type, wants_json)
                self.breaker.success(bkey)
                self.store.log_attempt(request_id=rid, alias=spec.alias, provider=spec.provider, tier=spec.tier,
                                       model=route.model, via=route.via, task_type=task_type, status="ok",
                                       prompt_tokens=p_tok, completion_tokens=c_tok, latency_ms=ms, retry_index=i,
                                       exploratory=int(exploratory), auto_quality=aq, finish_reason=finish,
                                       original_tokens=orig_tok, compressed=int(bool(comp_info)))
                out = GatewayResponse(
                    request_id=rid, text=text, model_alias=spec.alias, model=route.model, via=route.via,
                    task_type=task_type, tier_policy=policy_tier, prompt_tokens=p_tok, completion_tokens=c_tok,
                    latency_ms=ms, attempts=len(attempts) + 1,
                    exploratory=exploratory, auto_quality=aq, finish_reason=finish, compression=comp_info, raw=resp)
                if use_cache:
                    self._cache[key] = (time.time(), out)
                self._maybe_judge(out, messages)
                self._maybe_auto_improve()
                return out
        raise AllRoutesFailed(attempts)

    def council_chat(self, messages: list[dict] | str, *, size: int = 3,
                     task_type: Optional[str] = None) -> GatewayResponse:
        """Let the models confer and produce one best answer.

        Several models answer the same conversation — the full chat history
        included, so the discussion stays in context — then the strongest
        respondent reads all the answers and writes the final one: keeping
        what they agree on, resolving disagreements by the strongest
        reasoning, and correcting errors. Members are picked for diversity
        (a different provider first) from the quality-ordered plan, and every
        call goes through chat(), so usage logging, failover and learning
        behave exactly as usual. If fewer than two models answer, it
        degrades gracefully to the best single answer.
        """
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        task_type = task_type or classify_task(messages)
        chain, _, _, _ = self.plan(messages, "quality", task_type, None, None)
        picked: list[str] = []
        seen_providers: set[str] = set()
        for spec, _, _ in chain:  # first pass: one model per provider, best first
            if spec.alias not in picked and spec.provider not in seen_providers:
                picked.append(spec.alias)
                seen_providers.add(spec.provider)
            if len(picked) >= size:
                break
        if len(picked) < size:  # second pass: fill from any remaining models
            for spec, _, _ in chain:
                if spec.alias not in picked:
                    picked.append(spec.alias)
                if len(picked) >= size:
                    break
        got: dict[str, GatewayResponse] = {}

        def ask(alias: str) -> None:
            try:
                r = self.chat(messages, strategy="quality", models=[alias],
                              use_cache=False, task_type=task_type, _allow_compress=False)
                if r.text.strip():
                    got[alias] = r
            except Exception:  # noqa: BLE001 — a member that fails simply does not vote
                pass

        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max(1, len(picked))) as ex:
            list(ex.map(ask, picked))
        answers = [(a, got[a]) for a in picked if a in got]
        if not answers:
            return self.chat(messages, task_type=task_type)  # nobody answered: normal path, with its error
        members = [{"alias": a, "provider": self.models[a].provider, "text": r.text} for a, r in answers]
        if len(answers) == 1:
            only = answers[0][1]
            only.council = {"members": members, "judge": None, "note": "only one model was available"}
            return only
        joined = "\n\n".join(f"--- Answer from {m['provider']} model {m['alias']} ---\n{m['text']}"
                             for m in members)
        synth = list(messages) + [{"role": "user", "content":
            "Several AI models have answered the conversation above, with its full history. "
            "Their answers are below. Write the single best final answer for the user: keep what "
            "they agree on, and where they disagree go with the strongest reasoning and correct "
            "any errors. Answer directly, as if you are the only one replying — do not mention "
            "the other models or this process.\n\n" + joined}]
        judge = answers[0][0]  # the strongest respondent synthesizes
        try:
            final = self.chat(synth, strategy="quality", models=[judge],
                              use_cache=False, task_type=task_type, _allow_compress=False)
        except Exception:  # noqa: BLE001 — synthesis failed: the best single answer stands
            final = answers[0][1]
            final.council = {"members": members, "judge": None, "note": "synthesis failed; best single answer shown"}
            return final
        final.council = {"members": members, "judge": judge}
        return final

    # -- learning signals --------------------------------------------------- #
    def feedback(self, request_id: str, score: float, note: str = "", source: str = "user") -> None:
        """Record explicit quality feedback in [0, 1] for a request. Overrides auto-eval."""
        score = max(0.0, min(1.0, float(score)))
        self.store.execute(
            "INSERT INTO feedback(request_id,score,source,note,ts) VALUES(?,?,?,?,?) "
            "ON CONFLICT(request_id) DO UPDATE SET score=excluded.score, source=excluded.source, note=excluded.note",
            (request_id, score, source, note, time.time()))
        self.learner.invalidate()

    def _maybe_judge(self, resp: GatewayResponse, messages: list[dict]) -> None:
        """Optional LLM-as-judge on a sample of traffic, using a cheap model."""
        lcfg = self.defaults.get("learning", {})
        rate = float(lcfg.get("judge_sample_rate", 0.0))
        judge = lcfg.get("judge_model")
        if rate <= 0 or not judge or judge == resp.model_alias or random.random() > rate:
            return
        prompt = messages[-1].get("content", "") if messages else ""
        jm = [{"role": "system", "content": "Rate the answer's correctness and helpfulness from 0 to 10. "
                                            "Reply with only the number."},
              {"role": "user", "content": f"QUESTION:\n{str(prompt)[:4000]}\n\nANSWER:\n{resp.text[:6000]}"}]
        try:
            r = self.chat(jm, models=[judge], use_cache=False, task_type="judge", strategy="efficient")
            m = re.search(r"\d+(\.\d+)?", r.text)
            if m:
                self.feedback(resp.request_id, float(m.group()) / 10, source="judge")
        except Exception as e:  # noqa: BLE001
            log.debug("judge failed: %s", e)

    def _maybe_auto_improve(self) -> None:
        every = int(self.defaults.get("auto_improve_every_n_requests", 50))
        if every <= 0:
            return
        count = self.store.request_count()
        last = int(self.store.get_policy("last_improve_at_count", 0))
        if count - last >= every:
            if self.store.compaction_needed():
                res = self.store.compact()
                log.info("storage compacted: %s", res)
            changes = self.optimizer.run(apply=True)
            for c in changes:
                log.info("auto-improve: %s %s %s -> %s (%s)", c["kind"], c["target"], c["old"], c["new"], c["reason"])

    # -- reporting ---------------------------------------------------------- #
    UNIT_NAMES = {"openai": "OpenAI", "anthropic": "Anthropic", "vertex_ai": "Google Vertex AI",
                  "deepseek": "DeepSeek", "openrouter": "OpenRouter"}

    def ai_units(self) -> dict:
        """Every AI unit (provider, local server or network node) MHexeract can reach right now, with live state:
        working = answered in the last 24 hours, ready = usable but idle, resting = paused after errors."""
        now = time.time()
        quarantine = {k: v for k, v in self.store.get_policy("quarantine", {}).items() if v > now}
        units: dict[str, dict] = {}
        model_unit: dict[str, str] = {}
        for spec in self.models.values():
            for r in spec.routes:
                if not self._route_available(spec, r):
                    continue
                key = "openrouter" if r.via == "openrouter" else spec.provider
                if key.startswith("node:"):
                    n = self._node_info.get(key[5:], {})
                    name, kind = n.get("name", key[5:]), n.get("kind", "local")
                else:
                    name, kind = self.UNIT_NAMES.get(key, key), "cloud"
                u = units.setdefault(key, {"id": key, "name": name, "kind": kind, "models": [], "resting_models": []})
                if spec.alias not in u["models"]:
                    u["models"].append(spec.alias)
                    model_unit.setdefault(spec.alias, key)
                t = self.breaker.opened_at.get(f"{spec.alias}|{r.via}")
                if (t is not None and now - t < self.breaker.cooldown) or spec.alias in quarantine:
                    u["resting_models"].append(spec.alias)
        since = now - 86400
        rows = self.store.query("""SELECT alias, via, provider, COUNT(*) n, SUM(status='ok') ok, MAX(ts) last,
                                   SUM(prompt_tokens+completion_tokens) tok FROM attempts
                                   WHERE ts>=? AND cached=0 GROUP BY alias, via""", (since,))
        for r in rows:
            key = "openrouter" if r["via"] == "openrouter" else (r["provider"] or model_unit.get(r["alias"], ""))
            u = units.get(key)
            if not u:
                continue
            u["requests_24h"] = u.get("requests_24h", 0) + (r["ok"] or 0)
            u["errors_24h"] = u.get("errors_24h", 0) + (r["n"] - (r["ok"] or 0))
            u["tokens_24h"] = u.get("tokens_24h", 0) + (r["tok"] or 0)
            if (r["ok"] or 0) and r["last"] > u.get("last_used", 0):
                u["last_used"], u["last_model"] = r["last"], r["alias"]
        last = self.store.query("""SELECT alias, via, provider, ts FROM attempts WHERE status='ok' AND cached=0
                                   ORDER BY id DESC LIMIT 1""")
        for u in units.values():
            for k in ("requests_24h", "errors_24h", "tokens_24h"):
                u.setdefault(k, 0)
            u["resting_models"] = sorted(set(u["resting_models"]))
            if u["models"] and len(u["resting_models"]) == len(u["models"]):
                u["state"] = "resting"
            elif u.get("last_used"):
                u["state"] = "working"
            else:
                u["state"] = "ready"
        order = {"working": 0, "ready": 1, "resting": 2}
        out = sorted(units.values(), key=lambda u: (order[u["state"]], -u["requests_24h"], u["name"].lower()))
        latest = None
        if last:
            lr = last[0]
            latest = {"model": lr["alias"], "unit": "openrouter" if lr["via"] == "openrouter" else lr["provider"],
                      "ts": lr["ts"]}
        nd = load_nodes(self.nodes_path) if getattr(self, "nodes_path", None) else {}
        free_only = not self.allow_paid()
        for u in out:
            u["free"] = u["id"].startswith("node:") and (u["kind"] != "cloud" or u["id"][5:] in FREE_NODE_IDS) \
                or u["id"] == "openrouter"
        return {"units": out, "latest": latest, "mock": self.mock, "free_only": free_only,
                "in_use": sum(u["state"] == "working" for u in out), "available": len(out),
                "models": sum(len(u["models"]) for u in out),
                "more_with_key": sum(n.get("status") == "needs_key" and (not free_only or n.get("id") in FREE_NODE_IDS)
                                     for n in nd.get("nodes", [])),
                "scanned_at": nd.get("scanned_at")}

    def report(self, days: int = 7) -> dict:
        since = time.time() - days * 86400
        q = self.store.query
        totals = dict(q("""SELECT SUM(CASE WHEN status='ok' THEN n END) requests, SUM(CASE WHEN status='ok' THEN n END) ok,
                      SUM(CASE WHEN status='error' THEN n END) errors, SUM(cached_n) cache_hits,
                      SUM(prompt_tokens) prompt_tokens, SUM(completion_tokens) completion_tokens,
                      SUM(compressed_n) compressed_requests,
                      SUM(CASE WHEN compressed_n>0 THEN original_tokens - prompt_tokens END) tokens_saved_by_compression
                      FROM usage_all WHERE ts>=?""", (since,))[0])
        for k in list(totals):
            totals[k] = totals[k] or 0
        attempts_total = totals["ok"] + totals["errors"]
        totals["success_rate"] = round(totals["ok"] / attempts_total, 4) if attempts_total else None
        qrow = q("""SELECT AVG(COALESCE(f.score,a.auto_quality)) q FROM attempts a LEFT JOIN feedback f
                    ON f.request_id=a.request_id WHERE a.ts>=? AND a.status='ok' AND a.cached=0""", (since,))[0]
        totals["avg_quality"] = round(qrow["q"], 3) if qrow["q"] is not None else None

        daily = q("""SELECT day, SUM(CASE WHEN status='ok' THEN n END) requests,
                     SUM(prompt_tokens) p_tok, SUM(completion_tokens) c_tok,
                     SUM(CASE WHEN status='error' THEN n END) errors
                     FROM usage_all WHERE ts>=? GROUP BY day ORDER BY day""", (since,))
        by_provider = q("""SELECT provider, SUM(prompt_tokens+completion_tokens) tokens,
                           SUM(CASE WHEN status='ok' THEN n END) ok FROM usage_all WHERE ts>=? AND cached_n=0
                           GROUP BY provider ORDER BY tokens DESC""", (since,))
        by_model = q("""SELECT alias, via, tier, SUM(n) attempts, SUM(CASE WHEN status='ok' THEN n ELSE 0 END) ok,
                        SUM(prompt_tokens) p_tok, SUM(completion_tokens) c_tok,
                        CAST(SUM(latency_sum)/SUM(n) AS INT) avg_ms
                        FROM usage_all WHERE ts>=? AND cached_n=0 GROUP BY alias, via ORDER BY attempts DESC""", (since,))
        by_task = q("""SELECT a.task_type, COUNT(*) n, ROUND(AVG(COALESCE(f.score,a.auto_quality)),3) quality,
                       CAST(AVG(a.prompt_tokens+a.completion_tokens) AS INT) avg_tokens FROM attempts a LEFT JOIN feedback f ON f.request_id=a.request_id
                       WHERE a.ts>=? AND a.status='ok' AND a.cached=0 GROUP BY a.task_type""", (since,))
        policy = self.task_policy()
        by_task = [dict(r, tier_policy=policy.get(r["task_type"], "-")) for r in by_task]
        errors = q("""SELECT error_type, COUNT(*) n FROM attempts WHERE ts>=? AND status='error'
                      GROUP BY error_type ORDER BY n DESC LIMIT 8""", (since,))
        arms = []
        for (alias, task), st in sorted(self.learner.all_stats().items()):
            if alias not in self.models:
                continue
            tot = st.ok + st.fail
            arms.append({"alias": alias, "task_type": task, "n": st.n,
                         "quality": round(self.learner.expected_quality(self.models[alias], task), 3),
                         "reliability": round(st.ok / tot, 3) if tot else None,
                         "avg_ms": int(st.latency_ms), "avg_out_tokens": int(st.out_tokens)})
        imps = q("SELECT ts,kind,target,old_value,new_value,reason,status FROM improvements ORDER BY id DESC LIMIT 50")
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "days": days,
            "mode": ("lite" if LITE else "litellm") + (" + mock" if self.mock else ""),
            "totals": totals,
            "daily": [dict(r) for r in daily],
            "by_provider": [dict(r) for r in by_provider],
            "by_model": [dict(r) for r in by_model],
            "by_task": by_task,
            "errors": [dict(r) for r in errors],
            "learned": arms,
            "policy": {"task_policy": policy,
                       "route_order": self.store.get_policy("route_order", {}),
                       "expected_output_tokens": self.store.get_policy("expected_output_tokens", {}),
                       "max_retries_per_route": self.max_retries(),
                       "compression_trigger_tokens": self.compression_trigger(),
                       "compression_keep_turns": self.compression_keep_turns(),
                       "quarantine": {k: v for k, v in self.store.get_policy("quarantine", {}).items()
                                      if v > time.time()}},
            "storage": self.store.storage_stats(),
            "recent_improvements": [dict(r) for r in imps],
            "models": [{"alias": m.alias, "provider": m.provider, "tier": m.tier, "quality": m.quality,
                        "effort": m.effort,
                        "routes": [{"via": r.via, "model": r.model, "available": self._route_available(m, r)}
                                   for r in m.routes]} for m in self.models.values()],
        }


# --------------------------------------------------------------------------- #
# Offline simulation (demonstrates learning + improvement without API keys)
# --------------------------------------------------------------------------- #
SIM_PROMPTS = {
    "simple": ["What's the capital of France?", "Say hello in Spanish.", "What day comes after Monday?"],
    "extraction": ["Extract the name and email fields as JSON from: Jane Doe, jane@x.com"],
    "summarization": ["Summarize the key points of this memo about Q3 hiring plans."],
    "creative": ["Write a short poem about Minneapolis winters."],
    "code": ["Write a python function to debug a failing regex for email validation."],
    "reasoning": ["Analyze the trade-offs of event sourcing vs CRUD for a payments system, step by step."],
}
# hidden "true" quality per (tier, task) used by the simulator
SIM_TRUTH = {
    "economy": {"simple": 0.9, "extraction": 0.85, "summarization": 0.8, "creative": 0.62, "code": 0.55, "reasoning": 0.45},
    "standard": {"simple": 0.92, "extraction": 0.9, "summarization": 0.88, "creative": 0.85, "code": 0.82, "reasoning": 0.7},
    "premium": {"simple": 0.93, "extraction": 0.92, "summarization": 0.9, "creative": 0.9, "code": 0.92, "reasoning": 0.9},
}


def simulate(gw: Gateway, n: int, seed: int = 7) -> None:
    rng = random.Random(seed)
    gw.learner.rng.seed(seed)
    tier_of = {r.model: s.tier for s in gw.models.values() for r in s.routes}

    def hook(route: Route, messages: list[dict]) -> tuple[str, str]:
        if rng.random() < (0.35 if route.model.startswith("deepseek/") else 0.03):  # flaky direct DeepSeek
            raise litellm.ServiceUnavailableError(message="sim 503", llm_provider="sim", model=route.model)
        return "x" * rng.randint(200, 1600), "stop"

    gw.mock, gw.mock_hook = True, hook
    for i in range(n):
        task = rng.choice(TASK_TYPES)
        prompt = rng.choice(SIM_PROMPTS[task]) + f" #{i}"
        try:
            r = gw.chat(prompt, use_cache=False)
        except GatewayError as e:
            print("request failed:", e)
            continue
        truth = SIM_TRUTH[tier_of[r.model]][r.task_type]
        gw.feedback(r.request_id, min(1, max(0, rng.gauss(truth, 0.08))), source="sim")


# --------------------------------------------------------------------------- #
# OpenAI model discovery: add every chat model your key can use to models.json
# --------------------------------------------------------------------------- #
_OPENAI_NON_CHAT = re.compile(r"(embedding|moderation|whisper|tts|transcribe|realtime|audio|image|dall-e|"
                              r"davinci|babbage|search|live|sora|cyber|rosalind|computer-use)", re.I)


def openai_tier(model_id: str) -> tuple[str, int]:
    m = model_id.lower()
    if any(x in m for x in ("luna", "nano")):
        return "economy", 5
    if any(x in m for x in ("terra", "mini")):
        return "standard", 7
    if any(x in m for x in ("astra", "pro")):
        return "premium", 10
    return "premium", 9


def discover_openai(config_path: str, write: bool = False) -> list[dict]:
    """List models from GET /v1/models and add missing chat models (direct + OpenRouter routes)."""
    import urllib.request
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise GatewayError("OPENAI_API_KEY is not set")
    req = urllib.request.Request("https://api.openai.com/v1/models", headers={"Authorization": f"Bearer {key}"})
    import urllib.error
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            ids = sorted(m["id"] for m in json.loads(r.read())["data"])
    except urllib.error.HTTPError as e:
        raise GatewayError(f"OpenAI rejected the request (HTTP {e.code}); check OPENAI_API_KEY") from None
    except (urllib.error.URLError, OSError) as e:
        raise GatewayError(f"could not reach OpenAI: {e}") from None
    chat = [i for i in ids if (i.startswith("gpt-") or re.match(r"^o\d", i)) and not _OPENAI_NON_CHAT.search(i)
            and not re.search(r"-\d{4}-\d{2}-\d{2}$", i)]  # skip dated snapshots of aliases
    cfg = json.loads(Path(config_path).read_text())
    known = {r["model"].split("/")[-1] for m in cfg["models"] for r in m["routes"]} | {m["alias"] for m in cfg["models"]}
    added = []
    for mid in chat:
        if mid in known:
            continue
        tier, quality = openai_tier(mid)
        added.append({"alias": mid, "provider": "openai", "tier": tier, "quality": quality, "discovered": True,
                      "routes": [{"model": mid, "via": "direct"},
                                 {"model": f"openrouter/openai/{mid}", "via": "openrouter"}]})
    if write and added:
        cfg["models"].extend(added)
        Path(config_path).write_text(json.dumps(cfg, indent=2) + "\n")
    return added


# --------------------------------------------------------------------------- #
# AI nodes: keep looking for more AI providers and servers, and route to them
# --------------------------------------------------------------------------- #
# A node is any OpenAI-compatible endpoint: a cloud provider you have a key for, a model server on this
# device (Ollama, LM Studio, llama.cpp, vLLM, Jan, KoboldCpp, LocalAI, text-generation-webui, GPT4All),
# or one on your home network. Discovered chat models join the router with a cautious tier/quality guess;
# the learner and quarantine then promote or drop them based on real results.
# (id, name, base_url, env key or "", where to get a key, models list path or "")
CLOUD_NODES = [
    ("openai", "OpenAI", "https://api.openai.com/v1", "OPENAI_API_KEY", "https://platform.openai.com/api-keys", "/models"),
    ("deepseek", "DeepSeek", "https://api.deepseek.com", "DEEPSEEK_API_KEY", "https://platform.deepseek.com/api_keys", "/models"),
    ("openrouter", "OpenRouter", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "https://openrouter.ai/keys", "/models"),
    ("gemini", "Google Gemini", "https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY", "https://aistudio.google.com/apikey", "/models"),
    ("groq", "Groq", "https://api.groq.com/openai/v1", "GROQ_API_KEY", "https://console.groq.com/keys", "/models"),
    ("mistral", "Mistral", "https://api.mistral.ai/v1", "MISTRAL_API_KEY", "https://console.mistral.ai/api-keys", "/models"),
    ("xai", "xAI Grok", "https://api.x.ai/v1", "XAI_API_KEY", "https://console.x.ai", "/models"),
    ("together", "Together AI", "https://api.together.xyz/v1", "TOGETHER_API_KEY", "https://api.together.ai/settings/api-keys", "/models"),
    ("fireworks", "Fireworks", "https://api.fireworks.ai/inference/v1", "FIREWORKS_API_KEY", "https://fireworks.ai/account/api-keys", "/models"),
    ("cerebras", "Cerebras", "https://api.cerebras.ai/v1", "CEREBRAS_API_KEY", "https://cloud.cerebras.ai", "/models"),
    ("sambanova", "SambaNova", "https://api.sambanova.ai/v1", "SAMBANOVA_API_KEY", "https://cloud.sambanova.ai/apis", "/models"),
    ("huggingface", "Hugging Face", "https://router.huggingface.co/v1", "HF_TOKEN", "https://huggingface.co/settings/tokens", "/models"),
    ("nvidia", "NVIDIA NIM", "https://integrate.api.nvidia.com/v1", "NVIDIA_API_KEY", "https://build.nvidia.com", "/models"),
    ("deepinfra", "DeepInfra", "https://api.deepinfra.com/v1/openai", "DEEPINFRA_API_KEY", "https://deepinfra.com/dash/api_keys", "/models"),
    ("novita", "Novita AI", "https://api.novita.ai/v3/openai", "NOVITA_API_KEY", "https://novita.ai/settings/key-management", "/models"),
    ("hyperbolic", "Hyperbolic", "https://api.hyperbolic.xyz/v1", "HYPERBOLIC_API_KEY", "https://app.hyperbolic.xyz/settings", "/models"),
    ("moonshot", "Moonshot Kimi", "https://api.moonshot.ai/v1", "MOONSHOT_API_KEY", "https://platform.moonshot.ai/console/api-keys", "/models"),
    ("dashscope", "Alibaba Qwen", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1", "DASHSCOPE_API_KEY", "https://modelstudio.console.alibabacloud.com", "/models"),
    ("zai", "Z.ai GLM", "https://api.z.ai/api/paas/v4", "ZAI_API_KEY", "https://z.ai/manage-apikey/apikey-list", "/models"),
    ("minimax", "MiniMax", "https://api.minimax.io/v1", "MINIMAX_API_KEY", "https://platform.minimax.io", "/models"),
    ("cohere", "Cohere", "https://api.cohere.ai/compatibility/v1", "COHERE_API_KEY", "https://dashboard.cohere.com/api-keys", "/models"),
    ("meta", "Meta Muse AI", "https://api.meta.ai/v1", "MODEL_API_KEY", "https://dev.meta.ai", "/models"),
    ("pollinations", "Pollinations", "https://text.pollinations.ai/openai", "", "", "/models"),
]
# Providers whose API keys work on a no-card free plan, so they cannot charge anything. The OpenRouter node only
# ever adds ":free" models. Everything else (and every paid OpenRouter route) is off unless paid AI is allowed.
FREE_NODE_IDS = {"openrouter", "gemini", "groq", "mistral", "sambanova", "nvidia", "huggingface", "cohere", "pollinations"}
NODE_KEY_NAMES = sorted({c[3] for c in CLOUD_NODES})
# local model servers: port -> software that usually listens there
LOCAL_PORTS = {11434: "Ollama", 1234: "LM Studio", 8080: "llama.cpp / LocalAI", 8000: "vLLM", 1337: "Jan",
               5001: "KoboldCpp", 5000: "text-generation-webui", 4891: "GPT4All", 8081: "llama.cpp"}
LAN_PORTS = (11434, 1234, 8080, 8000)
_NON_CHAT = re.compile(r"(embed|moderation|whisper|tts|transcri|realtime|audio|image|dall-e|sora|rerank|guard|"
                       r"safety|vision-only|ocr|speech|clip|bge|e5-|nomic|davinci|babbage|search|computer-use|"
                       r"flux|stable-diffusion|sdxl|video|music|lyria|veo|imagen)", re.I)
NODE_DEFAULTS = {"scan_every_minutes": 60, "lan_scan": True, "max_models_per_node": 12, "max_free_openrouter": 20}


class _AttrDict(dict):
    __getattr__ = dict.get


def _attr(d: Any) -> Any:
    if isinstance(d, dict):
        return _AttrDict({k: _attr(v) for k, v in d.items()})
    if isinstance(d, list):
        return [_attr(x) for x in d]
    return d


def node_key(env: str) -> Optional[str]:
    return os.getenv(env) if env else None


def _http_json(url: str, key: Optional[str] = None, timeout: float = 8, data: Optional[dict] = None,
               headers: Optional[dict] = None) -> Any:
    import urllib.request
    h = {"User-Agent": "MHexeract", "Accept": "application/json", **(headers or {})}
    if key:
        h["Authorization"] = f"Bearer {key}"
    body = None
    if data is not None:
        body = json.dumps(data).encode()
        h["Content-Type"] = "application/json"
    with urllib.request.urlopen(urllib.request.Request(url, data=body, headers=h), timeout=timeout) as r:
        return json.loads(r.read() or b"null")


def node_completion(base_url: str, key: Optional[str], wire_model: str, messages: list, timeout: float,
                    provider: str, model: str, extra_headers: Optional[dict] = None, **kw: Any) -> Any:
    """POST {base}/chat/completions on any OpenAI-compatible node; errors map onto the router's error types."""
    import urllib.error
    body = {"model": wire_model, "messages": messages,
            **{k: v for k, v in kw.items() if k in ("max_tokens", "temperature", "top_p", "response_format",
                                                    "stop", "seed", "tools", "tool_choice")}}
    if provider == "openai" and "max_tokens" in body:  # current OpenAI models use max_completion_tokens
        body["max_completion_tokens"] = body.pop("max_tokens")
    try:
        data = _http_json(f"{base_url.rstrip('/')}/chat/completions", key, timeout, body, extra_headers)
    except urllib.error.HTTPError as e:
        msg = e.read().decode(errors="replace")[:500]
        cls = {400: "BadRequestError", 401: "AuthenticationError", 402: "RateLimitError",
               403: "PermissionDeniedError", 404: "NotFoundError", 408: "Timeout", 413: "ContextWindowExceededError",
               429: "RateLimitError", 500: "InternalServerError", 502: "ServiceUnavailableError",
               503: "ServiceUnavailableError"}.get(e.code, "APIConnectionError")
        if e.code == 400 and "context" in msg.lower():
            cls = "ContextWindowExceededError"
        raise getattr(litellm, cls)(msg, provider, model) from None
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
        raise litellm.APIConnectionError(str(e), provider, model) from None
    if not isinstance(data, dict) or "error" in data or not data.get("choices"):
        raise litellm.ServiceUnavailableError(str((data or {}).get("error", data))[:500], provider, model)
    for c in data["choices"]:
        c.setdefault("message", {}).setdefault("content", "")
    data.setdefault("usage", {})
    return _attr(data)


def guess_tier(model_id: str, local: bool = False) -> tuple[str, int]:
    """Cautious tier and quality guess from the model name; real results take over once traffic flows."""
    m = model_id.lower()
    size = max((float(x) for x in re.findall(r"(\d+(?:\.\d+)?)b\b", m)), default=0.0)
    if re.search(r"(nano|mini|small|tiny|lite|flash|haiku|instant|\b[1-9]b\b|:free)", m) or (0 < size <= 14):
        tier, q = "economy", 4
    elif re.search(r"(opus|ultra|large|-pro\b|pro-|max\b|reasoner|r1\b|405b|671b|235b|480b|muse-spark|muse-wonder)", m) or size >= 120:
        tier, q = "premium", 7
    else:
        tier, q = "standard", 6
    if local:
        q = min(q, 5)
    return tier, q


def _norm(model_id: str) -> str:
    return re.sub(r"[^a-z0-9.]", "", model_id.lower().split("/")[-1].split(":")[0])


def _list_cloud(node: tuple, cfg: dict) -> dict:
    nid, name, base, env, signup, path = node
    rec = {"id": nid, "name": name, "kind": "cloud", "base_url": base, "env": env, "signup": signup,
           "trusted": True, "models": [], "models_total": 0,
           "free": nid in FREE_NODE_IDS}
    key = node_key(env)
    if env and not key:
        rec["status"] = "needs_key"
        return rec
    try:
        raw = _http_json(base + path, key, 12)
        raw = raw.get("data", raw) if isinstance(raw, dict) else raw
        items = [{"id": m.get("id") or m.get("name", ""), "ctx": m.get("context_length") or m.get("context_window"),
                  "free": str((m.get("pricing") or {}).get("prompt", "x")) in ("0", "0.0")} for m in raw or []]
    except Exception as e:  # noqa: BLE001
        rec.update(status="error", error=str(e)[:200])
        return rec
    for i in items:
        i["id"] = i["id"].removeprefix("models/")  # Gemini lists ids as models/<id>
    items = [i for i in items if i["id"] and not _NON_CHAT.search(i["id"])]
    rec["models_total"] = len(items)
    if nid in ("nvidia", "groq"):  # free hosts that carry Meta's Muse models: keep Muse inside the per-node cap
        items.sort(key=lambda i: 0 if "muse" in i["id"] else 1)
    if nid == "openrouter":  # every paid model is already an OpenRouter fallback route; add the free ones
        items = [i for i in items if i["id"].endswith(":free") or i.get("free")][: int(cfg["max_free_openrouter"])]
    rec["models"] = items[: int(cfg["max_models_per_node"])] if nid != "openrouter" else items
    rec["status"] = "online"
    return rec


def _probe_local(host: str, port: int, kind: str, timeout: float = 1.5) -> Optional[dict]:
    base = f"http://{host}:{port}/v1"
    try:
        raw = _http_json(base + "/models", None, timeout)
        ids = [m.get("id") for m in (raw.get("data") if isinstance(raw, dict) else raw) or [] if m.get("id")]
    except Exception:  # noqa: BLE001
        if port != 11434:
            return None
        try:  # older Ollama without the OpenAI-compatible model list
            ids = [m["name"] for m in _http_json(f"http://{host}:{port}/api/tags", None, timeout).get("models", [])]
        except Exception:  # noqa: BLE001
            return None
    ids = [i for i in ids if not _NON_CHAT.search(i)]
    local = host in ("127.0.0.1", "localhost")
    return {"id": f"{'local' if local else 'lan'}-{host.replace('.', '-')}-{port}",
            "name": f"{LOCAL_PORTS.get(port, 'Model server')} on {'this device' if local else host}",
            "kind": "local" if local else "lan", "base_url": base, "env": "", "status": "online",
            "trusted": local, "models": [{"id": i, "ctx": None} for i in ids], "models_total": len(ids)}


def _lan_hosts() -> list[str]:
    """Hosts on this device's private /24 network that have a common model-server port open."""
    import ipaddress
    import socket
    from concurrent.futures import ThreadPoolExecutor
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))  # no packet is sent; this only picks the outgoing interface
        me = s.getsockname()[0]
        s.close()
    except OSError:
        return []
    ip = ipaddress.ip_address(me)
    if not ip.is_private or ip.is_loopback:
        return []
    net = ipaddress.ip_network(f"{me}/24", strict=False)

    def check(target: tuple[str, int]) -> Optional[str]:
        try:
            with socket.create_connection(target, timeout=0.3):
                return f"{target[0]}:{target[1]}"
        except OSError:
            return None
    targets = [(str(h), p) for h in net.hosts() if str(h) != me for p in LAN_PORTS]
    with ThreadPoolExecutor(max_workers=128) as ex:
        return [r for r in ex.map(check, targets) if r]


def nodes_config(gw: Optional["Gateway"] = None) -> dict:
    cfg = dict(NODE_DEFAULTS, **((gw.defaults.get("nodes") or {}) if gw else {}))
    store = getattr(gw, "store", None)
    if store is not None:
        lan = store.get_policy("bg_lan_scan", None)
        if lan is not None:
            cfg["lan_scan"] = bool(lan)
    if os.getenv("MHEXERACT_LAN_SCAN") is not None:
        cfg["lan_scan"] = os.getenv("MHEXERACT_LAN_SCAN") == "1"
    return cfg


def load_nodes(path: str) -> dict:
    p = Path(path)
    try:
        return json.loads(p.read_text()) if p.exists() else {"scanned_at": None, "nodes": [], "trusted": []}
    except (OSError, json.JSONDecodeError):
        return {"scanned_at": None, "nodes": [], "trusted": []}


def scan_nodes(path: str, gw: Optional["Gateway"] = None) -> dict:
    """Look for every reachable AI node and save the result. Safe to call often; it only reads model lists."""
    from concurrent.futures import ThreadPoolExecutor
    cfg = nodes_config(gw)
    prev = load_nodes(path)
    trusted = set(prev.get("trusted", []))
    extra = [u.strip().rstrip("/") for u in os.getenv("MHEXERACT_NODES", "").split(",") if u.strip()]
    dead = set(load_key_health().get("dead_nodes", []))
    live_nodes = [n for n in CLOUD_NODES if n[0] not in dead]
    with ThreadPoolExecutor(max_workers=24) as ex:
        cloud = list(ex.map(lambda n: _list_cloud(n, cfg), live_nodes))
        local = [r for r in ex.map(lambda p: _probe_local("127.0.0.1", p[0], p[1]), LOCAL_PORTS.items()) if r]
        lan = []
        if cfg.get("lan_scan"):
            hp = [h.split(":") for h in _lan_hosts()]
            lan = [r for r in ex.map(lambda x: _probe_local(x[0], int(x[1]), ""), hp) if r]
    for url in extra:  # nodes you list yourself in MHEXERACT_NODES are trusted
        from urllib.parse import urlparse
        u = urlparse(url if "://" in url else "http://" + url)
        r = _probe_local(u.hostname or "", u.port or 80, "", 3)
        if r:
            r.update(kind="custom", trusted=True, name=f"Model server at {u.hostname}:{u.port or 80}")
            lan.append(r)
    for r in lan:
        if r["base_url"] in trusted:
            r["trusted"] = True
    nodes = cloud + local + lan
    seen = {_norm(r.model.split("/", 1)[-1]) for s in (gw.models.values() if gw else []) if not s.provider.startswith("node:")
            for r in s.routes} | {_norm(a) for a in (gw.models if gw else {}) if ":" not in a}
    for n in nodes:  # tier guess, context window, and skip models the main pool already covers
        fresh = []
        for m in n["models"]:
            k = _norm(m["id"])
            if n["id"] in ("openai", "deepseek") and k in seen:
                continue
            tier, q = guess_tier(m["id"], local=n["kind"] != "cloud")
            fresh.append({"id": m["id"], "tier": tier, "quality": q,
                          "ctx": int(m.get("ctx") or (32_000 if n["kind"] != "cloud" else 128_000))})
        n["models"] = fresh
    before = {n["id"] for n in prev.get("nodes", []) if n.get("status") == "online"}
    out = {"scanned_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "nodes": nodes,
           "trusted": sorted(trusted), "lan_scan": bool(cfg.get("lan_scan")),
           "new": sorted(n["id"] for n in nodes if n["status"] == "online" and n["id"] not in before)}
    Path(path).write_text(json.dumps(out, indent=1))
    return out


def trust_node(path: str, base_url: str, trusted: bool) -> dict:
    d = load_nodes(path)
    t = set(d.get("trusted", []))
    (t.add if trusted else t.discard)(base_url)
    d["trusted"] = sorted(t)
    for n in d.get("nodes", []):
        if n.get("base_url") == base_url and n.get("kind") == "lan":
            n["trusted"] = trusted
    Path(path).write_text(json.dumps(d, indent=1))
    return d


def nodes_summary(d: dict) -> dict:
    ns = d.get("nodes", [])
    used = [n for n in ns if n.get("status") == "online" and n.get("trusted")]
    return {"scanned_at": d.get("scanned_at"), "online": sum(n.get("status") == "online" for n in ns),
            "in_use": len(used), "models_in_use": sum(len(n.get("models", [])) for n in used),
            "needs_key": sum(n.get("status") == "needs_key" for n in ns), "new": d.get("new", [])}


# --------------------------------------------------------------------------- #
# GUI report server (stdlib only, so it also runs on Android/Termux)
# --------------------------------------------------------------------------- #
HERE = Path(__file__).resolve().parent
MANIFEST = {
    "name": "MHexeract Gateway", "short_name": "MHexeract", "start_url": "/", "display": "standalone",
    "background_color": "#0f1417", "theme_color": "#0f1417", "orientation": "any",
    "icons": [{"src": "/icon.svg", "sizes": "any", "type": "image/svg+xml", "purpose": "any maskable"}],
}
ICON_SVG = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><rect width="64" height="64" rx="14" '
            'fill="#14b8a6"/><path d="M14 46V18l9 14 9-14 9 14 9-14v28" fill="none" stroke="#0f1417" '
            'stroke-width="5" stroke-linejoin="round"/></svg>')
# network-first service worker: works offline with the last report, never serves stale data when online
SERVICE_WORKER = """
const C='mhexeract-v6';
self.addEventListener('install',e=>{self.skipWaiting();e.waitUntil(caches.open(C).then(c=>c.addAll(['/','/game.html','/solitaire.html','/tetris.html','/platformer.html','/oregon.html','/blackjack.html','/manifest.webmanifest','/icon.svg'])))});
self.addEventListener('activate',e=>e.waitUntil(self.clients.claim()));
self.addEventListener('fetch',e=>{if(e.request.method!=='GET')return;
  e.respondWith(fetch(e.request).then(r=>{const c=r.clone();caches.open(C).then(x=>x.put(e.request,c));return r})
  .catch(()=>caches.match(e.request)))});
"""


def export_html(gw: "Gateway", path: str, days: int = 7) -> str:
    """Write a self-contained static HTML report (open it on any device, no server needed)."""
    html = (HERE / "dashboard.html").read_text()
    data = json.dumps(gw.report(days), default=str).replace("</", "<\\/")
    html = html.replace("<script>\nconst STATIC", f"<script>window.__REPORT__={data};</script>\n<script>\nconst STATIC", 1)
    Path(path).write_text(html)
    return path


# --------------------------------------------------------------------------- #
# Repository registry: auto-add every GitHub repository your token can reach
# --------------------------------------------------------------------------- #
def _github_token() -> Optional[str]:
    return os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")


def _shape_repo(x: dict) -> dict:
    return {"full_name": x["full_name"], "private": x["private"], "fork": x["fork"],
            "archived": x.get("archived", False), "default_branch": x.get("default_branch"),
            "clone_url": x["clone_url"], "html_url": x["html_url"],
            "description": x.get("description") or "", "updated_at": x.get("updated_at")}


def _gh_cli_repos() -> Optional[list[dict]]:
    """Use `gh api` when no token is exported; gh keeps its own credentials."""
    import shutil
    import subprocess
    if not shutil.which("gh"):
        return None
    try:
        out = subprocess.run(["gh", "api", "--paginate",
                              "user/repos?per_page=100&affiliation=owner,collaborator,organization_member"],
                             check=True, capture_output=True, text=True, timeout=120).stdout
    except (subprocess.SubprocessError, OSError):
        return None
    data = json.loads("[" + out.replace("][", "],[") + "]") if out.strip() else []
    flat = [x for page in data for x in (page if isinstance(page, list) else [page])]
    return [_shape_repo(x) for x in flat]


def discover_repos(token: Optional[str] = None) -> list[dict]:
    """All repositories the token can access: owned, collaborator and organization repos."""
    import urllib.error
    import urllib.request
    token = token or _github_token()
    if not token:
        gh = _gh_cli_repos()  # fall back to an authenticated GitHub CLI, if present
        if gh is not None:
            return gh
        raise GatewayError("GITHUB_TOKEN is not set and the GitHub CLI (gh) is not signed in")
    repos, page = [], 1
    while True:
        req = urllib.request.Request(
            f"https://api.github.com/user/repos?per_page=100&page={page}"
            "&affiliation=owner,collaborator,organization_member&sort=updated",
            headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
                     "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "MHexeract"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                batch = json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise GatewayError(f"GitHub rejected the request (HTTP {e.code}); check GITHUB_TOKEN") from None
        except (urllib.error.URLError, OSError) as e:
            raise GatewayError(f"could not reach GitHub: {e}") from None
        repos += [_shape_repo(x) for x in batch]
        if len(batch) < 100:
            return repos
        page += 1


def sync_repos(registry_path: str = "repos.json", clone_dir: Optional[str] = None,
               include_archived: bool = False) -> dict:
    """Refresh the registry, adding new repositories automatically; optionally clone or update them."""
    import subprocess
    path = Path(registry_path)
    old = {r["full_name"]: r for r in json.loads(path.read_text()).get("repos", [])} if path.exists() else {}
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    found = [r for r in discover_repos() if include_archived or not r["archived"]]
    added = [r["full_name"] for r in found if r["full_name"] not in old]
    removed = sorted(set(old) - {r["full_name"] for r in found})
    for r in found:
        r["added_at"] = old.get(r["full_name"], {}).get("added_at", now)
    cloned, updated, failed = [], [], []
    if clone_dir:
        token = _github_token()
        # token goes in a per-command header, never into the remote URL or .git/config;
        # without a token, git uses whatever credential helper is configured (e.g. `gh auth setup-git`)
        auth = ["-c", f"http.extraHeader=Authorization: Basic "
                f"{__import__('base64').b64encode(f'x-access-token:{token}'.encode()).decode()}"] if token else []
        for r in found:
            dest = Path(clone_dir) / r["full_name"]
            cmd = (["git", *auth, "-C", str(dest), "pull", "--ff-only", "-q"] if (dest / ".git").exists()
                   else ["git", *auth, "clone", "-q", r["clone_url"], str(dest)])
            try:
                dest.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run(cmd, check=True, capture_output=True, timeout=600)
                (updated if "pull" in cmd else cloned).append(r["full_name"])
                r["local_path"] = str(dest)
            except (subprocess.SubprocessError, OSError) as e:
                failed.append({"repo": r["full_name"], "error": str(getattr(e, "stderr", b"") or e)[:300]})
    path.write_text(json.dumps({"synced_at": now, "repos": found}, indent=2) + "\n")
    return {"synced_at": now, "total": len(found), "added": added, "removed": removed,
            "cloned": cloned, "updated": updated, "failed": failed}


def load_repos(registry_path: str = "repos.json") -> dict:
    p = Path(registry_path)
    return json.loads(p.read_text()) if p.exists() else {"synced_at": None, "repos": []}


def run_scheduled_improvement(gw: "Gateway", force: bool = False) -> Optional[list]:
    """Time-based process improvement: compact storage and tune the policy when it is due.

    Complements the request-count trigger so the policy keeps improving on quiet
    days. Interval comes from defaults.auto_improve_every_hours (0 disables).
    """
    hours = float(os.getenv("GATEWAY_AUTO_IMPROVE_HOURS") or gw.defaults.get("auto_improve_every_hours", 6))
    if hours <= 0 and not force:
        return None
    last = float(gw.store.get_policy("last_scheduled_improve_ts", 0) or 0)
    if not force and time.time() - last < hours * 3600:
        return None
    if gw.store.compaction_needed():
        log.info("scheduled compaction: %s", gw.store.compact())
    changes = gw.optimizer.run(apply=True)
    gw.store.set_policy("last_scheduled_improve_ts", time.time())
    for c in changes:
        log.info("scheduled improve: %s %s %s -> %s (%s)", c["kind"], c["target"], c["old"], c["new"], c["reason"])
    return changes


def _scheduled_improver(gw: "Gateway", lock: threading.Lock, poll_s: float = 300) -> None:
    while True:
        try:
            if gw.store.get_policy("bg_auto_improve", True):
                with lock:
                    run_scheduled_improvement(gw)
        except Exception:  # noqa: BLE001
            log.exception("scheduled improvement failed")
        try:  # auto-add new repositories (registry only; cloning stays manual)
            hours = float(os.getenv("GATEWAY_REPO_SYNC_HOURS") or gw.defaults.get("repo_sync_every_hours", 6))
            if not gw.store.get_policy("bg_repo_sync", True):
                hours = 0
            last = float(gw.store.get_policy("last_repo_sync_ts", 0) or 0)
            if (_github_token() or __import__("shutil").which("gh")) and hours > 0 \
                    and time.time() - last >= hours * 3600:
                res = sync_repos(gw.repo_registry)
                gw.store.set_policy("last_repo_sync_ts", time.time())
                if res["added"]:
                    log.info("auto-added repositories: %s", ", ".join(res["added"]))
        except Exception:  # noqa: BLE001
            log.exception("repository sync failed")
        try:  # keep looking for more AI nodes (cloud keys, this device, home network)
            if gw.store.get_policy("bg_node_scan", True):
                run_node_scan(gw, lock)
        except Exception:  # noqa: BLE001
            log.exception("AI node scan failed")
        try:  # check keys and providers, clean what is broken, adopt free nodes
            if gw.store.get_policy("bg_gardener", True):
                run_gardener(gw)
        except Exception:  # noqa: BLE001
            log.exception("provider gardener failed")
        try:  # strive toward the no-cost goals: verify keyless providers, hunt for new ones
            if gw.store.get_policy("bg_strive", True):
                run_strive(gw)
        except Exception:  # noqa: BLE001
            log.exception("goal striving failed")
        try:  # audit the pages for the automated Android UI polish
            if gw.store.get_policy("bg_ui_polish", True):
                run_ui_polish(gw)
        except Exception:  # noqa: BLE001
            log.exception("UI polish audit failed")
        try:  # refresh the OBD trouble-code database from its open-source upstream
            if gw.store.get_policy("bg_obd_update", True):
                run_obd_update(gw)
        except Exception:  # noqa: BLE001
            log.exception("OBD code database update failed")
        time.sleep(poll_s)


def run_node_scan(gw: "Gateway", lock: threading.Lock, force: bool = False) -> Optional[dict]:
    """Scan for AI nodes when due (defaults.nodes.scan_every_minutes; 0 or MHEXERACT_NODE_SCAN=0 disables)."""
    minutes = float(nodes_config(gw).get("scan_every_minutes", 60))
    if not force and (minutes <= 0 or os.getenv("MHEXERACT_NODE_SCAN", "1") == "0" or gw.mock):
        return None
    last = float(gw.store.get_policy("last_node_scan_ts", 0) or 0)
    keys_now = ",".join(k for k in NODE_KEY_NAMES if node_key(k))  # names only, never values
    keys_changed = keys_now != gw.store.get_policy("node_keys_seen", None)
    if not force and not keys_changed and time.time() - last < minutes * 60:
        return None
    res = scan_nodes(gw.nodes_path, gw)  # network work happens outside the lock
    with lock:
        n = gw.load_nodes()
        gw.store.set_policy("last_node_scan_ts", time.time())
        gw.store.set_policy("node_keys_seen", keys_now)
    s = nodes_summary(res)
    log.info("AI node scan: %s online, %s in use with %s models, %s need a key%s", s["online"], s["in_use"], n,
             s["needs_key"], (", new: " + ", ".join(s["new"])) if s["new"] else "")
    return res


# --------------------------------------------------------------------------- #
# Web browser (reader mode): fetch a page and return readable text + links
# --------------------------------------------------------------------------- #
WEB_UA = "Mozilla/5.0 (Linux; Android 14) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Mobile Safari/537.36 MHexeract"


def _public_host(host: str) -> bool:
    """True when every address the host resolves to is public (blocks localhost/LAN fetches)."""
    import ipaddress
    import socket
    if os.getenv("GATEWAY_ALLOW_PRIVATE_FETCH") == "1":
        return True
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return False
    for info in infos:
        ip = ipaddress.ip_address(info[4][0].split("%")[0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast or ip.is_unspecified:
            return False
    return True


def normalize_web_input(q: str) -> str:
    """URL as typed, or a web search when it does not look like an address."""
    from urllib.parse import quote_plus
    q = q.strip()
    if re.match(r"^https?://", q, re.I):
        return q
    if " " not in q and re.match(r"^[\w.-]+\.[a-z]{2,}(:\d+)?(/.*)?$", q, re.I):
        return "https://" + q
    return "https://html.duckduckgo.com/html/?q=" + quote_plus(q)


def fetch_page(url: str, max_bytes: int = 3_000_000) -> dict:
    import urllib.request
    from html.parser import HTMLParser
    from urllib.parse import parse_qs as _pq, urljoin, urlparse as _up

    def check(u: str) -> None:
        pu = _up(u)
        if pu.scheme not in ("http", "https") or not pu.hostname:
            raise ValueError("only http and https addresses can be opened")
        if not _public_host(pu.hostname):
            raise ValueError("that address points to a private or local network and is blocked")

    class Guard(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
            check(newurl)  # a public page must not redirect us into the LAN
            return super().redirect_request(req, fp, code, msg, headers, newurl)

    url = normalize_web_input(url)
    check(url)
    opener = urllib.request.build_opener(Guard)
    req = urllib.request.Request(url, headers={"User-Agent": WEB_UA, "Accept": "text/html,text/plain;q=0.9,*/*;q=0.5",
                                               "Accept-Language": "en-US,en;q=0.8"})
    with opener.open(req, timeout=15) as resp:
        final = resp.geturl()
        ctype = resp.headers.get("Content-Type", "")
        raw = resp.read(max_bytes + 1)[:max_bytes]
        charset = resp.headers.get_content_charset() or "utf-8"
    body = raw.decode(charset, errors="replace")
    if "html" not in ctype and not body.lstrip()[:200].lower().startswith(("<!doctype", "<html")):
        return {"url": final, "title": final, "text": body[:100_000], "links": [], "content_type": ctype}

    class Reader(HTMLParser):
        SKIP = {"script", "style", "noscript", "svg", "template", "head", "iframe", "form", "button", "select",
                "nav", "aside", "footer"}  # menus and chrome are not content
        BLOCK = {"p", "div", "section", "article", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6",
                 "header", "footer", "pre", "blockquote", "table", "ul", "ol", "dd", "dt"}

        def __init__(self) -> None:
            super().__init__(convert_charrefs=True)
            self.skip = 0
            self.title = ""
            self.in_title = False
            self.out: list[str] = []
            self.links: list[dict] = []
            self.href: Optional[str] = None
            self.atext: list[str] = []
            self.main_depth = 0
            self.main: list[str] = []  # text inside <main>/<article>: preferred when substantial

        def emit(self, t: str) -> None:
            self.out.append(t)
            if self.main_depth:
                self.main.append(t)

        def handle_starttag(self, tag, attrs):  # noqa: ANN001
            if tag == "title":
                self.in_title = True
            if tag in self.SKIP:
                self.skip += 1
            if tag in ("main", "article"):
                self.main_depth += 1
            if tag in self.BLOCK:
                self.emit("\n")
            if tag in ("h1", "h2", "h3"):
                self.emit("## ")
            if tag == "li":
                self.emit("- ")
            if tag == "a":
                self.href, self.atext = dict(attrs).get("href"), []

        def handle_endtag(self, tag):  # noqa: ANN001
            if tag == "title":
                self.in_title = False
            if tag in self.SKIP and self.skip:
                self.skip -= 1
            if tag in ("main", "article") and self.main_depth:
                self.main_depth -= 1
            if tag in self.BLOCK:
                self.emit("\n")
            if tag == "a" and self.href:
                href = urljoin(final, self.href)
                pu = _up(href)
                if "duckduckgo.com" in pu.netloc and pu.path.startswith("/l/"):
                    href = (_pq(pu.query).get("uddg") or [href])[0]  # unwrap search-result redirects
                text = " ".join("".join(self.atext).split())
                if text and href.startswith("http") and len(self.links) < 400:
                    self.links.append({"text": text[:160], "url": href})
                self.href = None

        def handle_data(self, data):  # noqa: ANN001
            if self.in_title:
                self.title += data
            if self.skip:
                return
            self.emit(data)
            if self.href is not None:
                self.atext.append(data)

    rd = Reader()
    try:
        rd.feed(body)
    except Exception:  # noqa: BLE001  - broken markup still yields what was parsed
        pass
    main_text = "".join(rd.main)
    chosen = main_text if len(main_text.strip()) > 600 else "".join(rd.out)
    lines = [" ".join(ln.split()) for ln in chosen.split("\n")]
    text, blank = [], 0
    for ln in lines:
        if not ln or ln in ("##", "-"):
            blank += 1
            continue
        if blank and text:
            text.append("")
        blank = 0
        text.append(ln)
    seen, links = set(), []
    for lk in rd.links:
        if lk["url"] not in seen and "duckduckgo.com" not in _up(lk["url"]).netloc:
            seen.add(lk["url"])
            links.append(lk)
    return {"url": final, "title": " ".join(rd.title.split()) or final, "text": "\n".join(text)[:100_000],
            "links": links[:200], "content_type": ctype}


# --------------------------------------------------------------------------- #
# Self-update: install the newest MHexeract from GitHub (verified, backed up, reversible)
# --------------------------------------------------------------------------- #
UPDATE_REPO = os.getenv("MHEXERACT_REPO", "dconcretemick-netizen/MCUBED")
UPDATE_BRANCH = os.getenv("MHEXERACT_BRANCH", "main")
UPDATE_EXT = (".py", ".html", ".json", ".sh", ".md", ".txt")
UPDATE_SKIP = {"repos.json", "build_info.json", "package.json", "package-lock.json", "keys.json"}  # data, not program files
UPDATE_CONFIG = {"models.json", "news.json"}  # your edits here are kept (the new version is saved as *.new)
_update_cache: dict = {}


def _is_android() -> bool:
    return os.getenv("MHEXERACT_ANDROID") == "1"


def _channel() -> str:
    """Where this copy came from: "direct" (GitHub/sideload, self-updates)
    or "play" (Google Play, which handles updates and forbids self-install)."""
    return "play" if os.getenv("MHEXERACT_CHANNEL") == "play" else "direct"


def _gh_api(path: str, raw: bool = False, timeout: float = 30) -> Any:
    """GitHub API GET with GITHUB_TOKEN, else the signed-in GitHub CLI, else anonymous (public repos)."""
    import shutil
    import subprocess
    import urllib.error
    import urllib.request
    accept = "application/vnd.github.raw" if raw else "application/vnd.github+json"
    token = _github_token()
    if not token and shutil.which("gh"):
        try:
            out = subprocess.run(["gh", "api", "-H", f"Accept: {accept}", path], capture_output=True, timeout=timeout,
                                 check=True).stdout
            return out if raw else json.loads(out)
        except (subprocess.SubprocessError, OSError, json.JSONDecodeError):
            pass
    headers = {"Accept": accept, "User-Agent": "MHexeract", "X-GitHub-Api-Version": "2022-11-28"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        with urllib.request.urlopen(urllib.request.Request("https://api.github.com/" + path.lstrip("/"),
                                                           headers=headers), timeout=timeout) as r:
            data = r.read()
    except urllib.error.HTTPError as e:
        hint = " (GitHub may be rate limiting this network: add a GitHub token in AI keys or .env)" if e.code in (403, 429) and not token else ""
        raise GatewayError(f"GitHub returned HTTP {e.code} for {path.split('?')[0]}{hint}") from None
    except (urllib.error.URLError, OSError) as e:
        raise GatewayError(f"could not reach GitHub: {e}") from None
    return data if raw else json.loads(data)


def _git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _update_manifest_path() -> Path:
    return HERE / ".mhexeract_update.json"


def current_version() -> dict:
    """What is installed now. Android: the APK's build info. Desktop: git checkout, then the last
    self-update, then the build info shipped in the download (so a fresh zip knows its version)."""
    import subprocess

    def build_info() -> Optional[dict]:
        bi = HERE / "build_info.json"
        try:
            d = json.loads(bi.read_text()) if bi.exists() else None
        except (OSError, json.JSONDecodeError):
            return None
        return d and {"commit": d.get("commit", ""), "date": d.get("built_at", ""),
                      "source": "apk" if _is_android() else "build"}
    if _is_android():
        return build_info() or {"commit": "", "date": "", "source": "unknown"}
    if (HERE / ".git").exists():
        try:
            out = subprocess.run(["git", "-C", str(HERE), "log", "-1", "--format=%H%n%cI"], capture_output=True,
                                 text=True, timeout=10, check=True).stdout.split("\n")
            return {"commit": out[0], "date": out[1], "source": "git"}
        except (subprocess.SubprocessError, OSError, IndexError):
            pass
    mp = _update_manifest_path()
    if mp.exists():
        try:
            d = json.loads(mp.read_text())
            return {"commit": d.get("commit", ""), "date": d.get("date", ""), "source": "update"}
        except (OSError, json.JSONDecodeError):
            pass
    return build_info() or {"commit": "", "date": "", "source": "unknown"}


def check_update(force: bool = False) -> dict:
    hit = _update_cache.get("check")
    if hit and not force and time.time() - hit["checked_at"] < 900:
        return hit
    cur = current_version()
    head = _gh_api(f"repos/{UPDATE_REPO}/commits/{UPDATE_BRANCH}")
    latest = {"commit": head["sha"], "date": head["commit"]["committer"]["date"],
              "message": head["commit"]["message"].split("\n")[0]}
    changes: list[dict] = []
    behind = None
    if cur["commit"] and cur["commit"] != latest["commit"]:
        try:
            cmp_ = _gh_api(f"repos/{UPDATE_REPO}/compare/{cur['commit']}...{latest['commit']}")
            behind = cmp_.get("ahead_by")
            changes = [{"commit": c["sha"][:7], "date": c["commit"]["committer"]["date"],
                        "message": c["commit"]["message"].split("\n")[0]} for c in reversed(cmp_.get("commits", []))][:30]
        except GatewayError:
            pass
    elif cur["commit"] == latest["commit"]:
        behind = 0
    res = {"checked_at": time.time(), "repo": UPDATE_REPO, "branch": UPDATE_BRANCH, "current": cur, "latest": latest,
           "behind": behind, "available": cur["commit"] != latest["commit"], "changes": changes,
           "mode": "apk" if _is_android() else "git" if cur["source"] == "git" else "files",
           "backups": [b.name for b in _update_backups()]}
    if res["mode"] == "apk":
        res["apk"] = None
        try:
            rel = _gh_api(f"repos/{UPDATE_REPO}/releases/tags/latest")
            asset = next((a for a in rel.get("assets", []) if a["name"].endswith(".apk")), None)
            res["apk"] = {"commit": rel.get("target_commitish", ""), "name": asset["name"] if asset else "",
                          "size": asset["size"] if asset else 0, "page": rel.get("html_url", "")}
            res["available"] = bool(asset) and rel.get("target_commitish", "") != cur["commit"]
        except GatewayError as e:
            res["apk_error"] = str(e)
    _update_cache["check"] = res
    return res


def _update_backups() -> list[Path]:
    d = HERE / ".mhexeract_backup"
    return sorted((p for p in d.iterdir() if p.is_dir()), reverse=True) if d.exists() else []


# Only code signed with this certificate is accepted as an MHexeract APK update (SHA-256 of the DER certificate).
MCUBED_CERT_SHA256 = "53b57eb8251c6528061e05766f424c41ee1e94801a76a9f944f7c1a19dd9cccc"


def apk_signer_sha256(path) -> list[str]:
    """SHA-256 of each signing certificate in an APK's v2/v3 signature block (stdlib only)."""
    import hashlib
    import struct
    d = Path(path).read_bytes()
    eocd = d.rfind(b"PK\x05\x06")
    if eocd < 0:
        raise GatewayError("not an APK (zip end record missing)")
    cd = struct.unpack("<I", d[eocd + 16:eocd + 20])[0]
    if d[cd - 16:cd] != b"APK Sig Block 42":
        raise GatewayError("APK has no v2/v3 signature block")
    size = struct.unpack("<Q", d[cd - 24:cd - 16])[0]
    p, end, out = cd - size - 8 + 8, cd - 24, []

    def lp(b: bytes, o: int) -> tuple[bytes, int]:
        n = struct.unpack("<I", b[o:o + 4])[0]
        return b[o + 4:o + 4 + n], o + 4 + n
    while p < end:
        n = struct.unpack("<Q", d[p:p + 8])[0]
        bid = struct.unpack("<I", d[p + 8:p + 12])[0]
        if bid in (0x7109871A, 0xF05368C0):  # v2, v3
            signers, _ = lp(d[p + 12:p + 8 + n], 0)
            o = 0
            while o < len(signers):
                signer, o = lp(signers, o)
                signed, _ = lp(signer, 0)
                _, k = lp(signed, 0)
                certs, _ = lp(signed, k)
                c, _ = lp(certs, 0)
                out.append(hashlib.sha256(c).hexdigest())
        p += 8 + n
    return sorted(set(out))


def apk_signature_schemes(path) -> list[str]:
    """Which APK signature schemes are present: v1 (JAR), v2, v3."""
    import struct
    import zipfile
    found = []
    with zipfile.ZipFile(path) as z:
        if any(re.fullmatch(r"META-INF/[^/]+\.(RSA|EC|DSA)", n) for n in z.namelist()):
            found.append("v1")
    d = Path(path).read_bytes()
    cd = struct.unpack("<I", d[d.rfind(b"PK\x05\x06") + 16:][:4])[0]
    if d[cd - 16:cd] == b"APK Sig Block 42":
        size = struct.unpack("<Q", d[cd - 24:cd - 16])[0]
        p, end = cd - size, cd - 24
        while p < end:
            n = struct.unpack("<Q", d[p:p + 8])[0]
            bid = struct.unpack("<I", d[p + 8:p + 12])[0]
            found += {0x7109871A: ["v2"], 0xF05368C0: ["v3"], 0x1B93AD61: ["v3.1"]}.get(bid, [])
            p += 8 + n
    return found


def _check_lineage(current: str, new: str) -> None:
    """Refuse an update unless the new commit builds on the installed one (blocks rewritten history)."""
    if not current or not new or current == new:
        return
    try:
        cmp = _gh_api(f"repos/{UPDATE_REPO}/compare/{current}...{new}")
    except GatewayError as e:
        raise GatewayError(f"could not confirm the update continues from this version ({e}); not installing") from None
    if cmp.get("status") not in ("ahead", "identical"):
        raise GatewayError(f"update refused: {new[:7]} does not build on the installed {current[:7]} "
                           f"(history was {cmp.get('status')}). Someone may have rewritten main.")


def _gh_file(path: str, sha: str) -> bytes:
    """One file at a commit. Without a token, use raw.githubusercontent.com (no API rate limit for public repos)."""
    import shutil
    import urllib.request
    if _github_token() or not os.getenv("MHEXERACT_RAW_DOWNLOADS", "1") == "1" and shutil.which("gh"):
        return _gh_api(f"repos/{UPDATE_REPO}/contents/{path}?ref={sha}", raw=True, timeout=60)
    url = f"https://raw.githubusercontent.com/{UPDATE_REPO}/{sha}/{path}"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "MHexeract"}), timeout=60) as r:
            return r.read()
    except OSError:
        return _gh_api(f"repos/{UPDATE_REPO}/contents/{path}?ref={sha}", raw=True, timeout=60)


def apply_update() -> dict:
    """Install the newest files from GitHub. Every file is checked against GitHub's hash and the new
    gateway.py must compile before anything is replaced; the old files are backed up for rollback."""
    import shutil
    import subprocess
    if _is_android():
        raise GatewayError("in the Android app, updates install as a new APK: use Download and install")
    cur = current_version()
    if cur["source"] == "git":
        r = subprocess.run(["git", "-C", str(HERE), "pull", "--ff-only", "origin", UPDATE_BRANCH],
                           capture_output=True, text=True, timeout=180)
        if r.returncode:
            raise GatewayError("git pull failed: " + (r.stderr or r.stdout).strip()[-300:])
        _update_cache.clear()
        return {"mode": "git", "output": r.stdout.strip()[-1000:], "restart": True, **current_version()}
    head = _gh_api(f"repos/{UPDATE_REPO}/commits/{UPDATE_BRANCH}")
    sha = head["sha"]
    _check_lineage(cur.get("commit", ""), sha)
    tree = _gh_api(f"repos/{UPDATE_REPO}/git/trees/{sha}")
    files = [t for t in tree["tree"] if t["type"] == "blob" and "/" not in t["path"]
             and t["path"].endswith(UPDATE_EXT) and t["path"] not in UPDATE_SKIP and not t["path"].startswith(".")]
    manifest = json.loads(_update_manifest_path().read_text()) if _update_manifest_path().exists() else {}
    installed = manifest.get("files", {})
    staged: dict[str, bytes] = {}
    for t in files:
        data = _gh_file(t["path"], sha)
        if _git_blob_sha(data) != t["sha"]:
            raise GatewayError(f"{t['path']} failed its integrity check; nothing was changed")
        staged[t["path"]] = data
    if "gateway.py" in staged:
        compile(staged["gateway.py"], "gateway.py", "exec")  # SyntaxError aborts before any file is touched
    for name, data in staged.items():
        if name.endswith(".json"):
            json.loads(data)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    bdir = HERE / ".mhexeract_backup" / stamp
    bdir.mkdir(parents=True, exist_ok=True)
    changed, kept, same = [], [], []
    for name, data in staged.items():
        dst = HERE / name
        old = dst.read_bytes() if dst.exists() else None
        if old == data:
            same.append(name)
            continue
        if old is not None:
            (bdir / name).write_bytes(old)
        edited = old is not None and name in UPDATE_CONFIG and installed.get(name) not in (None, _git_blob_sha(old))
        if edited:  # you changed this config since the last update: keep yours, save ours beside it
            (HERE / (name + ".new")).write_bytes(data)
            kept.append(name)
            continue
        tmp = dst.with_name(dst.name + ".tmp")
        tmp.write_bytes(data)
        if old is not None:
            shutil.copymode(dst, tmp)
        os.replace(tmp, dst)
        changed.append(name)
    (bdir / "_info.json").write_text(json.dumps({"from": cur, "to": sha, "files": changed}))
    _update_manifest_path().write_text(json.dumps({"commit": sha, "date": head["commit"]["committer"]["date"],
                                                   "files": {n: _git_blob_sha(d) for n, d in staged.items()}}, indent=1))
    for old_b in _update_backups()[5:]:  # keep the five most recent backups
        shutil.rmtree(old_b, ignore_errors=True)
    if not any((bdir / n).exists() for n in changed):
        shutil.rmtree(bdir, ignore_errors=True)
    _update_cache.clear()
    return {"mode": "files", "commit": sha, "changed": changed, "kept_your_edits": kept, "unchanged": len(same),
            "backup": bdir.name if bdir.exists() else None, "restart": bool(changed)}


def rollback_update() -> dict:
    bs = _update_backups()
    if not bs:
        raise GatewayError("no backup to roll back to")
    b = bs[0]
    info = json.loads((b / "_info.json").read_text()) if (b / "_info.json").exists() else {}
    restored = []
    for f in b.iterdir():
        if f.name == "_info.json":
            continue
        os.replace(f, HERE / f.name)
        restored.append(f.name)
    prev = info.get("from") or {}
    mp = _update_manifest_path()
    if prev.get("commit") and mp.exists():
        m = json.loads(mp.read_text())
        m.update(commit=prev["commit"], date=prev.get("date", ""))
        mp.write_text(json.dumps(m, indent=1))
    __import__("shutil").rmtree(b, ignore_errors=True)
    _update_cache.clear()
    return {"restored": restored, "commit": prev.get("commit", ""), "restart": True}


def download_apk() -> dict:
    """Android: fetch the newest APK into the app's data folder; the app then opens the system installer."""
    import urllib.request
    rel = _gh_api(f"repos/{UPDATE_REPO}/releases/tags/latest")
    asset = next((a for a in rel.get("assets", []) if a["name"].endswith(".apk")), None)
    if not asset:
        raise GatewayError("the latest release has no APK yet")
    headers = {"Accept": "application/octet-stream", "User-Agent": "MHexeract"}
    if _github_token():
        headers["Authorization"] = f"Bearer {_github_token()}"
    out = Path(os.getenv("MHEXERACT_DATA", str(HERE))) / "update.apk"
    tmp = out.with_suffix(".part")
    with urllib.request.urlopen(urllib.request.Request(asset["url"], headers=headers), timeout=60) as r, open(tmp, "wb") as f:
        shutil_copy = __import__("shutil").copyfileobj
        shutil_copy(r, f, 1 << 16)
    if tmp.stat().st_size != asset["size"]:
        tmp.unlink(missing_ok=True)
        raise GatewayError("download was incomplete; try again")
    digest = (asset.get("digest") or "").removeprefix("sha256:")
    if digest and __import__("hashlib").sha256(tmp.read_bytes()).hexdigest() != digest:
        tmp.unlink(missing_ok=True)
        raise GatewayError("downloaded APK does not match GitHub's SHA-256; not installing")
    try:
        signers = apk_signer_sha256(tmp)
    except (GatewayError, OSError, ValueError) as e:
        tmp.unlink(missing_ok=True)
        raise GatewayError(f"could not read the APK signature: {e}") from None
    if signers != [MCUBED_CERT_SHA256]:
        tmp.unlink(missing_ok=True)
        raise GatewayError("APK is not signed with the MHexeract key; not installing")
    # The commit the APK was built from is the one the "latest" tag points at. The release's target_commitish
    # can be stale (it is not moved with the tag), so it is only a fallback.
    target = ""
    try:
        ref = _gh_api(f"repos/{UPDATE_REPO}/git/ref/tags/latest")
        obj = ref.get("object", {})
        if obj.get("type") == "tag":  # annotated tag: follow to the commit
            obj = _gh_api(f"repos/{UPDATE_REPO}/git/tags/{obj['sha']}").get("object", {})
        target = obj.get("sha", "")
    except GatewayError:
        target = rel.get("target_commitish", "")
    if re.fullmatch(r"[0-9a-f]{40}", target or ""):
        _check_lineage(current_version().get("commit", ""), target)
    os.replace(tmp, out)
    return {"path": str(out), "size": asset["size"], "name": asset["name"], "commit": target,
            "sha256": digest, "signer": MCUBED_CERT_SHA256[:16]}


def restart_process(delay: float = 0.8) -> None:
    """Re-exec this process so the newly installed files take effect (server sockets close on exec)."""
    import sys

    def go() -> None:
        time.sleep(delay)
        os.execv(sys.executable, [sys.executable] + sys.argv)
    threading.Thread(target=go, name="mhexeract-restart", daemon=True).start()


# --------------------------------------------------------------------------- #
# News: many outlets across the spectrum, grouped by story, with coverage balance
# --------------------------------------------------------------------------- #
NEWS_LEANS = ["Left", "Lean Left", "Center", "Lean Right", "Right", "Unrated"]
NEWS_SIDE = {"Left": "left", "Lean Left": "left", "Center": "center", "Lean Right": "right", "Right": "right"}
_NEWS_STOP = set("""a an and are as at be been but by for from has have he her his how in into is it its new news
not of on or our over says said she than that the their them they this to up us was were what when who why will with
you after about more amid could would should may can just out off against first two year years live latest update
watch video report reports here man woman people""".split())
# common words in political headlines: they count half when deciding two headlines describe the same event
_NEWS_GENERIC = set("""trump biden white house president administration democrat democratic republican gop senate
congress house state government court official officials country world nation federal leader minister police""".split())
_news_cache: dict = {}
_news_lock = threading.Lock()

NEWS_BRIEF_SYSTEM = (
    "You are a strictly neutral news analyst. You will receive headlines and summaries about one story from several "
    "outlets, each labeled with its outlet name and AllSides bias rating. Write a balanced brief. Rules: use only the "
    "material provided; attribute every claim to the outlet(s) that reported it; never take a side or use loaded "
    "words; separate facts from opinion; say plainly when something is unverified or when sources conflict. "
    "Format exactly:\n"
    "Summary: two or three neutral sentences.\n"
    "Agreed facts: bullet points reported consistently across outlets with differing ratings.\n"
    "Disputed or unverified: bullet points where outlets differ, with who says what.\n"
    "Emphasis by perspective: one line each for left-leaning, center and right-leaning outlets on what they emphasize "
    "or leave out (write 'no coverage in this set' when a group is missing).\n"
    "Missing context: what a reader should check next.")


def news_config(path: Optional[str] = None) -> dict:
    p = Path(path or os.getenv("GATEWAY_NEWS", "") or HERE / "news.json")
    if not p.exists():
        return {"feeds": [], "cache_seconds": 600, "max_age_hours": 48}
    cfg = json.loads(p.read_text())
    for f in cfg.get("feeds", []):
        if f.get("lean") not in NEWS_LEANS:
            f["lean"] = "Unrated"
    return cfg


def _strip_html(s: str) -> str:
    import html as _html
    return re.sub(r"\s+", " ", _html.unescape(re.sub(r"<[^>]+>", " ", s or ""))).strip()


def parse_feed(raw: bytes) -> list[dict]:
    """RSS 2.0, RSS 1.0 (RDF) and Atom -> [{title, link, summary, published(epoch|None)}]."""
    import xml.etree.ElementTree as ET
    from email.utils import parsedate_to_datetime
    root = ET.fromstring(raw)
    out = []

    def local(tag: str) -> str:
        return tag.rsplit("}", 1)[-1].lower()

    for el in root.iter():
        if local(el.tag) not in ("item", "entry"):
            continue
        d: dict = {"title": "", "link": "", "summary": "", "published": None}
        for c in el:
            t, txt = local(c.tag), (c.text or "").strip()
            if t == "title":
                d["title"] = _strip_html(txt)
            elif t == "link":
                href = c.get("href")
                if href and c.get("rel", "alternate") == "alternate":
                    d["link"] = href
                elif txt and not d["link"]:
                    d["link"] = txt
            elif t in ("description", "summary", "content", "encoded") and not d["summary"]:
                d["summary"] = _strip_html(txt)[:400]
            elif t in ("pubdate", "published", "updated", "date") and d["published"] is None and txt:
                try:
                    d["published"] = parsedate_to_datetime(txt).timestamp()
                except (TypeError, ValueError, IndexError):
                    try:
                        d["published"] = datetime.fromisoformat(txt.replace("Z", "+00:00")).timestamp()
                    except ValueError:
                        pass
        if d["title"] and d["link"].startswith(("http://", "https://")):
            out.append(d)
    return out


def _news_tokens(text: str) -> set[str]:
    toks = set()
    for w in re.findall(r"[a-z0-9][a-z0-9'-]+", text.lower()):
        w = w.strip("'-").removesuffix("'s")
        if len(w) < 3 or w in _NEWS_STOP or w.isdigit():
            continue
        if len(w) > 4 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        toks.add(w)
    return toks


def cluster_news(items: list[dict]) -> list[dict]:
    """Group headlines about the same event.

    An item joins a story only if it closely matches one member (shared headline words) and also shares a
    headline word with at least half of the members, so stories cannot chain into unrelated topics."""
    def weight(toks: set[str]) -> float:
        return sum(0.5 if t in _NEWS_GENERIC else 1.0 for t in toks)

    def match(a: dict, b: dict) -> float:
        st = weight(a["_tt"] & b["_tt"])
        sk = weight(a["_tk"] & b["_tk"])
        if st >= 2.5 or (st >= 2 and sk >= 5):
            return st + sk / 10
        return 0.0

    clusters: list[dict] = []
    for it in sorted(items, key=lambda x: -(x.get("published") or 0)):
        it = dict(it)
        it["_tt"] = _news_tokens(it["title"])
        it["_tk"] = it["_tt"] | _news_tokens(it["summary"][:300])
        best, best_s = None, 0.0
        for c in clusters:
            s_ = max(match(it, m) for m in c["items"])
            if s_ <= best_s:
                continue
            overlap = sum(1 for m in c["items"] if it["_tt"] & m["_tt"])
            if overlap * 2 >= len(c["items"]):
                best, best_s = c, s_
        if best is not None and any(i["source"] == it["source"] for i in best["items"]):
            continue  # same outlet, same story (a follow-up): one item per outlet keeps coverage counts honest
        if best is None:
            best = {"items": []}
            clusters.append(best)
        best["items"].append(it)
    for c in clusters:
        for i in c["items"]:
            i.pop("_tt", None)
            i.pop("_tk", None)
    out = []
    for c in clusters:
        its = c["items"]
        counts = {k: 0 for k in NEWS_LEANS}
        for i in its:
            counts[i["lean"]] += 1
        sides = {"left": 0, "center": 0, "right": 0}
        for i in its:
            if i["lean"] in NEWS_SIDE:
                sides[NEWS_SIDE[i["lean"]]] += 1
        rated = sum(sides.values())
        if rated >= 2 and sides["left"] and not sides["right"]:
            note = "Covered here only by left-leaning and center outlets" if sides["center"] else "Covered here only by left-leaning outlets"
        elif rated >= 2 and sides["right"] and not sides["left"]:
            note = "Covered here only by right-leaning and center outlets" if sides["center"] else "Covered here only by right-leaning outlets"
        elif sides["left"] and sides["right"]:
            note = "Covered across the spectrum"
        else:
            note = ""
        # headline shown first: the wording most shared with the other outlets (the common core of the story),
        # so no single outlet's framing leads; ties go to a Center outlet
        toks = [_news_tokens(i["title"]) for i in its]
        rep = its[max(range(len(its)), key=lambda k: (sum(len(toks[k] & t) for j, t in enumerate(toks) if j != k),
                                                      its[k]["lean"] == "Center", -len(its[k]["title"])))]
        key = hashlib.sha1("|".join(sorted(i["link"] for i in its)).encode()).hexdigest()[:10]
        out.append({"id": key, "title": rep["title"], "summary": rep["summary"], "n_sources": len(its),
                    "leans": counts, "sides": sides, "balance_note": note,
                    "published": max((i.get("published") or 0) for i in its) or None,
                    "items": sorted(its, key=lambda i: NEWS_LEANS.index(i["lean"]))})
    out.sort(key=lambda c: (-c["n_sources"], -(sum(1 for v in c["sides"].values() if v)), -(c["published"] or 0)))
    return out


def fetch_news(force: bool = False, config: Optional[str] = None, include: Optional[list[str]] = None,
               fetcher=None) -> dict:
    import concurrent.futures as cf
    import urllib.request
    from urllib.parse import urlparse as _up
    cfg = news_config(config)
    feeds = [f for f in cfg.get("feeds", []) if (f["name"] in include if include else f.get("enabled", True))]
    ttl = float(cfg.get("cache_seconds", 600))
    ck = json.dumps(sorted(f["name"] for f in feeds))
    with _news_lock:
        hit = _news_cache.get(ck)
        if hit and not force and time.time() - hit["generated_at"] < ttl:
            return hit

    def get(f: dict) -> tuple[dict, list[dict], str]:
        try:
            if fetcher:
                raw = fetcher(f["url"])
            else:
                host = _up(f["url"]).hostname or ""
                if not _public_host(host):
                    raise ValueError("private address blocked")
                req = urllib.request.Request(f["url"], headers={"User-Agent": WEB_UA, "Accept": "application/rss+xml, application/xml, text/xml, */*"})
                with urllib.request.urlopen(req, timeout=12) as r:
                    raw = r.read(5_000_000)
            return f, parse_feed(raw), ""
        except Exception as e:  # noqa: BLE001 - one bad feed must not break the page
            return f, [], f"{type(e).__name__}: {str(e)[:120]}"

    cutoff = time.time() - float(cfg.get("max_age_hours", 48)) * 3600
    items, status = [], []
    with cf.ThreadPoolExecutor(max_workers=12) as ex:
        for f, its, err in ex.map(get, feeds):
            fresh = [i for i in its if not i["published"] or i["published"] >= cutoff][:40]
            status.append({"name": f["name"], "lean": f["lean"], "rating_url": f.get("rating_url", ""),
                           "ok": not err, "items": len(fresh), "error": err})
            for i in fresh:
                items.append({**i, "source": f["name"], "lean": f["lean"], "rating_url": f.get("rating_url", "")})
    res = {"generated_at": time.time(), "clusters": cluster_news(items), "sources": status,
           "available": [{"name": f["name"], "lean": f["lean"], "enabled": f.get("enabled", True)} for f in cfg.get("feeds", [])],
           "ratings": "Bias ratings from AllSides (allsides.com/media-bias/ratings). Ratings change, and a Center "
                      "rating does not mean more accurate. Edit news.json to change sources or labels."}
    with _news_lock:
        _news_cache[ck] = res
    return res


def news_brief_messages(cluster: dict) -> list[dict]:
    lines = []
    for i in cluster.get("items", [])[:16]:
        lines.append(f"- {i.get('source')} [{i.get('lean')}]: {i.get('title')}\n  {str(i.get('summary', ''))[:400]}")
    return [{"role": "system", "content": NEWS_BRIEF_SYSTEM},
            {"role": "user", "content": "Story coverage:\n" + "\n".join(lines)}]


# --------------------------------------------------------------------------- #
# Wine: run Windows programs where Wine (or Box64 + Wine on ARM) is installed
# --------------------------------------------------------------------------- #
WINE_HINTS = {
    "debian": "sudo apt install wine", "ubuntu": "sudo apt install wine", "fedora": "sudo dnf install wine",
    "arch": "sudo pacman -S wine", "macos": "brew install --cask wine-stable",
    "termux": "Termux cannot run Wine directly. Install Winlator (https://winlator.org) for Windows x86_64 apps on Android 9+.",
    "android": "Install Winlator (https://winlator.org): Wine + Box64 for Windows apps on Android 9+, no root needed.",
    "windows": "You are on Windows: .exe files run natively.",
}


def wine_status() -> dict:
    import platform
    import shutil
    import subprocess
    system = platform.system().lower()
    is_android = "ANDROID_ROOT" in os.environ or "com.termux" in os.environ.get("PREFIX", "")
    found = {b: shutil.which(b) for b in ("wine", "wine64", "box64", "box86", "winetricks")}
    wine = found["wine64"] or found["wine"]
    version = ""
    if wine:
        try:
            version = subprocess.run([wine, "--version"], capture_output=True, text=True, timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    distro = "macos" if system == "darwin" else "windows" if system == "windows" else "linux"
    if is_android:
        distro = "termux" if "com.termux" in os.environ.get("PREFIX", "") else "android"
    elif distro == "linux" and Path("/etc/os-release").exists():
        osr = Path("/etc/os-release").read_text().lower()
        distro = next((d for d in ("ubuntu", "debian", "fedora", "arch") if d in osr), "linux")
    arm = platform.machine().lower() in ("aarch64", "arm64", "armv7l")
    return {"available": bool(wine) or system == "windows", "wine": wine, "version": version,
            "box64": found["box64"], "winetricks": found["winetricks"], "platform": distro, "arch": platform.machine(),
            "needs_box64": arm and not found["box64"] and distro not in ("macos", "windows"),
            "install_hint": WINE_HINTS.get(distro, "Install Wine from your distribution's package manager (see winehq.org).")}


def wine_run(exe: str, args: Optional[list[str]] = None, wait: bool = False) -> dict:
    import subprocess
    st = wine_status()
    p = Path(exe).expanduser()
    if not p.exists():
        raise ValueError(f"file not found: {exe}")
    if p.suffix.lower() not in (".exe", ".msi", ".bat", ".com"):
        raise ValueError("expected a Windows program (.exe, .msi, .bat or .com)")
    if st["platform"] == "windows":
        cmd = [str(p)] + list(args or [])
    elif not st["wine"]:
        raise GatewayError("Wine is not installed. " + st["install_hint"])
    else:
        cmd = ([st["wine"], "msiexec", "/i", str(p)] if p.suffix.lower() == ".msi" else [st["wine"], str(p)]) + list(args or [])
        if st["box64"] and st["arch"].lower() in ("aarch64", "arm64"):  # x86_64 Wine on ARM runs under Box64
            cmd = [st["box64"]] + cmd
    if wait:
        r = subprocess.run(cmd, capture_output=True, text=True)
        return {"cmd": cmd, "returncode": r.returncode, "stdout": r.stdout[-4000:], "stderr": r.stderr[-4000:]}
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    return {"cmd": cmd, "pid": proc.pid}

# --------------------------------------------------------------------------- #
# OpenAI-compatible API: MHexeract itself behaves like an AI model
# --------------------------------------------------------------------------- #
OPENAI_PASSTHROUGH = ("temperature", "top_p", "max_tokens", "max_completion_tokens", "stop", "seed",
                      "response_format", "presence_penalty", "frequency_penalty")
MHEXERACT_MODELS = {"mhexeract": "balanced", "mhexeract-auto": "balanced", "mhexeract-efficient": "efficient",
                 "mhexeract-quality": "quality"}


def openai_models(gw: "Gateway") -> dict:
    now = int(time.time())
    ids = list(MHEXERACT_MODELS) + [a for a, s in gw.models.items()
                                 if any(gw._route_available(s, r) for r in s.routes)]
    return {"object": "list", "data": [{"id": i, "object": "model", "created": now, "owned_by": "mhexeract"} for i in ids]}


def openai_chat(gw: "Gateway", body: dict) -> dict:
    """Serve an OpenAI /v1/chat/completions request through the gateway router.

    model "mhexeract" (or mhexeract-efficient / mhexeract-quality) lets the router choose; any pool alias pins that model.
    """
    msgs = body.get("messages")
    if not isinstance(msgs, list) or not msgs:
        raise ValueError("messages must be a non-empty list")
    clean = []
    for m in msgs:
        if not isinstance(m, dict) or m.get("role") not in ("system", "developer", "user", "assistant", "tool"):
            raise ValueError("each message needs a role of system, developer, user, assistant or tool")
        content = m.get("content")
        if isinstance(content, list):  # text parts only; other part types are dropped
            content = "\n".join(str(p.get("text", "")) for p in content if isinstance(p, dict) and p.get("type") == "text")
        clean.append({"role": "system" if m["role"] == "developer" else ("user" if m["role"] == "tool" else m["role"]),
                      "content": "" if content is None else str(content)})
    model = str(body.get("model") or "mhexeract")
    strategy, pinned = MHEXERACT_MODELS.get(model), None
    if strategy is None:
        if model not in gw.models:
            raise KeyError(f"unknown model '{model}'. Use 'mhexeract' or one of GET /v1/models")
        pinned = [model]
    kw = {k: body[k] for k in OPENAI_PASSTHROUGH if body.get(k) is not None}
    if "max_completion_tokens" in kw:
        kw["max_tokens"] = kw.pop("max_completion_tokens")
    r = gw.chat(clean, strategy=strategy, models=pinned, use_cache=not body.get("temperature"), **kw)
    return {
        "id": f"chatcmpl-{r.request_id}", "object": "chat.completion", "created": int(time.time()),
        "model": model, "system_fingerprint": f"mhexeract:{r.model_alias}",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": r.text},
                     "finish_reason": r.finish_reason or "stop"}],
        "usage": {"prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens,
                  "total_tokens": r.prompt_tokens + r.completion_tokens},
        "mhexeract": {k: v for k, v in r.summary().items() if k in (
            "request_id", "model_alias", "model", "via", "task_type", "tier_policy", "latency_ms", "attempts", "cached")},
    }


def openai_stream_chunks(resp: dict) -> list[bytes]:
    """Server-sent events for stream=true: role, content, then finish (the answer is produced in one piece)."""
    base = {k: resp[k] for k in ("id", "created", "model", "system_fingerprint")}
    base["object"] = "chat.completion.chunk"
    ch = resp["choices"][0]
    events = [dict(base, choices=[{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]),
              dict(base, choices=[{"index": 0, "delta": {"content": ch["message"]["content"]}, "finish_reason": None}]),
              dict(base, choices=[{"index": 0, "delta": {}, "finish_reason": ch["finish_reason"]}], usage=resp["usage"])]
    return [f"data: {json.dumps(e)}\n\n".encode() for e in events] + [b"data: [DONE]\n\n"]



# --------------------------------------------------------------------------- #
# AI provider keys: saved from the dashboard's AI keys tab (or the Android dialog)
# --------------------------------------------------------------------------- #
# (env name, label, where to get a key, free-plan?) — cloud nodes first, then the
# direct-route and GitHub keys that are not nodes themselves.
KEY_SLOTS = [(env, name, signup, nid in FREE_NODE_IDS)
             for nid, name, _base, env, signup, _path in CLOUD_NODES if env]
KEY_SLOTS += [("ANTHROPIC_API_KEY", "Anthropic Claude", "https://console.anthropic.com/settings/keys", False),
              ("GITHUB_TOKEN", "GitHub token (optional: repositories)", "https://github.com/settings/tokens", True)]
KEY_NAMES = [env for env, *_ in KEY_SLOTS]


def keys_path() -> Path:
    """Where saved keys live: the app's private data dir on Android, the gateway folder elsewhere."""
    return Path(os.getenv("MHEXERACT_DATA", str(HERE))) / "keys.json"


def load_saved_keys() -> int:
    """Apply keys saved from the AI keys tab to the environment. Returns how many are set."""
    try:
        saved = json.loads(keys_path().read_text())
    except (OSError, ValueError):
        saved = None
    if isinstance(saved, dict):
        for name in KEY_NAMES:
            value = str(saved.get(name) or "").strip()
            if value:
                os.environ[name] = value
            elif name in saved:  # present but empty means removed: do not resurrect it
                os.environ.pop(name, None)
    return sum(1 for n in KEY_NAMES if os.getenv(n))


def keys_status() -> dict:
    """Which key slots are set — never the keys themselves, only a last-4 hint."""
    out = []
    for env, name, signup, free in KEY_SLOTS:
        value = os.getenv(env) or ""
        out.append({"env": env, "name": name, "signup": signup, "free": free,
                    "set": bool(value), "hint": ("\u2026" + value[-4:]) if value else ""})
    return {"keys": out, "saved": sum(1 for k in out if k["set"])}


def save_keys(updates: dict) -> dict:
    """Persist key changes from the AI keys tab and apply them live.

    A non-empty value sets a key, an empty string removes it, and a name that is
    not mentioned is left as it is. Keys live only in keys.json on this device
    (0600, git-ignored, skipped by updates) and are never sent anywhere else."""
    if not isinstance(updates, dict):
        raise ValueError("keys object required")
    unknown = [k for k in updates if k not in KEY_NAMES]
    if unknown:
        raise ValueError(f"unknown key name: {unknown[0]}")
    try:
        saved = json.loads(keys_path().read_text())
        if not isinstance(saved, dict):
            saved = {}
    except (OSError, ValueError):
        saved = {}
    for name, raw in updates.items():
        value = str(raw or "").strip()
        if value:
            if len(value) > 512 or any(c.isspace() for c in value):
                raise ValueError(f"{name} does not look like an API key (check for spaces)")
            saved[name] = value
            os.environ[name] = value
        else:
            saved.pop(name, None)
            os.environ.pop(name, None)
    path = keys_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("keys.json.tmp")
    tmp.write_text(json.dumps(saved, indent=1))
    try:
        tmp.chmod(0o600)
    except OSError:
        pass
    tmp.replace(path)
    return keys_status()




# --------------------------------------------------------------------------- #
# Provider gardener: automated checking, updating and cleaning of API keys and
# providers, plus automatic adoption of free AI nodes.
# --------------------------------------------------------------------------- #
GARDENER_EVERY_HOURS = 6
STRIVE_EVERY_HOURS = 12
STRIVE_CANDIDATES = [  # anonymous/keyless providers the strive job watches (reported, never auto-enabled)
    {"id": "llm7", "name": "LLM7", "base": "https://api.llm7.io/v1"},
]
GARDENER_MAX_FAILS = 3  # consecutive failed checks before a key/provider is cleaned


def key_health_path() -> Path:
    """Where the gardener keeps key/provider health: next to keys.json."""
    return Path(os.getenv("MHEXERACT_DATA", str(HERE))) / "key_health.json"


def load_key_health() -> dict:
    try:
        h = json.loads(key_health_path().read_text())
        return h if isinstance(h, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_key_health(h: dict) -> None:
    path = key_health_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name("key_health.json.tmp")
        tmp.write_text(json.dumps(h, indent=1))
        try:
            tmp.chmod(0o600)
        except OSError:
            pass
        tmp.replace(path)
    except OSError:
        pass


def _http_status(url: str, key: Optional[str] = None, headers: Optional[dict] = None,
                 timeout: float = 8) -> Optional[int]:
    """HTTP status of a GET, or None when the host cannot be reached at all."""
    import urllib.error
    import urllib.request
    h = {"User-Agent": "MHexeract", "Accept": "application/json", **(headers or {})}
    if key:
        h["Authorization"] = f"Bearer {key}"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=timeout) as r:
            r.read(512)
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
    except Exception:  # noqa: BLE001 - DNS, TLS or timeout: the host is unreachable
        return None


def _classify(code: Optional[int]) -> str:
    if code is None:
        return "unreachable"
    if code == 200:
        return "ok"
    if code in (401, 403):
        return "invalid"
    if code == 429:
        return "limited"
    return "ok"  # any other reply still proves the endpoint is alive and answered


def _key_probe(env: str) -> str:
    """Check one saved key against its own provider."""
    key = os.getenv(env) or ""
    if not key:
        return "unset"
    if env == "ANTHROPIC_API_KEY":
        return _classify(_http_status("https://api.anthropic.com/v1/models", None,
                                      {"x-api-key": key, "anthropic-version": "2023-06-01"}))
    if env == "GITHUB_TOKEN":
        return _classify(_http_status("https://api.github.com/user", key))
    for _nid, _name, base, nenv, _signup, path in CLOUD_NODES:
        if nenv == env:
            return _classify(_http_status(base + (path or "/models"), key))
    return "ok"


def _provider_probe(node: tuple) -> str:
    """Is this provider reachable at all (and is its saved key the problem)?"""
    _nid, _name, base, env, _signup, path = node
    if env and node_key(env):
        return {"ok": "online", "limited": "online", "invalid": "key_invalid",
                "unreachable": "unreachable"}.get(_key_probe(env), "online")
    code = _http_status(base + (path or "/models"), None)
    if code is None:
        return "unreachable"
    return "online" if not env else "needs_key"  # keyless nodes are online when they answer


def run_gardener(gw: "Gateway", force: bool = False) -> Optional[dict]:
    """Check every saved key and provider; clean what is broken; adopt free nodes.

    Cleaning is deliberately conservative: only a definitive rejection counts.
    A key its provider rejects (401/403) on GARDENER_MAX_FAILS consecutive
    checks is removed — and only when it came from keys.json, never when it was
    supplied through the environment. A provider that cannot be reached at all
    on that many checks is parked (scans skip it) until a later check finds it
    alive again. Free and keyless nodes need no cleaning: the node scan that
    follows adopts their current models automatically.
    """
    every = float(os.getenv("GATEWAY_GARDENER_HOURS")
                  or gw.defaults.get("gardener_every_hours", GARDENER_EVERY_HOURS))
    last = float(gw.store.get_policy("last_gardener_ts", 0) or 0)
    if not force and every > 0 and time.time() - last < every * 3600:
        return None
    from concurrent.futures import ThreadPoolExecutor
    health = load_key_health()
    keys_h = health.setdefault("keys", {})
    prov_h = health.setdefault("providers", {})
    cleaned = health.setdefault("cleaned", [])
    try:
        saved = json.loads(keys_path().read_text())
        if not isinstance(saved, dict):
            saved = {}
    except (OSError, ValueError):
        saved = {}
    set_envs = [env for env in KEY_NAMES if os.getenv(env)]
    with ThreadPoolExecutor(max_workers=12) as ex:
        key_results = dict(zip(set_envs, ex.map(_key_probe, set_envs)))
        prov_results = dict(zip([n[0] for n in CLOUD_NODES], ex.map(_provider_probe, CLOUD_NODES)))
    now = time.time()
    removed = []
    for env, status in key_results.items():
        name = next((n for e, n, *_ in KEY_SLOTS if e == env), env)
        rec = keys_h.setdefault(env, {})
        if status == "invalid":
            rec["fails"] = int(rec.get("fails", 0)) + 1
        elif status in ("ok", "limited"):
            rec["fails"] = 0
        rec.update(name=name, status=status, checked_at=now)
        if status == "invalid" and rec["fails"] >= GARDENER_MAX_FAILS and env in saved:
            save_keys({env: ""})
            rec.update(status="removed", fails=0)
            entry = {"env": env, "name": name, "at": now,
                     "reason": "the provider rejected this key (401/403) on 3 checks in a row"}
            cleaned.append(entry)
            removed.append(entry)
            log.info("gardener: removed invalid key %s", env)
    health["cleaned"] = cleaned[-20:]
    dead = set(health.get("dead_nodes", []))
    for node in CLOUD_NODES:
        nid = node[0]
        status = prov_results.get(nid, "unreachable")
        rec = prov_h.setdefault(nid, {})
        if status == "unreachable":
            rec["fails"] = int(rec.get("fails", 0)) + 1
        elif status in ("online", "needs_key", "key_invalid"):
            rec["fails"] = 0
        rec.update(name=node[1], status=status, checked_at=now)
        if rec["fails"] >= GARDENER_MAX_FAILS and nid not in dead:
            dead.add(nid)
            log.info("gardener: provider %s unreachable on 3 checks; parked", nid)
        elif nid in dead and status != "unreachable":
            dead.discard(nid)
            log.info("gardener: provider %s answers again; unparked", nid)
    health["dead_nodes"] = sorted(dead)
    health["checked_at"] = now
    _save_key_health(health)
    gw.store.set_policy("last_gardener_ts", now)
    return {"checked_at": now, "keys": key_results, "providers": prov_results,
            "removed": removed, "dead_nodes": sorted(dead)}


def gardener_status(gw: "Gateway") -> dict:
    """Gardener summary for the Toolbox: toggle state plus the latest health."""
    health = load_key_health()
    nodes = load_nodes(gw.nodes_path)
    free_online = [n for n in nodes.get("nodes", []) if n.get("free") and n.get("status") == "online"]
    return {
        "on": bool(gw.store.get_policy("bg_gardener", True)),
        "every": float(os.getenv("GATEWAY_GARDENER_HOURS")
                       or gw.defaults.get("gardener_every_hours", GARDENER_EVERY_HOURS)),
        "unit": "hours",
        "last_at": float(gw.store.get_policy("last_gardener_ts", 0) or 0) or None,
        "keys": [{"env": e, "name": r.get("name", e), "status": r.get("status"),
                  "checked_at": r.get("checked_at")}
                 for e, r in sorted(health.get("keys", {}).items())],
        "providers_down": [r.get("name", nid) for nid, r in health.get("providers", {}).items()
                           if r.get("status") in ("unreachable", "key_invalid")],
        "dead_nodes": health.get("dead_nodes", []),
        "cleaned": health.get("cleaned", [])[-5:],
        "free_nodes_online": len(free_online),
        "free_models": sum(len(n.get("models", [])) for n in free_online),
    }


def _tiny_completion(base: str, model: str) -> bool:
    """One minimal keyless chat completion; True only if the provider answers."""
    try:
        out = _http_json(base.rstrip("/") + "/chat/completions", None, 25,
                         data={"model": model,
                               "messages": [{"role": "user", "content": "Reply with the word OK"}],
                               "max_tokens": 4})
        return bool(isinstance(out, dict) and out.get("choices"))
    except Exception:  # noqa: BLE001
        return False


def run_strive(gw: "Gateway", force: bool = False) -> Optional[dict]:
    """Strive toward the stated no-cost goals, automatically.

    The goals: real AI at no cost, keys for truly anonymous (keyless,
    no-account) models wherever they exist, and key maintenance that takes
    no effort. The gardener keeps saved keys healthy; this job covers the
    anonymous side. It proves each built-in keyless provider still answers
    a real completion (a model list alone can lie), and it watches
    candidate keyless providers — any that start answering are reported in
    the Toolbox, never enabled silently, so a human decides.
    """
    every = float(os.getenv("GATEWAY_STRIVE_HOURS")
                  or gw.defaults.get("strive_every_hours", STRIVE_EVERY_HOURS))
    last = float(gw.store.get_policy("last_strive_ts", 0) or 0)
    if not force and every > 0 and time.time() - last < every * 3600:
        return None
    now = time.time()
    health = load_key_health()
    st = health.setdefault("strive", {})
    keyless = {}
    for node in CLOUD_NODES:
        if node[3]:  # has a key env var: not an anonymous provider
            continue
        rec = _list_cloud(node, dict(NODE_DEFAULTS))
        models = [m["id"] for m in rec.get("models", [])]
        ok = bool(models) and _tiny_completion(node[2], models[0])
        keyless[node[0]] = {"name": node[1], "status": "ok" if ok else "down",
                            "model": models[0] if models else None, "checked_at": now}
    candidates = {}
    findings = list(st.get("findings", []))
    for cand in STRIVE_CANDIDATES:
        entry = {"name": cand["name"], "status": "unreachable", "checked_at": now}
        try:
            out = _http_json(cand["base"].rstrip("/") + "/models", None, 15)
            ids = [m.get("id", "") for m in (out or {}).get("data", []) if isinstance(m, dict)]
            ids = [i for i in ids if i and not _NON_CHAT.search(i)]
            entry["status"] = "listed_only"
            if ids and _tiny_completion(cand["base"], ids[0]):
                entry["status"] = "answers"
                if not any(f.get("id") == cand["id"] for f in findings):
                    findings.append({"id": cand["id"], "name": cand["name"], "base": cand["base"],
                                     "model": ids[0], "found_at": now})
                    log.info("strive: candidate %s now answers keyless completions", cand["id"])
        except Exception:  # noqa: BLE001
            pass
        candidates[cand["id"]] = entry
    st.update(checked_at=now, keyless=keyless, candidates=candidates, findings=findings[-5:])
    _save_key_health(health)
    gw.store.set_policy("last_strive_ts", now)
    return {"checked_at": now, "keyless": keyless, "candidates": candidates, "findings": st["findings"]}


def strive_status(gw: "Gateway") -> dict:
    """Strive summary for the Toolbox: the no-cost mission at a glance."""
    health = load_key_health()
    st = health.get("strive", {})
    nodes = load_nodes(gw.nodes_path)
    online = [n for n in nodes.get("nodes", []) if n.get("status") == "online"]
    return {
        "on": bool(gw.store.get_policy("bg_strive", True)),
        "every": float(os.getenv("GATEWAY_STRIVE_HOURS")
                       or gw.defaults.get("strive_every_hours", STRIVE_EVERY_HOURS)),
        "unit": "hours",
        "last_at": float(gw.store.get_policy("last_strive_ts", 0) or 0) or None,
        "keyless": st.get("keyless", {}),
        "candidates": st.get("candidates", {}),
        "findings": st.get("findings", []),
        "free_nodes_online": sum(1 for n in online if n.get("free")),
        "free_models": sum(len(n.get("models", [])) for n in online if n.get("free")),
    }


_GUI_STARTED_AT: float = 0.0
_GUI_SERVER = None  # the running ThreadingHTTPServer, so the Toolbox can shut it down


def shutdown_gateway(delay: float = 0.6) -> None:
    """Stop the GUI server for good. On desktop the process exits too; on Android
    the dashboard follows up with mhexeract://close-app and the app closes itself
    (stopping the keep-alive service and the process), so nothing is left running."""
    def go() -> None:
        time.sleep(delay)  # let the HTTP reply reach the dashboard first
        srv = _GUI_SERVER
        if srv is not None:
            try:
                srv.shutdown()
            except Exception:  # noqa: BLE001
                pass
            try:
                srv.server_close()
            except Exception:  # noqa: BLE001
                pass
        if not _is_android():
            os._exit(0)
    threading.Thread(target=go, name="mhexeract-shutdown", daemon=True).start()


# --------------------------------------------------------------------------- #
# Automated UI improvement for Android: every page the gateway serves is
# checked for the mobile polish below and gets it injected at serve time when
# it is missing (self-healing after updates), and a scheduled audit reports
# the state in the Toolbox. The polish itself is deliberately small and safe:
# finger-sized touch targets, no focus zoom on form fields, no tap flash,
# no overscroll chaining, and compact spacing on narrow screens.
# --------------------------------------------------------------------------- #
UI_POLISH_MARKER = "mhexeract-ui-polish"
UI_POLISH_VERSION = "mhexeract-ui-polish v2"
UI_POLISH_HTML = (
    "<style data-mhexeract-ui-polish>\n"
    "/* mhexeract-ui-polish v2: automated Android UI improvements, injected by the gateway */\n"
    "html{-webkit-tap-highlight-color:transparent}\n"
    "html,body{overflow-x:hidden;overflow-x:clip}\n"
    "body{overscroll-behavior-y:none}\n"
    "img,svg,video,canvas,iframe{max-width:100%}\n"
    "@media (pointer:coarse){\n"
    "  button,select,.btn,[role=tab]{min-height:48px}\n"
    "  input,textarea{font-size:16px}\n"
    "  input[type=checkbox][role=switch]{width:52px;height:30px}\n"
    "}\n"
    "@media (max-width:480px){\n"
    "  .card{padding:14px}\n"
    "  .row{gap:8px}\n"
    "}\n"
    "</style>"
)
UI_PAGES = ("dashboard.html", "oregon.html", "game.html", "solitaire.html",
            "tetris.html", "platformer.html", "blackjack.html")


def polish_html(body: bytes) -> bytes:
    """Add the automated Android UI polish (and any missing mobile basics) to a page.
    A page carrying an older built-in polish gets it refreshed to the current version."""
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError:
        return body
    if "</head>" not in text:
        return body
    add = ""
    if 'name="viewport"' not in text:
        add += '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
    elif "viewport-fit=cover" not in text:
        text = text.replace('initial-scale=1"', 'initial-scale=1, viewport-fit=cover"', 1)
    if 'name="theme-color"' not in text:
        add += '<meta name="theme-color" content="#0f1417">\n'
    if UI_POLISH_MARKER not in text and "mcubed-ui-polish" not in text:
        add += UI_POLISH_HTML + "\n"
    elif UI_POLISH_VERSION not in text:  # stale built-in polish: swap in the current one
        text = re.sub(r"<style data-(?:mcubed|mhexeract)-ui-polish>.*?</style>", UI_POLISH_HTML, text, count=1, flags=re.S)
    if add:
        text = text.replace("</head>", add + "</head>", 1)
    return text.encode("utf-8")


def ui_audit() -> dict:
    """Check each served page for the mobile basics and the UI polish."""
    pages = []
    for name in UI_PAGES:
        p = HERE / name
        if not p.exists():
            continue
        text = p.read_text(errors="replace")
        issues = []
        if 'name="viewport"' not in text:
            issues.append("no viewport meta")
        elif "viewport-fit=cover" not in text:
            issues.append("viewport misses viewport-fit=cover")
        if 'name="theme-color"' not in text:
            issues.append("no theme-color")
        pages.append({"page": name, "polished": UI_POLISH_MARKER in text,
                      "outdated": UI_POLISH_MARKER in text and UI_POLISH_VERSION not in text,
                      "issues": issues})
    return {"pages": pages,
            "auto_fixed": sum(1 for p in pages if not p["polished"] or p["outdated"] or p["issues"]),
            "checked_at": time.time()}


def run_ui_polish(gw: "Gateway", force: bool = False) -> Optional[dict]:
    """Scheduled audit (~daily). Nothing is rewritten on disk: pages that miss
    the polish get it injected by polish_html() every time they are served."""
    last = float(gw.store.get_policy("last_ui_audit_ts", 0) or 0)
    if not force and time.time() - last < 24 * 3600:
        return None
    audit = ui_audit()
    gw.store.set_policy("ui_audit", json.dumps(audit))
    gw.store.set_policy("last_ui_audit_ts", audit["checked_at"])
    log.info("ui polish audit: %d pages, %d auto-fixed at serve time",
             len(audit["pages"]), audit["auto_fixed"])
    return audit


def ui_status(gw: "Gateway") -> dict:
    audit = {}
    try:
        audit = json.loads(gw.store.get_policy("ui_audit", "") or "{}")
    except ValueError:
        audit = {}
    return {"on": bool(gw.store.get_policy("bg_ui_polish", True)),
            "every": 24, "unit": "hours",
            "last_at": float(gw.store.get_policy("last_ui_audit_ts", 0) or 0) or None,
            "pages": audit.get("pages", []),
            "auto_fixed": audit.get("auto_fixed", 0)}


def toolbox_status(gw: "Gateway") -> dict:
    """Toolbox view: is the app running, and what is it allowed to do in the background?"""
    def job(policy: str, every: float, unit: str, last_key: str) -> dict:
        return {"on": bool(gw.store.get_policy(policy, True)), "every": every, "unit": unit,
                "last_at": float(gw.store.get_policy(last_key, 0) or 0) or None}
    return {
        "running": True,
        "started_at": _GUI_STARTED_AT or None,
        "uptime_s": int(time.time() - _GUI_STARTED_AT) if _GUI_STARTED_AT else None,
        "android": _is_android(),
        "mock": bool(gw.mock),
        "keys_saved": keys_status()["saved"],
        "background": {
            "auto_improve": job("bg_auto_improve",
                                float(os.getenv("GATEWAY_AUTO_IMPROVE_HOURS")
                                      or gw.defaults.get("auto_improve_every_hours", 6)), "hours",
                                "last_scheduled_improve_ts"),
            "repo_sync": job("bg_repo_sync",
                             float(os.getenv("GATEWAY_REPO_SYNC_HOURS")
                                   or gw.defaults.get("repo_sync_every_hours", 6)), "hours",
                             "last_repo_sync_ts"),
            "node_scan": job("bg_node_scan",
                             float(nodes_config(gw).get("scan_every_minutes", 60)), "minutes",
                             "last_node_scan_ts"),
            "lan_scan": {"on": bool(nodes_config(gw).get("lan_scan", True))},
            "gardener": gardener_status(gw),
            "strive": strive_status(gw),
            "ui_polish": ui_status(gw),
            "obd_update": obd_data_status(gw),
        },
        "gardener": gardener_status(gw),
        "strive": strive_status(gw),
        "obd_data": obd_data_status(gw),
    }



# --------------------------------------------------------------------------- #
# OBD-II diagnostics: ELM327 adapters over Wi-Fi (TCP), Bluetooth, or demo.
# Diagnostics only: read/clear trouble codes and live data. No ECU flashing.
# --------------------------------------------------------------------------- #
class ObdError(Exception):
    pass


def _fuel_system_status(d: bytes) -> str:
    """PID 0x03: bit-coded open/closed loop state for each bank."""
    def one(b: int) -> str:
        if b & 0x02:
            return "closed loop"
        if b & 0x10:
            return "closed loop (fault)"
        if b & 0x08:
            return "open loop (failure)"
        if b & 0x04:
            return "open loop (load)"
        if b & 0x01:
            return "open loop (warming up)"
        return "—"
    out = one(d[0])
    if len(d) > 1 and d[1]:
        out += " · bank 2: " + one(d[1])
    return out


_OBD_STANDARDS = {1: "OBD-II (California ARB)", 2: "OBD (US EPA)", 3: "OBD-I and OBD-II",
                  4: "OBD-I", 5: "No OBD", 6: "EOBD (Europe)", 7: "EOBD and OBD-II",
                  8: "EOBD and OBD", 9: "EOBD, OBD and OBD-II", 10: "JOBD (Japan)",
                  11: "JOBD and OBD-II", 12: "JOBD and EOBD", 13: "JOBD, EOBD and OBD-II"}

_FUEL_TYPES = {1: "Gasoline", 2: "Methanol", 3: "Ethanol", 4: "Diesel", 5: "LPG",
               6: "CNG", 7: "Propane", 8: "Electric", 9: "Bifuel — gasoline",
               10: "Bifuel — methanol", 11: "Bifuel — ethanol", 12: "Bifuel — LPG",
               13: "Bifuel — CNG", 14: "Bifuel — propane", 15: "Bifuel — electric",
               16: "Hybrid gasoline", 17: "Hybrid ethanol", 18: "Hybrid diesel",
               19: "Hybrid electric", 20: "Hybrid running electric", 21: "Bifuel running electric"}


# pid -> (label, unit, decode bytes -> value, byte count). Standard SAE J1979
# Mode 01 formulas, cross-checked against the python-OBD project's decoders.
OBD_PIDS = {
    0x03: ("Fuel system", "", _fuel_system_status, 2),
    0x04: ("Engine load", "%", lambda d: d[0] * 100 / 255, 1),
    0x05: ("Coolant temp", "°C", lambda d: d[0] - 40, 1),
    0x06: ("Short fuel trim B1", "%", lambda d: (d[0] - 128) * 100 / 128, 1),
    0x07: ("Long fuel trim B1", "%", lambda d: (d[0] - 128) * 100 / 128, 1),
    0x08: ("Short fuel trim B2", "%", lambda d: (d[0] - 128) * 100 / 128, 1),
    0x09: ("Long fuel trim B2", "%", lambda d: (d[0] - 128) * 100 / 128, 1),
    0x0A: ("Fuel pressure", "kPa", lambda d: d[0] * 3, 1),
    0x0B: ("Intake pressure", "kPa", lambda d: d[0], 1),
    0x0C: ("RPM", "rpm", lambda d: (d[0] * 256 + d[1]) / 4, 2),
    0x0D: ("Speed", "km/h", lambda d: d[0], 1),
    0x0E: ("Timing advance", "°", lambda d: d[0] / 2 - 64, 1),
    0x0F: ("Intake air temp", "°C", lambda d: d[0] - 40, 1),
    0x10: ("MAF air flow", "g/s", lambda d: (d[0] * 256 + d[1]) / 100, 2),
    0x11: ("Throttle", "%", lambda d: d[0] * 100 / 255, 1),
    0x14: ("O2 voltage B1S1", "V", lambda d: d[0] / 200, 2),
    0x15: ("O2 voltage B1S2", "V", lambda d: d[0] / 200, 2),
    0x16: ("O2 voltage B1S3", "V", lambda d: d[0] / 200, 2),
    0x17: ("O2 voltage B1S4", "V", lambda d: d[0] / 200, 2),
    0x18: ("O2 voltage B2S1", "V", lambda d: d[0] / 200, 2),
    0x19: ("O2 voltage B2S2", "V", lambda d: d[0] / 200, 2),
    0x1A: ("O2 voltage B2S3", "V", lambda d: d[0] / 200, 2),
    0x1B: ("O2 voltage B2S4", "V", lambda d: d[0] / 200, 2),
    0x1C: ("OBD standard", "", lambda d: _OBD_STANDARDS.get(d[0], f"standard {d[0]}"), 1),
    0x1F: ("Run time", "s", lambda d: d[0] * 256 + d[1], 2),
    0x21: ("Distance with light on", "km", lambda d: d[0] * 256 + d[1], 2),
    0x22: ("Fuel rail pressure", "kPa", lambda d: (d[0] * 256 + d[1]) * 0.079, 2),
    0x23: ("Fuel rail pressure (gauge)", "kPa", lambda d: (d[0] * 256 + d[1]) * 10, 2),
    0x2C: ("Commanded EGR", "%", lambda d: d[0] * 100 / 255, 1),
    0x2D: ("EGR error", "%", lambda d: (d[0] - 128) * 100 / 128, 1),
    0x2E: ("Evap purge", "%", lambda d: d[0] * 100 / 255, 1),
    0x2F: ("Fuel level", "%", lambda d: d[0] * 100 / 255, 1),
    0x30: ("Warm-ups since codes cleared", "", lambda d: d[0], 1),
    0x31: ("Distance since codes cleared", "km", lambda d: d[0] * 256 + d[1], 2),
    0x32: ("Evap vapor pressure", "Pa", lambda d: ((d[0] * 256 + d[1]) - (65536 if d[0] & 0x80 else 0)) / 4, 2),
    0x33: ("Barometric pressure", "kPa", lambda d: d[0], 1),
    0x34: ("O2 ratio B1S1", "lambda", lambda d: (d[0] * 256 + d[1]) / 32768, 4),
    0x35: ("O2 ratio B1S2", "lambda", lambda d: (d[0] * 256 + d[1]) / 32768, 4),
    0x36: ("O2 ratio B1S3", "lambda", lambda d: (d[0] * 256 + d[1]) / 32768, 4),
    0x37: ("O2 ratio B1S4", "lambda", lambda d: (d[0] * 256 + d[1]) / 32768, 4),
    0x38: ("O2 ratio B2S1", "lambda", lambda d: (d[0] * 256 + d[1]) / 32768, 4),
    0x39: ("O2 ratio B2S2", "lambda", lambda d: (d[0] * 256 + d[1]) / 32768, 4),
    0x3A: ("O2 ratio B2S3", "lambda", lambda d: (d[0] * 256 + d[1]) / 32768, 4),
    0x3B: ("O2 ratio B2S4", "lambda", lambda d: (d[0] * 256 + d[1]) / 32768, 4),
    0x3C: ("Catalyst temp B1S1", "°C", lambda d: (d[0] * 256 + d[1]) / 10 - 40, 2),
    0x3D: ("Catalyst temp B2S1", "°C", lambda d: (d[0] * 256 + d[1]) / 10 - 40, 2),
    0x3E: ("Catalyst temp B1S2", "°C", lambda d: (d[0] * 256 + d[1]) / 10 - 40, 2),
    0x3F: ("Catalyst temp B2S2", "°C", lambda d: (d[0] * 256 + d[1]) / 10 - 40, 2),
    0x42: ("Battery voltage", "V", lambda d: (d[0] * 256 + d[1]) / 1000, 2),
    0x43: ("Absolute load", "%", lambda d: (d[0] * 256 + d[1]) * 100 / 255, 2),
    0x44: ("Fuel-air ratio", "lambda", lambda d: (d[0] * 256 + d[1]) / 32768, 2),
    0x45: ("Relative throttle", "%", lambda d: d[0] * 100 / 255, 1),
    0x46: ("Ambient temp", "°C", lambda d: d[0] - 40, 1),
    0x47: ("Throttle position B", "%", lambda d: d[0] * 100 / 255, 1),
    0x48: ("Throttle position C", "%", lambda d: d[0] * 100 / 255, 1),
    0x49: ("Accelerator pedal D", "%", lambda d: d[0] * 100 / 255, 1),
    0x4A: ("Accelerator pedal E", "%", lambda d: d[0] * 100 / 255, 1),
    0x4B: ("Accelerator pedal F", "%", lambda d: d[0] * 100 / 255, 1),
    0x4C: ("Throttle actuator", "%", lambda d: d[0] * 100 / 255, 1),
    0x4D: ("Time with light on", "min", lambda d: d[0] * 256 + d[1], 2),
    0x4E: ("Time since codes cleared", "min", lambda d: d[0] * 256 + d[1], 2),
    0x51: ("Fuel type", "", lambda d: _FUEL_TYPES.get(d[0], f"type {d[0]}"), 1),
    0x5A: ("Relative accelerator", "%", lambda d: d[0] * 100 / 255, 1),
    0x5B: ("Hybrid battery", "%", lambda d: d[0] * 100 / 255, 1),
    0x5C: ("Oil temp", "°C", lambda d: d[0] - 40, 1),
    0x5D: ("Fuel injection timing", "°", lambda d: (d[0] * 256 + d[1]) / 128 - 210, 2),
    0x5E: ("Engine fuel rate", "L/h", lambda d: (d[0] * 256 + d[1]) / 20, 2),
    0x61: ("Demand torque", "%", lambda d: d[0] - 125, 1),
    0x62: ("Engine torque", "%", lambda d: d[0] - 125, 1),
    0x63: ("Reference torque", "Nm", lambda d: d[0] * 256 + d[1], 2),
}
OBD_LIVE_PIDS = [0x0C, 0x0D, 0x05, 0x42, 0x04, 0x11, 0x2F, 0x10, 0x0F, 0x06, 0x07, 0x1F, 0x14, 0x5C]

# Trouble-code descriptions come from the open-source dtc-database project by
# Wal33D (MIT License, Copyright (c) 2024 Wal33D):
# https://github.com/Wal33D/dtc-database — 12,128 generic SAE J2012 and
# manufacturer-specific codes, bundled as dtc_codes.json (see
# dtc_codes.SOURCE.txt) and refreshed by the weekly check in run_obd_update().
# The small hand-written table below stays as a fallback for anything the
# database does not cover; database text wins where both have a code.
DTC_DESCRIPTIONS = {
    "P0101": "Mass air flow sensor range/performance problem",
    "P0102": "Mass air flow sensor low input",
    "P0113": "Intake air temperature sensor high input",
    "P0118": "Coolant temperature sensor high input",
    "P0128": "Coolant thermostat — engine running too cool",
    "P0131": "Oxygen sensor low voltage (bank 1, sensor 1)",
    "P0133": "Oxygen sensor slow response (bank 1, sensor 1)",
    "P0135": "Oxygen sensor heater fault (bank 1, sensor 1)",
    "P0141": "Oxygen sensor heater fault (bank 1, sensor 2)",
    "P0171": "System too lean (bank 1) — unmetered air or low fuel delivery",
    "P0172": "System too rich (bank 1)",
    "P0174": "System too lean (bank 2)",
    "P0175": "System too rich (bank 2)",
    "P0300": "Random/multiple cylinder misfire detected",
    "P0301": "Cylinder 1 misfire detected",
    "P0302": "Cylinder 2 misfire detected",
    "P0303": "Cylinder 3 misfire detected",
    "P0304": "Cylinder 4 misfire detected",
    "P0305": "Cylinder 5 misfire detected",
    "P0306": "Cylinder 6 misfire detected",
    "P0325": "Knock sensor fault (bank 1)",
    "P0335": "Crankshaft position sensor fault",
    "P0340": "Camshaft position sensor fault",
    "P0401": "EGR flow insufficient",
    "P0402": "EGR flow excessive",
    "P0411": "Secondary air injection — incorrect flow",
    "P0420": "Catalyst system efficiency below threshold (bank 1) — often a tired catalytic converter or O2 sensor",
    "P0430": "Catalyst system efficiency below threshold (bank 2)",
    "P0442": "Evaporative emissions — small leak detected",
    "P0446": "Evaporative emissions vent control fault",
    "P0455": "Evaporative emissions — large leak detected (check the gas cap first)",
    "P0456": "Evaporative emissions — very small leak detected",
    "P0500": "Vehicle speed sensor fault",
    "P0505": "Idle air control system fault",
    "P0507": "Idle speed higher than expected",
    "P0562": "System voltage low — check battery and alternator",
    "P0563": "System voltage high — check alternator/regulator",
    "P0601": "ECU memory checksum error",
    "P0700": "Transmission control system fault (see transmission codes)",
    "P0715": "Transmission input speed sensor fault",
    "P0730": "Incorrect gear ratio",
    "P2181": "Cooling system performance problem",
    "C0035": "Left front wheel speed sensor fault (ABS)",
    "C0040": "Right front wheel speed sensor fault (ABS)",
    "U0100": "Lost communication with the engine computer",
    "U0121": "Lost communication with the ABS module",
}


def decode_dtc(b1: int, b2: int) -> str:
    """Two raw bytes -> a DTC string like P0420."""
    return f"{'PCBU'[(b1 >> 6) & 3]}{(b1 >> 4) & 3}{(b1 & 0xF):X}{b2 >> 4:X}{(b2 & 0xF):X}"


def describe_dtc(code: str) -> str:
    if code in DTC_DATABASE:
        return DTC_DATABASE[code]
    if code in DTC_DESCRIPTIONS:
        return DTC_DESCRIPTIONS[code]
    if code.startswith("P03") and code[-1].isdigit() and code != "P0300":
        return f"Cylinder {code[-1]} misfire detected"
    family = {"P": "Powertrain", "C": "Chassis/ABS", "B": "Body", "U": "Network/communication"}.get(code[:1], "")
    return f"{family} trouble code — look up {code} in the service manual for this vehicle".strip()


# --------------------------------------------------------------------------- #
# DTC description database: bundled dtc_codes.json plus a weekly automated
# check of the upstream project (download-only; a failed or implausible
# download never replaces the database already in use).
# --------------------------------------------------------------------------- #
DTC_DB_URL = "https://raw.githubusercontent.com/Wal33D/dtc-database/main/data/dtc_codes.db"
DTC_DB_SOURCE = "Wal33D/dtc-database on GitHub (MIT License)"
DTC_DB_MAX_BYTES = 6 * 1024 * 1024
_DTC_CODE_RE = re.compile(r"[PCBU][0-9A-F]{4}")

DTC_DATABASE: dict = {}
DTC_DB_INFO: dict = {"source": "built-in table", "count": 0, "bundled_count": 0, "total": 0}


def dtc_bundled_path() -> Path:
    return HERE / "dtc_codes.json"


def dtc_live_path() -> Path:
    """Where a downloaded database lives: the same data dir as the saved keys."""
    return Path(os.getenv("MHEXERACT_DATA", str(HERE))) / "dtc_codes.json"


def _read_dtc_json(path: Path) -> dict:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return {k: v.strip() for k, v in raw.items()
            if isinstance(k, str) and _DTC_CODE_RE.fullmatch(k)
            and isinstance(v, str) and v.strip()}


def load_dtc_database() -> dict:
    """(Re)load the trouble-code database. A downloaded copy in the data dir
    wins over the bundled dtc_codes.json; both win over the built-in table,
    which (with the pattern fallbacks) covers anything they miss."""
    global DTC_DATABASE, DTC_DB_INFO
    bundled = _read_dtc_json(dtc_bundled_path())
    live = _read_dtc_json(dtc_live_path()) if dtc_live_path() != dtc_bundled_path() else {}
    merged = dict(DTC_DESCRIPTIONS)
    source, count = "built-in table", 0
    if bundled:
        merged.update(bundled)
        source, count = "bundled", len(bundled)
    if live:
        merged.update(live)
        source, count = "downloaded", len(live)
    DTC_DATABASE = merged
    DTC_DB_INFO = {"source": source, "count": count,
                   "bundled_count": len(bundled), "total": len(merged)}
    return DTC_DB_INFO


def _dtc_map_from_sqlite(raw: bytes) -> dict:
    """Extract a code -> description map from an upstream dtc-database SQLite
    file. Generic (SAE) definitions win; for manufacturer-only codes the most
    common definition across brands is used."""
    import tempfile
    from collections import Counter
    fd, tmp = tempfile.mkstemp(suffix=".db")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(raw)
        con = sqlite3.connect(f"file:{tmp}?mode=ro", uri=True)
        try:
            rows = con.execute("SELECT code, manufacturer, description, is_generic "
                               "FROM dtc_definitions WHERE locale='en'").fetchall()
        finally:
            con.close()
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass
    generic, votes = {}, {}
    for code, maker, desc, is_gen in rows:
        code = (code or "").strip().upper()
        desc = " ".join((desc or "").split())
        if not _DTC_CODE_RE.fullmatch(code) or not desc:
            continue
        if is_gen or (maker or "").strip().upper() == "GENERIC":
            generic.setdefault(code, desc)
        else:
            votes.setdefault(code, Counter())[desc] += 1
    out = dict(generic)
    for code, counter in votes.items():
        if code in out:
            continue
        short = {d: n for d, n in counter.items() if len(d) <= 200} or counter
        out[code] = sorted(short.items(), key=lambda kv: (-kv[1], len(kv[0]), kv[0]))[0][0]
    return out


def _download_dtc_db() -> bytes:
    import urllib.request
    req = urllib.request.Request(DTC_DB_URL, headers={"User-Agent": "MHexeract"})
    with urllib.request.urlopen(req, timeout=30) as r:
        chunks, total = [], 0
        while True:
            chunk = r.read(1 << 16)
            if not chunk:
                break
            total += len(chunk)
            if total > DTC_DB_MAX_BYTES:
                raise ValueError(f"upstream database is larger than {DTC_DB_MAX_BYTES} bytes")
            chunks.append(chunk)
    raw = b"".join(chunks)
    if not raw.startswith(b"SQLite format 3\x00"):
        raise ValueError("upstream file is not a SQLite database")
    return raw


def run_obd_update(gw: "Gateway", force: bool = False) -> Optional[dict]:
    """Weekly check for a newer trouble-code database at the upstream project.
    The download is converted and validated (plausible size, at least as many
    codes as the bundled database) before it replaces anything; any failure
    keeps the database already in use."""
    days = float(os.getenv("GATEWAY_OBD_UPDATE_DAYS") or 7)
    last = float(gw.store.get_policy("last_obd_update_ts", 0) or 0)
    if not force and time.time() - last < days * 86400:
        return None
    now = time.time()
    try:
        new = _dtc_map_from_sqlite(_download_dtc_db())
        bundled = DTC_DB_INFO.get("bundled_count") or 0
        if not 5000 <= len(new) <= 60000:
            raise ValueError(f"upstream database has an implausible code count ({len(new)})")
        if bundled and len(new) < bundled:
            raise ValueError(f"upstream database has fewer codes ({len(new)}) than the "
                             f"bundled one ({bundled}); keeping the bundled one")
        target = dtc_live_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.parent / (target.name + ".tmp")
        tmp.write_text(json.dumps(new, ensure_ascii=False, separators=(",", ":"), sort_keys=True),
                       encoding="utf-8")
        os.replace(tmp, target)
        load_dtc_database()
        res = {"ok": True, "at": now, "codes": len(new)}
        log.info("OBD code database updated: %d codes from %s", len(new), DTC_DB_URL)
    except Exception as e:  # noqa: BLE001 - any failure keeps the current database
        res = {"ok": False, "at": now, "error": f"{type(e).__name__}: {e}"[:200]}
        log.info("OBD code database check failed: %s", res["error"])
    gw.store.set_policy("last_obd_update_ts", now)
    gw.store.set_policy("obd_update_result", json.dumps(res))
    return res


def obd_data_status(gw: "Gateway") -> dict:
    """OBD code-database summary for the Toolbox and the OBD tab."""
    result = {}
    try:
        result = json.loads(gw.store.get_policy("obd_update_result", "") or "{}")
    except ValueError:
        result = {}
    return {"on": bool(gw.store.get_policy("bg_obd_update", True)),
            "every": float(os.getenv("GATEWAY_OBD_UPDATE_DAYS") or 7), "unit": "days",
            "last_at": float(gw.store.get_policy("last_obd_update_ts", 0) or 0) or None,
            "source": DTC_DB_SOURCE, "url": DTC_DB_URL,
            "codes": DTC_DB_INFO.get("count", 0),
            "bundled_codes": DTC_DB_INFO.get("bundled_count", 0),
            "database": DTC_DB_INFO.get("source", "built-in table"),
            "last_result": result}


load_dtc_database()


def _hex_frames(raw: str) -> list:
    """ELM327 text -> list of byte strings, one per clean hex line."""
    out = []
    for line in re.split(r"[\r\n]+", raw or ""):
        t = re.sub(r"[^0-9A-Fa-f]", "", line)
        if len(t) >= 2 and len(t) % 2 == 0:
            try:
                out.append(bytes.fromhex(t))
            except ValueError:
                pass
    return out


class _TcpTransport:
    def __init__(self, host: str, port: int):
        self.host, self.port, self.sock = host, port, None

    def open(self):
        import socket
        self.sock = socket.create_connection((self.host, self.port), timeout=6)
        self.sock.settimeout(0.4)

    def exchange(self, cmd: str, timeout: float = 5.0) -> str:
        import socket
        self.sock.sendall((cmd + "\r").encode())
        buf, end = b"", time.time() + timeout
        while time.time() < end and b">" not in buf:
            try:
                chunk = self.sock.recv(1024)
                if chunk:
                    buf += chunk
            except socket.timeout:
                pass
            except OSError as e:
                raise ObdError(f"adapter connection lost ({e})") from None
        return buf.decode("latin-1").replace(">", " ")

    def close(self):
        try:
            if self.sock:
                self.sock.close()
        except OSError:
            pass
        self.sock = None


class _RfcommTransport:
    """Bluetooth SPP on desktop Linux (Python AF_BLUETOOTH)."""

    def __init__(self, address: str):
        self.address, self.sock = address, None

    def open(self):
        import socket
        if not hasattr(socket, "AF_BLUETOOTH"):
            raise ObdError("Bluetooth sockets are not available on this system — use a Wi-Fi adapter")
        self.sock = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM)
        self.sock.settimeout(8)
        self.sock.connect((self.address, 1))
        self.sock.settimeout(0.4)

    exchange = _TcpTransport.exchange
    close = _TcpTransport.close


class _AndroidBtTransport:
    """Bluetooth SPP on Android, bridged to Java (ObdBridge) by Chaquopy."""

    def __init__(self, address: str):
        self.address, self.bridge = address, None

    def open(self):
        try:
            from java import jclass
            self.bridge = jclass("com.mhexeract.app.ObdBridge")
        except Exception:
            raise ObdError("Bluetooth bridge is not available in this app build") from None
        if not self.bridge.connect(self.address):
            raise ObdError("Bluetooth connect failed — is the adapter paired, in range, and Bluetooth on?")

    def exchange(self, cmd: str, timeout: float = 5.0) -> str:
        return str(self.bridge.exchange(cmd, int(timeout * 1000)))

    def close(self):
        try:
            if self.bridge:
                self.bridge.close()
        except Exception:  # noqa: BLE001
            pass


class _DemoTransport:
    """A simulated car + ELM327, so the OBD tab works with no adapter at all."""

    def __init__(self):
        self.t0 = time.time()
        self.dtcs = [0x0420, 0x0171]
        self.pending = [0x0300]

    def open(self):
        pass

    def close(self):
        pass

    def _pid01(self) -> bytes:
        a = (0x80 if self.dtcs else 0x00) | len(self.dtcs)
        if getattr(self, "just_cleared", False) and not self.dtcs:
            return bytes([a, 0x07, 0xEF, 0xEF])  # just cleared: no monitor has run yet
        return bytes([a, 0x07, 0xEF, 0x04])       # normally: only the EVAP monitor still pending

    def _sim(self) -> dict:
        t = time.time() - self.t0
        rpm = 1500 + 850 * math.sin(t / 17) + 260 * math.sin(t / 5.3)
        return {
            0x01: self._pid01(),
            0x04: bytes([int(28 + 18 * math.sin(t / 17)) * 255 // 100]),
            0x05: bytes([int(min(92, 22 + t / 3)) + 40]),
            0x06: bytes([128 + int(6 * math.sin(t / 9))]),
            0x07: bytes([128 + int(9 * math.sin(t / 23))]),
            0x0B: bytes([38]),
            0x0C: (lambda v: bytes([int(v) >> 8, int(v) & 0xFF]))(max(0, rpm) * 4),
            0x0D: bytes([int(max(0, (rpm - 1000) / 26))]),
            0x0E: bytes([int((12 + 4 * math.sin(t / 7)) * 2 + 128)]),
            0x0F: bytes([24 + 40]),
            0x10: (lambda v: bytes([int(v) >> 8, int(v) & 0xFF]))((4 + rpm / 700) * 100),
            0x11: bytes([int(22 + 16 * math.sin(t / 17)) * 255 // 100]),
            0x1F: (lambda v: bytes([int(v) >> 8, int(v) & 0xFF]))(min(65535, int(t) + 640)),
            0x2F: bytes([63 * 255 // 100]),
            0x33: bytes([101]),
            0x42: (lambda v: bytes([int(v) >> 8, int(v) & 0xFF]))(13800 + int(300 * math.sin(t / 11))),
            0x46: bytes([24 + 40]),
        }

    def exchange(self, cmd: str, timeout: float = 5.0) -> str:
        c = cmd.strip().upper()
        if c.startswith("AT"):
            if c == "ATZ":
                return "ELM327 v1.5"
            if c == "ATDP":
                return "ISO 15765-4 (CAN 11/500)"
            if c == "ATRV":
                return "13.9V"
            return "OK"
        if c == "0100":
            return "4100BE1FA813"
        if c == "0120":
            return "412080022001"
        if c == "0140":
            return "414040084000"
        sim = self._sim()
        if re.fullmatch(r"01[0-9A-F]{2}", c):
            pid = int(c[2:4], 16)
            return "41" + c[2:4] + sim[pid].hex().upper() if pid in sim else "NO DATA"
        if c == "03":
            body = "".join(f"{d:04X}" for d in self.dtcs) or "0000"
            return "43" + body
        if c == "07":
            body = "".join(f"{d:04X}" for d in self.pending) or "0000"
            return "47" + body
        if c == "020200":
            return "420200" + (f"{self.dtcs[0]:04X}" if self.dtcs else "0000")
        if re.fullmatch(r"02[0-9A-F]{2}00", c):
            pid = int(c[2:4], 16)
            return "42" + c[2:4] + "00" + sim[pid].hex().upper() if pid in sim else "NO DATA"
        if c == "0902":
            vin = "1HGCM82633A123456"
            lines = []
            for i in range(0, len(vin), 4):
                chunk = vin[i:i + 4].encode().hex().upper()
                lines.append(f"4902{i // 4 + 1:02X}{chunk}")
            return "\r".join(lines)
        if c == "0A":
            return "4A0420"  # P0420 stays permanent until the car itself verifies the repair
        if c == "04":
            self.dtcs, self.pending = [], []
            self.just_cleared = True
            return "44"
        return "?"


class ObdSession:
    """One ELM327 connection, serialized. All reads go through the lock."""

    def __init__(self):
        self.lock = threading.Lock()
        self.transport = None
        self.kind = ""
        self.target = ""
        self.adapter = ""
        self.protocol = ""
        self.vin = ""
        self.supported: set = set()
        self.last_error = ""
        self.connected_at = 0.0
        self._mon_cache: tuple = (0.0, {})

    # -- connection -------------------------------------------------------- #
    def status(self) -> dict:
        return {"connected": self.transport is not None, "transport": self.kind,
                "target": self.target, "adapter": self.adapter, "protocol": self.protocol,
                "vin": self.vin, "demo": self.kind == "demo",
                "last_error": self.last_error,
                "connected_at": self.connected_at or None}

    def _cmd(self, cmd: str, timeout: float = 5.0) -> str:
        raw = self.transport.exchange(cmd, timeout)
        if "STOPPED" in raw:  # interrupted by a previous long op: retry once
            raw = self.transport.exchange(cmd, timeout)
        return raw

    def connect(self, kind: str, host: str = "", port: int = 35000, address: str = "") -> dict:
        with self.lock:
            self.disconnect()
            try:
                if kind == "demo":
                    t = _DemoTransport()
                    target = "simulated car"
                elif kind == "wifi":
                    t = _TcpTransport(host or "192.168.0.10", int(port or 35000))
                    target = f"{host or '192.168.0.10'}:{int(port or 35000)}"
                elif kind == "bluetooth":
                    if not address:
                        raise ObdError("pick a paired Bluetooth adapter first")
                    t = _AndroidBtTransport(address) if _is_android() else _RfcommTransport(address)
                    target = address
                else:
                    raise ObdError("transport must be wifi, bluetooth or demo")
                t.open()
                banner = ""
                try:
                    banner = t.exchange("ATZ", 3)
                except Exception:  # noqa: BLE001 - some clones answer slowly; setup below still works
                    pass
                m = re.search(r"ELM327[^\r\n]*", banner or "")
                self.adapter = m.group(0).strip() if m else "ELM327 adapter"
                for cmd in ("ATE0", "ATL0", "ATS0", "ATH0", "ATSP0"):
                    try:
                        t.exchange(cmd, 2)
                    except Exception:  # noqa: BLE001
                        pass
                self.transport, self.kind, self.target = t, kind, target
                probe = self._cmd("0100", 8)
                if "UNABLE TO CONNECT" in probe:
                    raise ObdError("The adapter answers, but cannot talk to the car. Is the ignition on?")
                if "NO DATA" in probe and kind != "demo":
                    raise ObdError("The car did not answer. Check the adapter is fully plugged in and the ignition is on.")
                self.supported = self._read_supported()
                try:
                    self.protocol = self._cmd("ATDP", 2).strip().split("\r")[0].strip()
                except Exception:  # noqa: BLE001
                    self.protocol = ""
                self.vin = self._read_vin()
                self.connected_at = time.time()
                self.last_error = ""
                return self.status()
            except Exception as e:  # noqa: BLE001
                try:
                    t.close()
                except Exception:  # noqa: BLE001
                    pass
                self.disconnect()
                self.last_error = str(e) if isinstance(e, ObdError) else f"{type(e).__name__}: {e}"
                raise ObdError(self.last_error) from None

    def disconnect(self):
        if self.transport is not None:
            try:
                self.transport.close()
            except Exception:  # noqa: BLE001
                pass
        self.transport, self.kind, self.target = None, "", ""
        self.adapter = self.protocol = self.vin = ""
        self.supported = set()
        self.connected_at = 0.0

    # -- protocol helpers --------------------------------------------------- #
    def _frame_for(self, raw: str, first: int, second: Optional[int] = None) -> Optional[bytes]:
        for f in _hex_frames(raw):
            if f and f[0] == first and (second is None or (len(f) >= 2 and f[1] == second)):
                return f
        return None

    def _read_supported(self) -> set:
        supported = set()
        for base in (0x00, 0x20, 0x40):
            f = self._frame_for(self._cmd(f"01{base:02X}", 5), 0x41, base)
            if not f or len(f) < 6:
                break
            bits = int.from_bytes(f[2:6], "big")
            for i in range(32):
                if bits & (1 << (31 - i)):
                    supported.add(base + i + 1)
            if base + 0x20 not in supported:
                break
        return supported

    def _read_pid(self, pid: int, mode: int = 1, frame: int = 0) -> Optional[bytes]:
        cmd = f"{mode:02X}{pid:02X}" + (f"{frame:02X}" if mode == 2 else "")
        raw = self._cmd(cmd, 5)
        if "NO DATA" in raw or "ERROR" in raw:
            return None
        f = self._frame_for(raw, 0x40 + mode, pid)
        if f is None:
            return None
        return f[3:] if mode == 2 else f[2:]

    def read_value(self, pid: int) -> Optional[dict]:
        name, unit, fn, _n = OBD_PIDS[pid]
        data = self._read_pid(pid)
        if data is None:
            return None
        try:
            v = fn(data)
        except (IndexError, ZeroDivisionError):
            return None
        return {"pid": f"{pid:02X}", "name": name, "value": round(v, 2) if isinstance(v, float) else v, "unit": unit}

    def monitor_status(self) -> dict:
        if self._mon_cache[0] and time.time() - self._mon_cache[0] < 5:
            return self._mon_cache[1]
        out = {"mil_on": None, "dtc_count": None}
        data = self._read_pid(0x01)
        if data:
            out = {"mil_on": bool(data[0] & 0x80), "dtc_count": data[0] & 0x7F}
        self._mon_cache = (time.time(), out)
        return out

    def live(self) -> dict:
        if self.transport is None:
            raise ObdError("not connected")
        with self.lock:
            values = {}
            for pid in OBD_LIVE_PIDS:
                if self.supported and pid not in self.supported:
                    continue
                got = self.read_value(pid)
                if got:
                    values[got["pid"]] = got
            return {"at": time.time(), "values": values, **self.monitor_status()}

    def _read_codes(self, mode: int, first: int) -> list:
        raw = self._cmd(f"{mode:02X}", 6)
        codes = []
        for f in _hex_frames(raw):
            if not f or f[0] != first:
                continue
            body = f[1:]
            for i in range(0, len(body) - 1, 2):
                v = (body[i] << 8) | body[i + 1]
                if v:
                    code = decode_dtc(body[i], body[i + 1])
                    codes.append({"code": code, "description": describe_dtc(code)})
        return codes

    def codes(self) -> dict:
        if self.transport is None:
            raise ObdError("not connected")
        with self.lock:
            stored = self._read_codes(3, 0x43)
            pending = self._read_codes(7, 0x47)
            freeze = {}
            fdtc = self._read_pid(0x02, mode=2)
            if fdtc and len(fdtc) >= 2:
                code = decode_dtc(fdtc[0], fdtc[1])
                freeze["dtc"] = {"code": code, "description": describe_dtc(code)}
                for pid in (0x0C, 0x0D, 0x05):
                    data = self._read_pid(pid, mode=2)
                    if data:
                        name, unit, fn, _n = OBD_PIDS[pid]
                        try:
                            freeze[name] = f"{round(fn(data), 1)} {unit}"
                        except (IndexError, ZeroDivisionError):
                            pass
            return {"stored": stored, "pending": pending, "freeze": freeze,
                    **self.monitor_status()}

    _CONT_MONITORS = ("Misfire", "Fuel system", "Comprehensive components")
    _NONCONT_MONITORS = ("Catalyst", "Heated catalyst", "Evaporative system", "Secondary air system",
                         "A/C refrigerant", "Oxygen sensor", "Oxygen sensor heater", "EGR/VVT system")

    def emissions(self) -> dict:
        """The inspection-station check (I/M readiness): MIL state, stored /
        pending / permanent codes, and whether every emissions monitor has
        completed since the codes were last cleared (PID 0x01)."""
        if self.transport is None:
            raise ObdError("not connected")
        with self.lock:
            data = self._read_pid(0x01)
            if not data or len(data) < 4:
                raise ObdError("the car did not answer the readiness check (PID 01)")
            a, b, c, d = data[0], data[1], data[2], data[3]
            mil, count = bool(a & 0x80), a & 0x7F
            monitors = []
            for i, name in enumerate(self._CONT_MONITORS):
                if b & (1 << i):
                    monitors.append({"name": name, "ready": not (b & (1 << (4 + i)))})
            for i, name in enumerate(self._NONCONT_MONITORS):
                if c & (1 << i):
                    monitors.append({"name": name, "ready": not (d & (1 << i))})
            stored = self._read_codes(3, 0x43)
            pending = self._read_codes(7, 0x47)
            permanent = self._read_codes(0x0A, 0x4A)
            incomplete = [m["name"] for m in monitors if not m["ready"]]
            if mil or stored or permanent:
                bad = stored or permanent
                verdict = "fail"
                headline = "Would fail an emissions inspection"
                detail = ("The check-engine light is " + ("on" if mil else "off") +
                          f" with {len(stored)} stored and {len(permanent)} permanent code(s)"
                          + (f" — first: {bad[0]['code']}" if bad else "") +
                          ". Permanent codes cannot be cleared with a scanner; the car removes them "
                          "itself once it verifies the repair. Fix the fault, then drive until every "
                          "monitor has run before retesting.")
            elif incomplete:
                verdict = "not_ready"
                headline = "Not ready for inspection yet"
                detail = ("No codes, but these monitors have not completed since the codes were "
                          "last cleared: " + ", ".join(incomplete) +
                          ". Drive normally for a few days — mixed city and highway trips from a "
                          "cold start — then check again.")
            else:
                verdict = "pass"
                headline = "Ready — should pass the emissions check"
                detail = ("Check-engine light off, no stored, pending or permanent codes, and "
                          "every emissions monitor has completed.")
            return {"mil_on": mil, "dtc_count": count, "monitors": monitors,
                    "stored": stored, "pending": pending, "permanent": permanent,
                    "verdict": verdict, "headline": headline, "detail": detail}

    def _read_vin(self) -> str:
        try:
            raw = self._cmd("0902", 6)
        except Exception:  # noqa: BLE001
            return ""
        parts = {}
        for f in _hex_frames(raw):
            if len(f) >= 3 and f[0] == 0x49 and f[1] == 0x02:
                parts[f[2]] = f[3:]
        if not parts:
            return ""
        raw_bytes = b"".join(parts[k] for k in sorted(parts))
        vin = "".join(chr(b) for b in raw_bytes if 32 <= b < 127).strip("\x00 ")
        return vin if len(vin) >= 11 else ""

    def clear_codes(self) -> dict:
        if self.transport is None:
            raise ObdError("not connected")
        with self.lock:
            raw = self._cmd("04", 6)
            if self._frame_for(raw, 0x44) is None and "NO DATA" not in raw:
                raise ObdError(f"the car did not confirm clearing (answer: {raw.strip()[:80]})")
            self._mon_cache = (0.0, {})
            return {"cleared": True, **self.monitor_status()}


_OBD_SINGLETON: Optional[ObdSession] = None


def obd_session() -> ObdSession:
    global _OBD_SINGLETON
    if _OBD_SINGLETON is None:
        _OBD_SINGLETON = ObdSession()
    return _OBD_SINGLETON


def obd_bluetooth() -> dict:
    """Bonded Bluetooth devices, for the adapter picker (Android bridge)."""
    if _is_android():
        try:
            from java import jclass
            bridge = jclass("com.mhexeract.app.ObdBridge")
            devices = json.loads(str(bridge.devices()))
            return {"available": bool(bridge.available()), "devices": devices}
        except Exception:  # noqa: BLE001
            return {"available": False, "devices": []}
    return {"available": hasattr(__import__("socket"), "AF_BLUETOOTH"), "devices": []}


def serve_gui(gw: "Gateway", host: str = "127.0.0.1", port: int = 8765, token: Optional[str] = None) -> None:
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from urllib.parse import parse_qs, urlparse

    lock = threading.Lock()  # one gateway, many phone/browser clients
    global _GUI_STARTED_AT, _GUI_SERVER
    _GUI_STARTED_AT = time.time()
    if gw.mock and os.getenv("GATEWAY_MOCK") != "1" and load_saved_keys():
        gw.mock = False  # keys saved in the tab on an earlier run: demo mode is over
    else:
        load_saved_keys()

    class H(BaseHTTPRequestHandler):
        server_version = "MHexeract/1.0"

        def log_message(self, fmt: str, *a: Any) -> None:
            log.debug("gui " + fmt, *a)

        def _send(self, code: int, body: Any, ctype: str = "application/json") -> None:
            raw = body if isinstance(body, bytes) else (
                body.encode() if isinstance(body, str) else json.dumps(body, default=str).encode())
            gz = "gzip" in (self.headers.get("Accept-Encoding") or "") and len(raw) > 1024
            if gz:  # compress responses: smaller payloads on mobile data
                import gzip
                raw = gzip.compress(raw, 6)
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            if gz:
                self.send_header("Content-Encoding", "gzip")
            self.end_headers()
            self.wfile.write(raw)

        def _authorized(self, q: dict) -> bool:
            if not token:
                return True
            auth = self.headers.get("Authorization") or ""
            bearer = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
            got = self.headers.get("X-Gateway-Token") or bearer or (q.get("token") or [""])[0] or \
                (self.headers.get("Cookie") or "").partition("mhexeract_token=")[2].split(";")[0]
            import hmac
            return bool(got) and hmac.compare_digest(got, token)

        def _openai(self) -> None:
            def err(code: int, msg: str, typ: str) -> None:
                self._send(code, {"error": {"message": msg, "type": typ, "code": None}})
            try:
                body = self._body()
                with lock:
                    resp = openai_chat(gw, body)
            except KeyError as e:
                return err(404, str(e).strip("'\""), "invalid_request_error")
            except (ValueError, json.JSONDecodeError) as e:
                return err(400, str(e), "invalid_request_error")
            except AllRoutesFailed as e:
                return err(502, str(e), "upstream_error")
            except GatewayError as e:
                return err(503, str(e), "service_unavailable")
            except Exception as e:  # noqa: BLE001
                log.exception("openai api error")
                return err(500, f"{type(e).__name__}: {e}", "server_error")
            if not body.get("stream"):
                return self._send(200, resp)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            for chunk in openai_stream_chunks(resp):
                self.wfile.write(chunk)
                self.wfile.flush()
            self.close_connection = True

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            if n > 1_000_000:
                raise GatewayError("request body too large")
            return json.loads(self.rfile.read(n) or b"{}")

        def do_GET(self) -> None:  # noqa: N802
            u = urlparse(self.path)
            q = parse_qs(u.query)
            static = {"/manifest.webmanifest": (json.dumps(MANIFEST), "application/manifest+json"),
                      "/icon.svg": (ICON_SVG, "image/svg+xml"),
                      "/sw.js": (SERVICE_WORKER, "text/javascript")}
            if u.path in static:
                return self._send(200, *static[u.path])
            if not self._authorized(q):
                if u.path.startswith("/v1/"):
                    return self._send(401, {"error": {"message": "invalid API key: use the gateway token",
                                                      "type": "invalid_request_error", "code": "invalid_api_key"}})
                return self._send(401, {"error": "token required: open /?token=..."})
            if u.path == "/v1/models":
                return self._send(200, openai_models(gw))
            if u.path in ("/", "/index.html"):
                body = polish_html((HERE / "dashboard.html").read_bytes())
                self.send_response(200)
                if token:
                    self.send_header("Set-Cookie", f"mhexeract_token={token}; Path=/; SameSite=Strict; HttpOnly")
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if u.path in ("/game.html", "/solitaire.html", "/tetris.html", "/platformer.html", "/oregon.html", "/blackjack.html") and (HERE / u.path[1:]).exists():
                return self._send(200, polish_html((HERE / u.path[1:]).read_bytes()), "text/html; charset=utf-8")
            if u.path == "/api/repos":
                return self._send(200, dict(load_repos(gw.repo_registry),
                                             github_token=bool(_github_token() or __import__("shutil").which("gh"))))
            if u.path == "/api/news":
                inc = [x for x in (q.get("sources") or [""])[0].split(",") if x] or None
                try:
                    return self._send(200, fetch_news(force=(q.get("refresh") or ["0"])[0] == "1", include=inc))
                except Exception as e:  # noqa: BLE001
                    return self._send(502, {"error": f"news unavailable ({type(e).__name__}: {str(e)[:160]})"})
            if u.path == "/api/wine":
                return self._send(200, wine_status())
            if u.path == "/api/obd/status":
                return self._send(200, obd_session().status())
            if u.path == "/api/obd/devices":
                return self._send(200, obd_bluetooth())
            if u.path == "/api/obd/live":
                try:
                    return self._send(200, obd_session().live())
                except ObdError as e:
                    return self._send(400, {"error": str(e)})
            if u.path == "/api/obd/codes":
                try:
                    return self._send(200, obd_session().codes())
                except ObdError as e:
                    return self._send(400, {"error": str(e)})
            if u.path == "/api/obd/emissions":
                try:
                    return self._send(200, obd_session().emissions())
                except ObdError as e:
                    return self._send(400, {"error": str(e)})
            if u.path == "/api/keys":
                return self._send(200, keys_status())
            if u.path == "/api/toolbox":
                with lock:
                    return self._send(200, toolbox_status(gw))
            if u.path == "/api/units":
                with lock:
                    return self._send(200, gw.ai_units())
            if u.path == "/api/nodes":
                d = load_nodes(gw.nodes_path)
                return self._send(200, dict(d, summary=nodes_summary(d), config=nodes_config(gw)))
            if u.path == "/api/update":
                try:
                    d = check_update(force=(q.get("refresh") or ["0"])[0] == "1")
                    d["channel"] = _channel()
                    if _channel() == "play":
                        d["available"] = False  # Play delivers updates, not GitHub
                    return self._send(200, d)
                except GatewayError as e:
                    return self._send(200, {"error_detail": str(e), "current": current_version(),
                                            "mode": "apk" if _is_android() else "files", "available": False,
                                            "channel": _channel(),
                                            "backups": [b.name for b in _update_backups()]})
            if u.path == "/api/report":
                days = int((q.get("days") or ["7"])[0])
                with lock:
                    return self._send(200, gw.report(days))
            self._send(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            u = urlparse(self.path)
            q = parse_qs(u.query)
            if not self._authorized(q):
                if u.path.startswith("/v1/"):
                    return self._send(401, {"error": {"message": "invalid API key: use the gateway token",
                                                      "type": "invalid_request_error", "code": "invalid_api_key"}})
                return self._send(401, {"error": "unauthorized"})
            origin = self.headers.get("Origin")  # CSRF guard: reject cross-site browser POSTs
            if origin and origin != "null" and urlparse(origin).netloc != self.headers.get("Host"):
                return self._send(403, {"error": "cross-origin request blocked"})
            if u.path == "/v1/chat/completions":
                return self._openai()
            if u.path in ("/api/update/apply", "/api/update/rollback", "/api/update/apk"):
                if _channel() == "play":
                    return self._send(400, {"error": "This copy of MHexeract updates through Google Play — open the Play Store to update."})
                try:  # outside the gateway lock: downloads can take a while
                    res = apply_update() if u.path.endswith("apply") else rollback_update() \
                        if u.path.endswith("rollback") else download_apk()
                except (GatewayError, ValueError, SyntaxError, OSError) as e:
                    return self._send(400, {"error": f"{type(e).__name__}: {e}" if isinstance(e, SyntaxError) else str(e)})
                self._send(200, res)
                if res.get("restart") and not _is_android():
                    restart_process()
                return
            if u.path == "/api/obd/connect":
                b = self._body()
                try:
                    res = obd_session().connect(str(b.get("transport") or "demo"),
                                                str(b.get("host") or ""), int(b.get("port") or 35000),
                                                str(b.get("address") or ""))
                except (ObdError, ValueError) as e:
                    return self._send(400, {"error": str(e)})
                return self._send(200, res)
            if u.path == "/api/obd/disconnect":
                obd_session().disconnect()
                return self._send(200, obd_session().status())
            if u.path == "/api/obd/clear":
                try:
                    return self._send(200, obd_session().clear_codes())
                except ObdError as e:
                    return self._send(400, {"error": str(e)})
            if u.path == "/api/keys":
                try:
                    res = save_keys(self._body().get("keys") or {})
                except (ValueError, GatewayError, OSError) as e:
                    return self._send(400, {"error": str(e)})
                if res["saved"] and gw.mock and os.getenv("GATEWAY_MOCK") != "1":
                    gw.mock = False  # a real key was just saved: leave demo mode
                with lock:
                    gw.load_nodes()  # cached scan: newly keyed providers join the pool
                return self._send(200, res)
            if u.path == "/api/toolbox":
                b = self._body()
                bg = b.get("background")
                if isinstance(bg, dict):  # toggles for the scheduled background jobs
                    for name, policy in (("auto_improve", "bg_auto_improve"), ("repo_sync", "bg_repo_sync"),
                                         ("node_scan", "bg_node_scan"), ("lan_scan", "bg_lan_scan"),
                                         ("gardener", "bg_gardener"), ("ui_polish", "bg_ui_polish"),
                                         ("strive", "bg_strive"),
                                         ("obd_update", "bg_obd_update")):
                        if name in bg:
                            gw.store.set_policy(policy, bool(bg[name]))
                action = b.get("action")
                if action == "restart":  # desktop/Termux only: Android restarts from its Toolbox menu
                    if _is_android():
                        return self._send(400, {"error": "on Android, restart from the app menu (Toolbox)"})
                    self._send(200, toolbox_status(gw))
                    restart_process()
                    return
                if action == "shutdown":  # full stop: server down; Android closes the app via mhexeract://close-app
                    self._send(200, {**toolbox_status(gw), "shutdown": True})
                    shutdown_gateway()
                    return
                if action == "run":  # run one background job right now
                    job_name = str(b.get("job") or "")
                    try:
                        if job_name == "improve":
                            with lock:
                                run_scheduled_improvement(gw, force=True)
                        elif job_name == "repos":
                            res = sync_repos(gw.repo_registry)
                            gw.store.set_policy("last_repo_sync_ts", time.time())
                        elif job_name == "nodes":
                            run_node_scan(gw, lock, force=True)
                        elif job_name == "gardener":
                            run_gardener(gw, force=True)
                            run_node_scan(gw, lock, force=True)  # adopt free nodes / drop cleaned ones
                        elif job_name == "strive":
                            run_strive(gw, force=True)
                        elif job_name == "ui":
                            run_ui_polish(gw, force=True)
                        elif job_name == "obd":
                            run_obd_update(gw, force=True)
                        else:
                            return self._send(400, {"error": "job must be improve, repos, nodes, gardener, strive, ui or obd"})
                    except (GatewayError, ValueError, OSError) as e:
                        return self._send(400, {"error": str(e)})
                with lock:
                    return self._send(200, toolbox_status(gw))
            if u.path == "/api/cost":
                b = self._body()
                with lock:
                    gw.store.set_policy("allow_paid", bool(b.get("allow_paid")))
                    gw.store._policy_ts = 0
                    return self._send(200, gw.ai_units())
            if u.path in ("/api/nodes/scan", "/api/nodes/trust"):
                try:
                    if u.path.endswith("scan"):
                        res = run_node_scan(gw, lock, force=True)
                    else:
                        b = self._body()
                        res = trust_node(gw.nodes_path, str(b.get("base_url", "")), bool(b.get("trusted")))
                        with lock:
                            gw.load_nodes()
                except (GatewayError, ValueError, OSError) as e:
                    return self._send(400, {"error": str(e)})
                return self._send(200, dict(res, summary=nodes_summary(res)))
            try:
                with lock:
                    if u.path == "/api/chat":
                        b = self._body()
                        msgs = b.get("messages")
                        if isinstance(msgs, list) and msgs:  # multi-turn from the main chat screen
                            msgs = [{"role": m["role"] if m.get("role") in ("system", "user", "assistant") else "user",
                                     "content": str(m.get("content", ""))[:200_000]} for m in msgs[-60:]]
                        else:
                            msgs = str(b.get("prompt", ""))[:200_000]
                        model = b.get("model")
                        pinned = [model] if model and model in gw.models else None
                        if b.get("council") and not pinned:  # models confer over the full history
                            r = gw.council_chat(msgs)
                        else:
                            r = gw.chat(msgs, strategy=b.get("strategy"), tier=b.get("tier") if not pinned else None,
                                        models=pinned)
                        return self._send(200, {**r.summary(), "text": r.text})
                    if u.path == "/api/fetch":
                        b = self._body()
                        try:
                            return self._send(200, fetch_page(str(b.get("url", ""))[:4000]))
                        except ValueError:
                            raise
                        except Exception as e:  # noqa: BLE001  - network errors are user-facing, not 500s
                            return self._send(502, {"error": f"could not open the page ({type(e).__name__}: {str(e)[:160]})"})
                    if u.path == "/api/news/brief":
                        b = self._body()
                        cl = b.get("cluster") or {}
                        if not isinstance(cl, dict) or not cl.get("items"):
                            raise ValueError("cluster with items required")
                        r = gw.chat(news_brief_messages(cl), strategy="quality")
                        return self._send(200, {**r.summary(), "text": r.text})
                    if u.path == "/api/feedback":
                        b = self._body()
                        gw.feedback(str(b["request_id"]), float(b["score"]), source="gui")
                        return self._send(200, {"ok": True})
                    if u.path == "/api/improve":
                        apply = (q.get("apply") or ["0"])[0] == "1"
                        return self._send(200, {"changes": gw.optimizer.run(apply=apply)})
                    if u.path == "/api/compact":
                        return self._send(200, gw.store.compact(force=True))
                    if u.path == "/api/repos/sync":
                        res = sync_repos(gw.repo_registry)
                        gw.store.set_policy("last_repo_sync_ts", time.time())
                        return self._send(200, res)
            except (GatewayError, KeyError, ValueError) as e:
                return self._send(400, {"error": str(e)})
            except Exception as e:  # noqa: BLE001
                log.exception("gui error")
                return self._send(500, {"error": f"{type(e).__name__}: {e}"})
            self._send(404, {"error": "not found"})

    srv = ThreadingHTTPServer((host, port), H)
    _GUI_SERVER = srv
    threading.Thread(target=_scheduled_improver, args=(gw, lock), name="mhexeract-improver", daemon=True).start()
    shown = "localhost" if host in ("127.0.0.1", "0.0.0.0") else host
    print(f"MHexeract GUI on http://{shown}:{port}/" + (f"?token={token}" if token else ""))
    if host == "0.0.0.0":
        print("Listening on all interfaces: open it from your phone with this machine's LAN IP.")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="MHexeract multi-model AI gateway")
    ap.add_argument("--config", default=os.getenv("GATEWAY_CONFIG", "models.json"))
    ap.add_argument("--db", default=os.getenv("GATEWAY_DB", "gateway_usage.db"))
    ap.add_argument("--mock", action="store_true", help="use LiteLLM mock responses (no API keys)")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("chat", help="send a prompt through the gateway")
    c.add_argument("prompt")
    c.add_argument("--system")
    c.add_argument("--strategy", choices=["balanced", "efficient", "quality"])
    c.add_argument("--tier", choices=TIERS)
    c.add_argument("--models", help="comma-separated aliases to restrict routing")
    c.add_argument("--max-tokens", type=int)
    c.add_argument("--no-cache", action="store_true")

    f = sub.add_parser("feedback", help="rate a response 0..1 (teaches the router)")
    f.add_argument("request_id")
    f.add_argument("score", type=float)
    f.add_argument("--note", default="")

    i = sub.add_parser("improve", help="run the automated process-improvement optimizer")
    i.add_argument("--dry-run", action="store_true")

    r = sub.add_parser("report", help="usage, tokens, quality, learned policy")
    r.add_argument("--days", type=int, default=7)
    r.add_argument("--html", metavar="FILE", help="export a self-contained GUI report instead of JSON")

    g = sub.add_parser("gui", help="serve the live GUI dashboard (phone-friendly, installable)")
    g.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to reach it from a phone on your LAN")
    g.add_argument("--port", type=int, default=8765)
    g.add_argument("--token", default=os.getenv("GATEWAY_GUI_TOKEN"), help="require a token (recommended with 0.0.0.0)")

    sub.add_parser("compact", help="force storage compaction (roll old rows into daily totals)")

    sub.add_parser("models", help="list model pool and availability")

    rp = sub.add_parser("repos", help="auto-add every GitHub repository your token can access")
    rp.add_argument("--clone", metavar="DIR", help="also clone new repos and fast-forward existing ones into DIR")
    rp.add_argument("--include-archived", action="store_true")
    rp.add_argument("--list", action="store_true", help="show the registry without syncing")

    d = sub.add_parser("discover-openai", help="find every OpenAI chat model your key can use")
    d.add_argument("--write", action="store_true", help="add the new models to models.json")

    nw = sub.add_parser("news", help="headlines grouped by story across outlets rated left to right (AllSides)")
    nw.add_argument("--limit", type=int, default=15)
    nw.add_argument("--brief", type=int, metavar="N", help="AI balanced brief of story N (1-based)")
    nw.add_argument("--sources", action="store_true", help="show source status and bias ratings")

    wn = sub.add_parser("wine", help="run Windows programs through Wine (Box64 on ARM, Winlator on Android)")
    wn.add_argument("action", choices=["status", "run"])
    wn.add_argument("file", nargs="?")
    wn.add_argument("args", nargs=argparse.REMAINDER)
    wn.add_argument("--wait", action="store_true")

    up = sub.add_parser("update", help="install the newest MHexeract from GitHub (verified, with backup)")
    up.add_argument("action", nargs="?", default="check", choices=["check", "apply", "rollback"])

    nd = sub.add_parser("nodes", help="find more AI nodes: cloud providers, local model servers, home network")
    nd.add_argument("action", nargs="?", default="scan", choices=["scan", "list", "trust", "untrust"])
    nd.add_argument("url", nargs="?", help="node base URL for trust/untrust")

    s = sub.add_parser("simulate", help="offline learning demo with synthetic traffic")
    s.add_argument("--n", type=int, default=300)

    args = ap.parse_args()
    try:
        _run(args)
    except GatewayError as e:
        raise SystemExit(f"gateway error: {e}")


def _run(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    gw = Gateway(args.config, args.db, mock=True if args.mock or args.cmd == "simulate" else None)

    if args.cmd == "chat":
        msgs = ([{"role": "system", "content": args.system}] if args.system else []) + \
               [{"role": "user", "content": args.prompt}]
        kw = {"max_tokens": args.max_tokens} if args.max_tokens else {}
        resp = gw.chat(msgs, strategy=args.strategy, tier=args.tier,
                       models=args.models.split(",") if args.models else None, use_cache=not args.no_cache, **kw)
        print(resp.text)
        print("\n---", json.dumps(resp.summary(), indent=2))
    elif args.cmd == "feedback":
        gw.feedback(args.request_id, args.score, args.note)
        print("feedback recorded")
    elif args.cmd == "improve":
        changes = gw.optimizer.run(apply=not args.dry_run)
        print(json.dumps(changes, indent=2) if changes else "No changes: not enough evidence or policy already optimal.")
    elif args.cmd == "report":
        if args.html:
            print("wrote", export_html(gw, args.html, args.days))
        else:
            print(json.dumps(gw.report(args.days), indent=2, default=str))
    elif args.cmd == "gui":
        if args.host == "0.0.0.0" and not args.token:
            args.token = uuid.uuid4().hex[:16]
            print("No --token given for a network-exposed GUI; generated one.")
        serve_gui(gw, args.host, args.port, args.token)
    elif args.cmd == "compact":
        print(json.dumps(gw.store.compact(force=True), indent=2))
    elif args.cmd == "repos":
        if args.list:
            reg = load_repos(gw.repo_registry)
        else:
            res = sync_repos(gw.repo_registry, args.clone, args.include_archived)
            print(json.dumps({k: v for k, v in res.items() if v or k == "total"}, indent=2))
            reg = load_repos(gw.repo_registry)
        for r in reg["repos"]:
            print(f"{'private' if r['private'] else 'public ':7}  {r['full_name']:40} {r.get('local_path', '')}")
    elif args.cmd == "discover-openai":
        added = discover_openai(args.config, write=args.write)
        for m in added:
            print(f"{'added' if args.write else 'new'}  {m['alias']:28} {m['tier']}")
        print(f"{len(added)} new OpenAI chat models" + ("" if args.write or not added else "; rerun with --write to add them"))
    elif args.cmd == "models":
        for s_ in gw.models.values():
            for rt in s_.routes:
                print(f"{s_.alias:22} {s_.provider:10} {s_.tier:9} q={s_.quality:<3} {rt.via:10} "
                                            f"{'available' if gw._route_available(s_, rt) else 'no key'}  {rt.model}")
    elif args.cmd == "news":
        res = fetch_news()
        if args.sources:
            for st in res["sources"]:
                print(f"{st['lean']:11} {st['name']:28} {'ok ' + str(st['items']) if st['ok'] else 'error ' + st['error']}")
            print(res["ratings"])
            return
        cl = res["clusters"]
        if args.brief:
            c = cl[args.brief - 1]
            print(c["title"], "\n")
            print(gw.chat(news_brief_messages(c), strategy="quality").text)
            return
        for n, c in enumerate(cl[:args.limit], 1):
            sd = c["sides"]
            print(f"{n:2}. {c['title']}\n    {c['n_sources']} sources  left {sd['left']} / center {sd['center']} / right {sd['right']}"
                  + (f"  ({c['balance_note']})" if c["balance_note"] else ""))
            for i in c["items"]:
                print(f"      {i['lean']:10} {i['source']:24} {i['title'][:80]}")
        print("\n" + res["ratings"])
    elif args.cmd == "wine":
        if args.action == "status":
            print(json.dumps(wine_status(), indent=2))
        else:
            if not args.file:
                raise SystemExit("usage: gateway.py wine run FILE.exe [args...]")
            print(json.dumps(wine_run(args.file, args.args, wait=args.wait), indent=2))
    elif args.cmd == "update":
        if args.action == "check":
            r = check_update(force=True)
            print(f"installed {r['current']['commit'][:7] or 'unknown'} ({r['current']['source']}), "
                  f"newest {r['latest']['commit'][:7]}: {r['latest']['message']}")
            for c in r["changes"]:
                print(f"  {c['commit']}  {c['message']}")
            print("Update available: run `python gateway.py update apply`" if r["available"] else "Up to date.")
        else:
            print(json.dumps(apply_update() if args.action == "apply" else rollback_update(), indent=2))
            print("Restart MHexeract to use the new version.")
    elif args.cmd == "nodes":
        if args.action in ("trust", "untrust"):
            if not args.url:
                raise SystemExit("usage: gateway.py nodes trust|untrust BASE_URL")
            trust_node(gw.nodes_path, args.url.rstrip("/"), args.action == "trust")
            d = load_nodes(gw.nodes_path)
        else:
            d = scan_nodes(gw.nodes_path, gw) if args.action == "scan" else load_nodes(gw.nodes_path)
        sm = nodes_summary(d)
        print(f"{sm['online']} nodes online, {sm['in_use']} in use with {sm['models_in_use']} models, "
              f"{sm['needs_key']} more available with a key (scanned {d.get('scanned_at')})")
        for n in sorted(d.get("nodes", []), key=lambda n: (n["status"] != "online", n["kind"], n["name"])):
            state = n["status"] if n["status"] != "online" else ("in use" if n.get("trusted") else "found, not trusted")
            extra = f"  set {n['env']} ({n.get('signup', '')})" if n["status"] == "needs_key" else \
                f"  {len(n.get('models', []))} of {n.get('models_total', 0)} models" if n["status"] == "online" else \
                f"  {n.get('error', '')[:80]}"
            print(f"  {n['kind']:6} {n['name']:34} {state:18}{extra}")
    elif args.cmd == "simulate":
        before = gw.task_policy()
        simulate(gw, args.n)
        rep = gw.report(days=1)
        print("policy before:", json.dumps(before))
        print("policy after: ", json.dumps(rep["policy"]["task_policy"]))
        print(json.dumps({k: rep[k] for k in ("totals", "by_task")}, indent=2))
        print("improvements:")
        for imp in reversed(rep["recent_improvements"]):
            print(f"  [{imp['status']}] {imp['kind']} {imp['target']}: {imp['old_value']} -> {imp['new_value']} ({imp['reason']})")


if __name__ == "__main__":
    main()
