"""Starlette ASGI entrypoint for the Zava sales agent.

Routes:
    GET  /                 — chat UI
    GET  /api/health       — readiness probe
    POST /api/chat         — NDJSON streaming chat with per-msg-id dedupe and
                             nano-utility tag filtering.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Load .env.local before importing modules that read env at import time.
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env.local")

# Wire Azure Monitor OpenTelemetry (only when APPLICATIONINSIGHTS_CONNECTION_STRING
# is set — i.e. in deployed environments). No-op locally.
if os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING"):
    try:
        from azure.monitor.opentelemetry import configure_azure_monitor

        configure_azure_monitor(logger_name="app")
    except Exception as exc:  # pragma: no cover — best-effort
        logging.getLogger(__name__).warning("App Insights setup failed: %s", exc)

from langchain_mcp_adapters.client import MultiServerMCPClient
from starlette.applications import Starlette
from starlette.responses import FileResponse, JSONResponse, StreamingResponse
from starlette.routing import Route

from app.agent import build_agent, build_models
from app.streaming import event, iter_message_events
from app.tools import LOCAL_TOOLS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

ENVIRONMENT = os.getenv("ENVIRONMENT", "production")
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000").rstrip("/")
if not MCP_SERVER_URL.endswith("/mcp"):
    MCP_SERVER_URL = f"{MCP_SERVER_URL}/mcp"


# Funnel step names that map to the UI's frosted pill.
KNOWN_STEPS = {"greet", "qualify", "educate", "objection", "book", "handoff_to_ae"}


# ---- Lifespan -------------------------------------------------------------
async def _connect_mcp_with_retry(client: MultiServerMCPClient, attempts: int = 5) -> list:
    """Fetch MCP tools, retrying with exponential backoff on transient errors.

    Container Apps may start the agent before the MCP service is reachable;
    we want a few retries before crash-looping the container.
    """
    delay = 1.0
    last_exc: Exception | None = None
    for i in range(1, attempts + 1):
        try:
            tools = await client.get_tools()
            logger.info("📦 Loaded %d MCP tool(s) from %s", len(tools), MCP_SERVER_URL)
            return tools
        except Exception as exc:
            last_exc = exc
            logger.warning("MCP get_tools attempt %d/%d failed: %s", i, attempts, exc)
            if i < attempts:
                await asyncio.sleep(delay)
                delay *= 2
    raise RuntimeError(f"Could not reach MCP server at {MCP_SERVER_URL}") from last_exc


@asynccontextmanager
async def lifespan(app: Starlette):
    logger.info("Initialising sales agent (env=%s, mcp=%s)…", ENVIRONMENT, MCP_SERVER_URL)

    main_model, nano_model, credential = build_models()

    # Cache one (mcp_client, mcp_tools, agent) per persona key. Personas
    # are switched from the UI dropdown; the persona's role + actor id
    # become headers on every MCP call so Postgres RLS can scope rows.
    app.state.persona_cache = {}
    app.state.persona_lock = asyncio.Lock()
    app.state.main_model = main_model
    app.state.nano_model = nano_model

    # Warm the default 'admin' persona so first request is fast.
    await _get_or_build_agent(app, "admin", "0")
    app.state.local_tool_count = len(LOCAL_TOOLS)
    app.state.ready = True
    logger.info("✅ Agent ready (default admin persona warmed)")

    try:
        yield
    finally:
        try:
            await credential.close()
        except Exception:
            pass


async def _get_or_build_agent(app: Starlette, role: str, actor_id: str):
    """Return a cached (agent, mcp_tool_count) for the persona, building if needed."""
    key = (role, actor_id)
    cached = app.state.persona_cache.get(key)
    if cached is not None:
        return cached

    async with app.state.persona_lock:
        cached = app.state.persona_cache.get(key)
        if cached is not None:
            return cached

        client = MultiServerMCPClient(
            {
                "zava-sales": {
                    "url": MCP_SERVER_URL,
                    "transport": "streamable_http",
                    "headers": {
                        "X-Sales-Role": role,
                        "X-Sales-Actor-Id": str(actor_id),
                    },
                }
            }
        )
        tools = await _connect_mcp_with_retry(client)
        agent = build_agent(app.state.main_model, app.state.nano_model, tools)
        cached = {"agent": agent, "mcp_tool_count": len(tools), "client": client}
        app.state.persona_cache[key] = cached
        logger.info("🪪 Built agent for persona role=%s actor=%s (%d tools)",
                    role, actor_id, len(tools))
        return cached


# ---- Routes ---------------------------------------------------------------
async def index(request):
    return FileResponse(STATIC_DIR / "index.html", media_type="text/html")


async def health(request):
    state = request.app.state
    ready = getattr(state, "ready", False)
    default_persona = state.persona_cache.get(("admin", "0")) if ready else None
    return JSONResponse(
        {
            "status": "healthy" if ready else "starting",
            "ready": ready,
            "environment": ENVIRONMENT,
            "mcp_server": MCP_SERVER_URL,
            "local_tool_count": getattr(state, "local_tool_count", 0),
            "mcp_tool_count": (default_persona or {}).get("mcp_tool_count", 0),
            "personas_cached": len(getattr(state, "persona_cache", {})),
        },
        status_code=200 if ready else 503,
    )


# Personas the UI dropdown can switch to. Kept in lock-step with what
# data/generate_database.py seeds. Pure presentation — Postgres RLS is
# what actually enforces the scoping.
PERSONAS = [
    {"key": "admin",       "role": "admin",    "actor_id": "0", "label": "Admin (bypass RLS)"},
    {"key": "manager-1",   "role": "manager",  "actor_id": "1", "label": "Manager · Maria Chen"},
    {"key": "manager-2",   "role": "manager",  "actor_id": "2", "label": "Manager · Marcus Johnson"},
    {"key": "ae-1",        "role": "ae",       "actor_id": "1", "label": "AE · Sarah Patel"},
    {"key": "ae-2",        "role": "ae",       "actor_id": "2", "label": "AE · Tom Rivera"},
    {"key": "ae-3",        "role": "ae",       "actor_id": "3", "label": "AE · Aisha Khan"},
    {"key": "ae-4",        "role": "ae",       "actor_id": "4", "label": "AE · Diego Romero"},
    {"key": "customer-1",  "role": "customer", "actor_id": "1", "label": "Customer #1"},
    {"key": "customer-2",  "role": "customer", "actor_id": "2", "label": "Customer #2"},
    {"key": "customer-3",  "role": "customer", "actor_id": "3", "label": "Customer #3"},
    {"key": "customer-4",  "role": "customer", "actor_id": "4", "label": "Customer #4"},
]


async def personas(request):
    return JSONResponse({"personas": PERSONAS, "default": "admin"})


async def chat(request):
    state = request.app.state
    if not getattr(state, "ready", False):
        return JSONResponse({"error": "Agent is not ready yet."}, status_code=503)

    try:
        body = await request.json()
    except json.JSONDecodeError:
        return JSONResponse({"error": "Invalid JSON body."}, status_code=400)

    message = body.get("message")
    if not message:
        return JSONResponse({"error": "message is required"}, status_code=400)

    role = (body.get("role") or "admin").lower()
    actor_id = str(body.get("actor_id") or "0")
    if role not in {"admin", "manager", "ae", "customer"}:
        return JSONResponse({"error": f"unknown role '{role}'"}, status_code=400)

    persona = await _get_or_build_agent(request.app, role, actor_id)
    agent = persona["agent"]

    history = body.get("history") or []
    thread_id = body.get("thread_id") or str(uuid.uuid4())

    history_msgs = [
        {"role": m["role"], "content": m["content"]}
        for m in history
        if m.get("role")
    ]
    history_msgs.append({"role": "user", "content": message})

    initial_state: dict[str, Any] = {"messages": history_msgs}

    config = {"configurable": {"thread_id": thread_id}}

    async def generate():
        # Always emit the thread id on the first event so the UI can
        # persist it and reuse it on the next turn.
        yield event({"thread_id": thread_id})

        full_text: list[str] = []
        last_step: str | None = None
        # Per-message dedupe: LangGraph can emit both streaming token deltas
        # AND a final aggregated chunk with the full cumulative text for the
        # same message id. Track per-msg-id text we've already streamed.
        text_by_msg: dict[str, str] = {}

        try:
            async for chunk in agent.astream(
                initial_state, config, stream_mode="messages"
            ):
                # stream_mode="messages" yields (AIMessageChunk, metadata) tuples.
                if isinstance(chunk, tuple) and len(chunk) >= 1:
                    msg = chunk[0]
                    metadata = chunk[1] if len(chunk) > 1 else {}
                else:
                    msg = chunk
                    metadata = {}

                # Drop chunks from internal nano-utility LLM calls (refine /
                # validate). They show up in the stream because LangGraph
                # streams every model call in the graph; we don't want them
                # in the user's chat bubble.
                tags = metadata.get("tags", []) if isinstance(metadata, dict) else []
                if "nano-utility" in tags:
                    continue

                # Surface step transitions from the per-chunk metadata.
                cur_step = None
                if isinstance(metadata, dict):
                    cur_step = metadata.get("current_step") or metadata.get("langgraph_node")
                if cur_step and cur_step != last_step and cur_step in KNOWN_STEPS:
                    last_step = cur_step
                    yield event({"step": cur_step})

                for ev in iter_message_events(msg):
                    if ev["kind"] == "text":
                        msg_id = getattr(msg, "id", None) or "_anon_"
                        chunk_text = ev["text"]
                        already = text_by_msg.get(msg_id, "")
                        new_emit = chunk_text
                        # Cumulative chunk that starts with what we've emitted.
                        if already and chunk_text.startswith(already):
                            new_emit = chunk_text[len(already):]
                        # Final aggregated chunk that repeats prior text exactly.
                        elif already and chunk_text == already:
                            continue
                        # Edge case: model produced an exact duplicate suffix.
                        elif already and already.endswith(chunk_text):
                            continue
                        if not new_emit:
                            continue
                        text_by_msg[msg_id] = already + new_emit
                        full_text.append(new_emit)
                        yield event({"chunk": new_emit})
                    elif ev["kind"] == "tool":
                        yield event({"tool": ev["tool"]})
                    elif ev["kind"] == "image":
                        yield event({"image": ev["image"]})
                    elif ev["kind"] == "citation":
                        yield event({"citation": ev["citation"]})
                    elif ev["kind"] == "citations":  # legacy
                        for doc_id in ev["doc_ids"]:
                            yield event({"citation": {"doc_id": doc_id}})
        except Exception as exc:
            logger.exception("Error during agent stream")
            yield event({"error": f"agent stream failed: {exc}"})

        yield event(
            {
                "message": "".join(full_text),
                "role": "assistant",
                "step": last_step,
                "done": True,
            }
        )

    return StreamingResponse(generate(), media_type="application/x-ndjson")


# ---- App ------------------------------------------------------------------
routes = [
    Route("/", index, methods=["GET"]),
    Route("/api/chat", chat, methods=["POST"]),
    Route("/api/health", health, methods=["GET"]),
    Route("/api/personas", personas, methods=["GET"]),
]

app = Starlette(debug=False, routes=routes, lifespan=lifespan)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
