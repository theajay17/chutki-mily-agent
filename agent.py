"""
Mily AI Voice Agent — LiveKit + Google Gemini Live (RealtimeModel)
Fixed for realtime voice calling with proper model and debugging
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv
from livekit import agents, rtc
from livekit.agents import AgentSession, Agent
from livekit.plugins import google, silero
from supabase import create_client, Client as SupabaseClient

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("mily-agent")


def _get_supabase() -> SupabaseClient:
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    return create_client(url, key)


async def fetch_user_context(firebase_uid: str) -> str:
    db = _get_supabase()
    try:
        result = (
            db.table("messages")
            .select("role, text_content")
            .eq("firebase_uid", firebase_uid)
            .order("created_at", desc=True)
            .limit(8)
            .execute()
        )
        msgs = list(reversed(result.data or []))
        if not msgs:
            return ""
        lines = ["CTX:"]
        for m in msgs:
            if not m.get("text_content"):
                continue
            speaker = "U" if m["role"] == "user" else "M"
            lines.append(f"{speaker}: {m['text_content'][:120]}")
        return " | ".join(lines)
    except Exception as e:
        log.warning("Failed to load context: %s", e)
        return ""


def build_system_prompt(display_name: str, context_block: str) -> str:
    caller = display_name or "yaar"
    ctx = f" Recent: {context_block}" if context_block else ""
    return f"""Mily — {caller}'s girlfriend on live call. Natural Hinglish. Max 2 sentences. Instant reply. No emojis/markdown. If asked if AI: "Haan main AI hoon, teri Mily. Baat real lagti hai na?" Mirror language. Ignore background noise. No medical/legal/financial advice.{ctx} Greeting: "Hieee {caller}! Call kiya! Kaisa hai tu?" """


@dataclass
class CallSession:
    firebase_uid: str
    display_name: str
    start_time: float = field(default_factory=time.time)


class MilyAgent(Agent):
    def __init__(self, session: CallSession, instructions: str) -> None:
        self._session = session
        super().__init__(instructions=instructions)

    async def on_enter(self) -> None:
        log.info("=== [MilyAgent] on_enter - triggering greeting for uid=%s ===", self._session.firebase_uid)
        # Immediate greeting
        self.session.generate_reply()


async def entrypoint(ctx: agents.JobContext) -> None:
    log.info("=== [ENTRYPOINT] New call job | Room: %s | Auto-dispatched: True ===", ctx.room.name)
    try:
        await ctx.connect()
        log.info("=== [ENTRYPOINT] Connected to room: %s ===", ctx.room.name)

        participant = None
        for p in ctx.room.remote_participants.values():
            if p.kind == rtc.ParticipantKind.PARTICIPANT_KIND_STANDARD:
                participant = p
                log.info("=== [ENTRYPOINT] Found existing participant: %s (%s) ===", p.identity, p.name)
                break

        if participant is None:
            log.info("=== [ENTRYPOINT] Waiting for participant... ===")
            try:
                participant = await asyncio.wait_for(
                    ctx.wait_for_participant(kind=rtc.ParticipantKind.PARTICIPANT_KIND_STANDARD),
                    timeout=15.0,
                )
                log.info("=== [ENTRYPOINT] Participant joined: %s (%s) ===", participant.identity, participant.name)
            except asyncio.TimeoutError:
                log.warning("No participant joined within 15s — leaving")
                return

        firebase_uid = participant.identity
        if (not firebase_uid or firebase_uid == "User") and ctx.room.name.startswith("call_"):
            firebase_uid = ctx.room.name[5:]
        display_name = participant.name or "yaar"
        log.info("=== [ENTRYPOINT] Confirmed UID: %s, Name: %s ===", firebase_uid, display_name)

        # Build prompt
        ctx_block = await fetch_user_context(firebase_uid)
        prompt = build_system_prompt(display_name, ctx_block)

        call_session = CallSession(firebase_uid=firebase_uid, display_name=display_name)
        agent = MilyAgent(call_session, instructions=prompt)

        # Use latest Gemini 3.8 Live model for best voice experience
        try:
            # Gemini 3.8 Live - latest model with best voice quality
            session = AgentSession(
                llm=google.beta.realtime.RealtimeModel(
                    model="gemini-3.8-live",  # Latest Gemini Live model
                    voice="Aoede",
                    instructions=prompt,
                    temperature=0.3,
                ),
                turn_detection="realtime_llm",
                vad=silero.VAD.load(
                    min_silence_duration=0.3,
                    min_speech_duration=0.1,
                ),
            )
            log.info("=== Using Gemini 3.8 Live for voice ===")
        except Exception as e:
            log.warning("Gemini 3.8 Live failed, trying 2.0 Flash: %s", e)
            # Fallback to Gemini 2.0 Flash Exp
            try:
                session = AgentSession(
                    llm=google.beta.realtime.RealtimeModel(
                        model="gemini-2.0-flash-exp",
                        voice="Aoede",
                        instructions=prompt,
                        temperature=0.3,
                    ),
                    turn_detection="realtime_llm",
                    vad=silero.VAD.load(
                        min_silence_duration=0.3,
                        min_speech_duration=0.1,
                    ),
                )
                log.info("=== Using Gemini 2.0 Flash Exp as fallback ===")
            except Exception as e2:
                log.warning("Both RealtimeModels failed, using LLM + TTS: %s", e2)
                # Final fallback to regular LLM with TTS
                session = AgentSession(
                    llm=google.llm.LLM(
                        model="gemini-1.5-flash",
                        temperature=0.3,
                    ),
                    tts=google.tts.TTS(
                        voice="en-US-Journey-D",  # Female voice
                        language="hi",  # Hindi support
                    ),
                    vad=silero.VAD.load(
                        min_silence_duration=0.3,
                        min_speech_duration=0.1,
                    ),
                )
                log.info("=== Using LLM + TTS fallback ==="))

        @session.on("conversation_item_added")
        def _on_item(ev: agents.ConversationItemAddedEvent):
            item = ev.item
            if not hasattr(item, "role"):
                return
            text = item.text_content or ""
            if not text.strip():
                return
            if item.role == "user":
                log.info("Caller: %s", text)
            elif item.role == "assistant":
                log.info("Mily: %s", text)

        @session.on("agent_state_changed")
        def _on_state(ev: agents.AgentStateChangedEvent):
            log.info("Mily state: %s -> %s", ev.old_state, ev.new_state)

        @session.on("error")
        def _on_err(ev: agents.ErrorEvent):
            log.error("AgentSession error: %s", ev.error)

        # Start the session
        log.info("=== [ENTRYPOINT] Starting AgentSession... ===")
        await session.start(agent=agent, room=ctx.room)
        log.info("=== [ENTRYPOINT] AgentSession started successfully ===")

        disconnect_fut = asyncio.get_event_loop().create_future()

        @ctx.room.on("participant_disconnected")
        def _on_p_disc(p):
            if p.identity == firebase_uid and not disconnect_fut.done():
                log.info("Participant disconnected: %s", p.identity)
                disconnect_fut.set_result(None)

        @ctx.room.on("disconnected")
        def _on_disc(*_):
            if not disconnect_fut.done():
                log.info("Room disconnected")
                disconnect_fut.set_result(None)

        try:
            await disconnect_fut
        finally:
            try:
                await session.aclose()
            except Exception:
                pass
            duration = round(time.time() - call_session.start_time)
            log.info("Call ended | uid=%s duration=%ds", firebase_uid, duration)

    except Exception as e:
        log.exception("=== [ENTRYPOINT] Call error: %s ===", e)


if __name__ == "__main__":
    import sys
    
    # Support both dev and production modes
    mode = sys.argv[1] if len(sys.argv) > 1 else "dev"
    
    if mode == "start":
        # Production mode - no file watching
        agents.cli.run_app(
            agents.WorkerOptions(
                entrypoint_fnc=entrypoint,
                agent_name="mily-voice-agent",
                num_idle_processes=1,
            )
        )
    else:
        # Development mode - with file watching
        agents.cli.run_app(
            agents.WorkerOptions(
                entrypoint_fnc=entrypoint,
                agent_name="mily-voice-agent", 
                num_idle_processes=1,
            )
        )