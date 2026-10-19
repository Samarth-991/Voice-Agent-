import os

# Fast, deterministic engines for protocol tests (set before app.config is imported).
os.environ.setdefault("VAD", "energy")
os.environ.setdefault("STT", "mock")
os.environ.setdefault("RESPONDER", "mock")
os.environ.setdefault("TTS", "mock")
# Tests that need persistence install their own store (tmp dir).
os.environ.setdefault("MEMORY", "connection")
