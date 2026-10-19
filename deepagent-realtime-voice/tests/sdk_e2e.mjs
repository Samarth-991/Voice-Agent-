// End-to-end check with the real @openai/agents SDK (same classes the browser
// uses): connect, send the startup greeting, and confirm the SDK itself emits
// audio + completes responses (i.e. our events are not treated as "generic").
// Usage: node --experimental-websocket tests/sdk_e2e.mjs <agents-realtime dist dir> [ws url]
import path from "node:path";
import { pathToFileURL } from "node:url";

const dist = process.argv[2];
const url = process.argv[3] || "ws://127.0.0.1:7860/v1/realtime";
const { RealtimeAgent, RealtimeSession, OpenAIRealtimeWebSocket } = await import(
  pathToFileURL(path.join(dist, "index.mjs")).href
);

const transport = new OpenAIRealtimeWebSocket({
  useInsecureApiKey: true,
  createWebSocket: async ({ url: u }) => new globalThis.WebSocket(u),
});
const session = new RealtimeSession(new RealtimeAgent({ name: "e2e", instructions: "Be brief.", voice: "Aiden" }), {
  transport, model: "s2s-local", tracingDisabled: true,
  config: { audio: { input: { turnDetection: { type: "server_vad", interruptResponse: true } } } },
});

let audioBytes = 0;
const done = [];
const generic = [];
session.on("audio", (e) => { audioBytes += e.data.byteLength; });
transport.on("turn_done", (e) => done.push(e.response.output?.[0]?.content?.[0]?.transcript ?? ""));
transport.on("*", (raw) => { if (raw.type === "error") generic.push(raw); });

const timeout = setTimeout(() => { console.log("TIMEOUT"); process.exit(1); }, 15000);
await session.connect({ apiKey: "x", url });
session.sendMessage("Start the conversation now with a brief greeting.");
while (done.length < 1) await new Promise((r) => setTimeout(r, 50));
session.sendMessage("hello from the sdk");
while (done.length < 2) await new Promise((r) => setTimeout(r, 50));
clearTimeout(timeout);
console.log(JSON.stringify({ audioBytes, transcripts: done, errors: generic, history: session.history.length }));
session.close();
process.exit(audioBytes > 0 && done.length === 2 && generic.length === 0 ? 0 : 1);
