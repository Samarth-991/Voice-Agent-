// Validate server events against the @openai/agents SDK's own zod schemas —
// the parser the browser actually runs. Reads a JSON array of events on stdin,
// prints failures, exits non-zero if any event would be treated as "generic"
// or would make the SDK's strict item parse throw.
// Usage: node tests/sdk_schema_check.mjs <path to agents-realtime/dist>
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
import path from "node:path";

const dist = process.argv[2];
const load = (file) => import(pathToFileURL(path.join(dist, file)).href);
const { realtimeServerEventSchema } = await load("openaiRealtimeEvents.mjs");
const { realtimeMcpCallItem } = await load("items.mjs");

const ITEM_EVENTS = new Set([
  "response.output_item.added", "response.output_item.done",
  "conversation.item.added", "conversation.item.done", "conversation.item.retrieved",
]);

const events = JSON.parse(readFileSync(0, "utf8"));
let failures = 0;
for (const event of events) {
  const result = realtimeServerEventSchema.safeParse(event);
  if (!result.success) {
    failures += 1;
    console.log(`${event.type}: ${JSON.stringify(result.error.issues)}`);
    continue;
  }
  // Mirror openaiRealtimeBase._onMessage: mcp items go through a strict .parse().
  const item = event.item;
  if (ITEM_EVENTS.has(event.type) && (item?.type === "mcp_call" || item?.type === "mcp_tool_call")) {
    const mcp = realtimeMcpCallItem.safeParse({
      itemId: item.id, type: item.type, status: "in_progress",
      arguments: item.arguments, name: item.name, output: item.output,
    });
    if (!mcp.success) {
      failures += 1;
      console.log(`${event.type} (mcp item): ${JSON.stringify(mcp.error.issues)}`);
    }
  }
}
console.log(`checked=${events.length} failures=${failures}`);
process.exit(failures ? 1 : 0);
