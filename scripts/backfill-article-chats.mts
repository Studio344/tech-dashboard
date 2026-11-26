/**
 * scripts/backfill-article-chats.mts
 *
 * Backfills the 記事ディスカッション (article chat) for entries that already
 * carry a real body in data/bodies.json but predate the chat feature.
 *
 * Direct KV writes are disabled: this local process cannot attest to the
 * deployed body consumer's compiled chat revision. Existing sidecar chats
 * remain untouched; the Publisher uses bounded body Queue repair for missing
 * or incompatible chats instead.
 *
 * Usage:
 *   npx tsx scripts/backfill-article-chats.mts                # dry-run report
 *   npx tsx scripts/backfill-article-chats.mts --limit 25     # dry-run, first 25
 *   --apply is intentionally rejected; do not bypass the Queue provenance gate.
 *
 * R-028: do not run concurrently with other automation in this checkout.
 */
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { spawnSync } from "node:child_process";
import {
  buildArticleChatPrompt,
  chatGroundingText,
  parseArticleChat,
  validateArticleChat,
} from "../worker/src/article-chat.ts";
import {
  callAnthropicText,
  DEFAULT_ANTHROPIC_MODEL,
  parseModelChain,
  requireAnthropicKey,
} from "../worker/src/anthropic-client.ts";
import { hasMaterialBodyGroundingConflict } from "../harness/pipeline/source-grounding.ts";
import { isRealBody, type BodyRecord } from "../worker/src/bodies-file.ts";
import { readBodyStorageFromDisk } from "./body-storage-node.mjs";

const KV_NAMESPACE_ID = "6d67debb991742efadfec473a121f5fc";
/** Same model as worker-body/wrangler.toml (Anthropic Claude API only). */
const MODEL_CHAIN = parseModelChain(process.env.BODY_MODEL || DEFAULT_ANTHROPIC_MODEL, process.env.BODY_MODEL_FALLBACKS);
const MAX_TOKENS = 16_000;
const CALL_TIMEOUT_MS = 180_000;
const CHAT_SYSTEM =
  "You are a professional technology editor. Return only the JSON the user requested — no code fences, no preface, no commentary.";

// ------------------------------------------------------------------- args ---
const args = process.argv.slice(2);
let apply = false;
let limit = 10;
for (let i = 0; i < args.length; i++) {
  if (args[i] === "--apply") apply = true;
  else if (args[i] === "--limit" && args[i + 1]) limit = Math.max(1, Number(args[++i]));
  else {
    console.error("Usage: npx tsx scripts/backfill-article-chats.mts [--apply] [--limit N]");
    process.exit(1);
  }
}
if (apply) {
  console.error("ERR: direct KV chat writes cannot prove the deployed consumer revision; use bounded Publisher Queue repair");
  process.exit(1);
}

// ---------------------------------------------------------------- anthropic --
const envLocalPath = join(process.cwd(), ".env.local");
const envLocalKey = existsSync(envLocalPath)
  ? readFileSync(envLocalPath, "utf8").match(/^ANTHROPIC_API_KEY=(.+)$/m)?.[1]?.trim()
  : undefined;
let anthropicKey: string;
try {
  anthropicKey = requireAnthropicKey({ ANTHROPIC_API_KEY: process.env.ANTHROPIC_API_KEY ?? envLocalKey });
} catch (err) {
  console.error(`ERR: ${err instanceof Error ? err.message : String(err)} (nothing was modified)`);
  process.exit(1);
}

async function generateChat(entry: Record<string, unknown>): Promise<ReturnType<typeof validateArticleChat>> {
  const prompt = buildArticleChatPrompt(entry as never);
  for (const model of MODEL_CHAIN) {
    try {
      const { text: raw } = await callAnthropicText(
        {
          model,
          systemPrompt: CHAT_SYSTEM,
          userPrompt: prompt,
          maxTokens: MAX_TOKENS,
          effort: "high",
        },
        { anthropicKey, timeoutMs: CALL_TIMEOUT_MS },
      );
      const turns = parseArticleChat(raw);
      if (!turns) continue; // structural failure: try the next model once
      const text = chatGroundingText(turns);
      if (hasMaterialBodyGroundingConflict(entry as never, { bodyJa: text.ja, bodyEn: text.en })) {
        return null; // grounding conflict is a content problem, not a model one
      }
      return turns;
    } catch (err) {
      console.warn(`  ${model} failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  }
  return null;
}

// ---------------------------------------------------------------------- kv ---
function kvKeyArgs(key: string): string[] {
  return ["--namespace-id", KV_NAMESPACE_ID, "--remote", key];
}
async function kvGet(key: string): Promise<Record<string, unknown> | null> {
  const out = spawnSync("npx", ["--yes", "wrangler@4.85.0", "kv", "key", "get", ...kvKeyArgs(key)], {
    encoding: "utf8",
    maxBuffer: 10 * 1024 * 1024,
  });
  if (out.status !== 0) return null; // miss (wrangler exits nonzero on missing key)
  try {
    return JSON.parse(out.stdout) as Record<string, unknown>;
  } catch {
    return null;
  }
}
function kvPut(key: string, value: string): boolean {
  const out = spawnSync(
    "npx",
    ["--yes", "wrangler@4.85.0", "kv", "key", "put", ...kvKeyArgs(key), value],
    { encoding: "utf8", maxBuffer: 10 * 1024 * 1024 },
  );
  if (out.status !== 0) console.error(`  KV put failed: ${out.stderr.slice(0, 200)}`);
  return out.status === 0;
}
async function bodyCacheKeyForUrl(url: string): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(url));
  const bytes = new Uint8Array(digest);
  let hex = "";
  for (const byte of bytes) hex += byte.toString(16).padStart(2, "0");
  return `b:${hex}`;
}

// -------------------------------------------------------------------- main ---
const index = JSON.parse(readFileSync(join(process.cwd(), "data", "index.json"), "utf8")) as {
  entries: Array<Record<string, unknown> & { id: string; url: string }>;
};
const bodies = readBodyStorageFromDisk() as {
  bodies: Record<string, BodyRecord>;
};

const candidates = index.entries.filter((entry) => {
  const record = bodies.bodies[entry.id];
  return record && isRealBody(record) && !validateArticleChat(record.chat);
});
console.log(`bodied entries without a chat: ${candidates.length}`);
const batch = candidates.slice(0, limit);
console.log(`${apply ? "APPLY" : "DRY-RUN"}: processing ${batch.length} (limit ${limit})\n`);

let generated = 0;
let skippedInKv = 0;
let failed = 0;
for (const entry of batch) {
  const label = `${entry.id} ${String(entry.title ?? "").slice(0, 50)}`;
  const key = await bodyCacheKeyForUrl(entry.url);
  if (apply) {
    const existing = await kvGet(key);
    if (existing && validateArticleChat(existing.chat)) {
      skippedInKv++;
      console.log(`SKIP (chat already in KV)  ${label}`);
      continue;
    }
    const chat = await generateChat(entry);
    if (!chat) {
      failed++;
      console.log(`FAIL                       ${label}`);
      continue;
    }
    const record = bodies.bodies[entry.id]!;
    // Body text comes from the published record: mergeBodies grafts only the
    // chat, but keeping the KV entry self-consistent costs nothing.
    const cacheEntry = {
      bodyJa: record.bodyJa,
      bodyEn: record.bodyEn,
      chat,
      model: `chat-backfill:${MODEL_CHAIN[0]}`,
      cachedAt: new Date().toISOString(),
    };
    if (kvPut(key, JSON.stringify(cacheEntry))) {
      generated++;
      console.log(`OK   ${chat.length} turns        ${label}`);
    } else {
      failed++;
    }
  } else {
    console.log(`would process              ${label}`);
  }
}

console.log(
  `\n${apply ? "APPLIED" : "DRY-RUN"}: generated=${generated} skippedInKv=${skippedInKv} failed=${failed} remaining=${candidates.length - batch.length}`,
);
if (!apply) console.log("Re-run with --apply to generate and write to KV.");
