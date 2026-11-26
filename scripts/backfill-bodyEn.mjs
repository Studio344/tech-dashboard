/**
 * backfill-bodyEn.mjs — bodyEn のみを生成するバックフィル.
 *
 * 既存の bodyJa を入力コンテキストとして与え、英文記事だけを生成させる。
 * bodyJa + bodyEn を同時に返させるとレスポンスが max_tokens を超えて
 * JSON が途中で切れるため、専用スクリプトで分けている。
 *
 * Usage (Anthropic Claude API, worker/src/anthropic-client.ts):
 *   ANTHROPIC_API_KEY は .env.local などで設定しておく
 *   BACKFILL_MAX=50 npx tsx --env-file-if-exists=.env.local scripts/backfill-bodyEn.mjs
 *   SUMMARIZE_MODEL (既定 claude-opus-5-5, claude-* のみ) / SUMMARIZE_EFFORT (既定 high)
 */
import fs from "node:fs";
import {
  callAnthropicText,
  DEFAULT_ANTHROPIC_MODEL,
  parseAnthropicEffort,
  parseModelChain,
  parsePositiveInt,
  requireAnthropicKey,
} from "../worker/src/anthropic-client.ts";

const INDEX = "data/index.json";
const CACHE = "data/_summary-cache.json";
const MODEL = parseModelChain(process.env.SUMMARIZE_MODEL || DEFAULT_ANTHROPIC_MODEL, undefined)[0];
const EFFORT = parseAnthropicEffort(process.env.SUMMARIZE_EFFORT, "high");
const MAX_NEW = Number(process.env.BACKFILL_MAX ?? "30");
const MAX_TOKENS = parsePositiveInt(process.env.SUMMARIZE_MAX_TOKENS, 16_000, "SUMMARIZE_MAX_TOKENS");
const REQUEST_TIMEOUT_MS = Number(process.env.SUMMARIZE_TIMEOUT_MS ?? "180000");
const CONCURRENCY = Number(process.env.SUMMARIZE_CONCURRENCY ?? "4");

function buildPrompt(e) {
  return [
    `# Article`,
    `Title: ${e.title}`,
    `Category: ${e.category}`,
    `Source: ${e.source}`,
    `URL: ${e.url}`,
    ``,
    `# Japanese version (already written, for context)`,
    e.bodyJa || "",
    ``,
    `# Task`,
    `Write a native English long-form article (500-800 words) covering the same topic and depth as the Japanese version above.`,
    `Do NOT translate literally — write as a professional tech editor would in English.`,
    `Structure: lead paragraph (1-2 sentences on subject and significance), main body (key points / technical content / background, multiple paragraphs separated by \\n\\n), related context (background, ecosystem references, hedged speculation when appropriate).`,
    `Plain text only. No Markdown headings or list symbols. Paragraphs separated by \\n\\n.`,
    `Tone: neutral, fact-based. Use hedged language ("appears to", "may", "is reportedly") for speculation.`,
    ``,
    `Return ONLY the following JSON, no extra text:`,
    `{ "bodyEn": "..." }`,
  ].join("\n");
}

async function callModel(anthropicKey, entry) {
  const { text } = await callAnthropicText(
    {
      model: MODEL,
      systemPrompt: "You are a professional English-language tech editor. Return only the requested JSON.",
      userPrompt: buildPrompt(entry),
      maxTokens: MAX_TOKENS,
      effort: EFFORT,
    },
    { anthropicKey, timeoutMs: REQUEST_TIMEOUT_MS },
  );
  const m = text.match(/\{[\s\S]*\}/);
  if (!m) throw new Error("no JSON in response");
  const obj = JSON.parse(m[0]);
  const bodyEn = String(obj.bodyEn ?? "").trim();
  if (!bodyEn) throw new Error("empty bodyEn");
  return bodyEn;
}

async function runWithConcurrency(items, fn, concurrency) {
  const results = new Array(items.length);
  let idx = 0;
  async function worker() {
    while (true) {
      const i = idx++;
      if (i >= items.length) return;
      try {
        results[i] = { ok: true, value: await fn(items[i]) };
      } catch (e) {
        results[i] = { ok: false, err: e.message };
      }
    }
  }
  await Promise.all(Array.from({ length: concurrency }, worker));
  return results;
}

// --- Main ---
const idx = JSON.parse(fs.readFileSync(INDEX, "utf8"));
const cache = JSON.parse(fs.readFileSync(CACHE, "utf8"));

const missing = idx.entries.filter(
  (e) => (e.bodyJa || "").trim() && !(e.bodyEn || "").trim(),
);
console.log(
  `[bodyEn] total=${idx.entries.length} has-bodyJa-no-bodyEn=${missing.length}`,
);

const target = missing.slice(0, MAX_NEW);
if (target.length === 0) {
  console.log("[bodyEn] nothing to do");
  process.exit(0);
}

const anthropicKey = requireAnthropicKey(process.env);
console.log(`[bodyEn] fetching ${target.length} entries (concurrency=${CONCURRENCY})...`);

const t0 = Date.now();
const results = await runWithConcurrency(
  target,
  async (e) => {
    const bodyEn = await callModel(anthropicKey, e);
    return { url: e.url, bodyEn };
  },
  CONCURRENCY,
);

let ok = 0,
  err = 0;
const byUrl = new Map();
for (let i = 0; i < results.length; i++) {
  const r = results[i];
  if (r.ok) {
    byUrl.set(r.value.url, r.value.bodyEn);
    ok++;
  } else {
    err++;
    console.warn(`[bodyEn] err ${target[i].url}: ${r.err}`);
  }
}

// Write into entries + cache
for (const e of idx.entries) {
  const bodyEn = byUrl.get(e.url);
  if (bodyEn) {
    e.bodyEn = bodyEn;
    if (cache[e.url]) {
      cache[e.url].bodyEn = bodyEn;
      cache[e.url].cachedAt = new Date().toISOString();
    }
  }
}
idx.generatedAt = new Date().toISOString();
fs.writeFileSync(INDEX, JSON.stringify(idx, null, 2) + "\n");
fs.writeFileSync(CACHE, JSON.stringify(cache, null, 2) + "\n");

const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
const remaining = idx.entries.filter(
  (e) => (e.bodyJa || "").trim() && !(e.bodyEn || "").trim(),
).length;
console.log(
  `[bodyEn] ok=${ok} err=${err} elapsed=${elapsed}s remaining=${remaining}`,
);
