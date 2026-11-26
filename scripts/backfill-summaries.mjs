// Backfill REAL bilingual AI summaries for the "snippet masquerade" entries
// (Issue #1, LL-118). These ~909 live entries were NEVER summarized by any AI
// model: the old normalize.placeholderSummary() sliced a raw RSS snippet at
// 120/200 chars (JA sources: summaryEn = the raw title) and the worker gate
// treated that non-empty/unmarked text as COMPLETE, so they were never queued.
//
// This script regenerates them LOCALLY using the SAME summary-only contract the
// worker queue uses (worker/src/prompt.ts buildSummaryPrompt + parseResponse),
// so the output is byte-compatible with the production pipeline. The default
// model is claude-opus-5-5 (SUMMARIZE_MODEL overrides it; only claude-* ids
// are accepted). Running locally avoids the Worker wall-time budget (LL-031/106).
//
// AUTH: Anthropic Claude API (worker/src/anthropic-client.ts). ANTHROPIC_API_KEY
//   を .env.local などで設定しておく。未設定なら明示エラーで終了する。
//
// RESUME: every successful generation is written to a gitignored cache
//   data/_summary-backfill-cache.json keyed by entry id, so re-running after an
//   interruption only regenerates what is missing (LL-117).
//
// DURABILITY: like the migration, real summaries survive re-collection ONLY once
// the FIXED worker is deployed (R-008/LL-073). After deploy, re-collection emits
// an empty summaryJa and entry-merge fillText() keeps the prior real summary.
//
// Usage:
//   npx tsx scripts/backfill-summaries.mjs --dry           # list targets, no API calls
//   npx tsx --env-file-if-exists=.env.local scripts/backfill-summaries.mjs # generate + write
//   SUMMARIZE_MODEL=claude-opus-5-5 SUMMARIZE_EFFORT=medium npx tsx ... (claude-* only)
//   ... --limit 20                                         # cap this run (smoke test)
import fs from "node:fs";
import { buildSummaryPrompt, parseResponse } from "../worker/src/prompt.ts";
import {
  callAnthropicText,
  DEFAULT_ANTHROPIC_MODEL,
  parseAnthropicEffort,
  parseModelChain,
  parsePositiveInt,
  requireAnthropicKey,
} from "../worker/src/anthropic-client.ts";

const DRY = process.argv.includes("--dry");
const limitArg = process.argv.indexOf("--limit");
const LIMIT = limitArg !== -1 ? Number(process.argv[limitArg + 1]) : Infinity;

const FILE = "data/index.json";
const CACHE_FILE = "data/_summary-backfill-cache.json";

const MODEL = parseModelChain(process.env.SUMMARIZE_MODEL || DEFAULT_ANTHROPIC_MODEL, undefined)[0];
const EFFORT = parseAnthropicEffort(process.env.SUMMARIZE_EFFORT, "medium");
const MAX_TOKENS = parsePositiveInt(process.env.SUMMARIZE_MAX_TOKENS, 16_000, "SUMMARIZE_MAX_TOKENS");
const TIMEOUT_MS = Number(process.env.SUMMARIZE_TIMEOUT_MS ?? "120000");
const CONCURRENCY = Number(process.env.SUMMARIZE_CONCURRENCY ?? "8");

// ---- masquerade detector (identical to migrate-stuck-summaries-to-pending.mjs) ----
const FB_JA_PREFIX = "このエントリは ";
const FB_JA_NEEDLE = "AI 要約が未生成";
const FB_EN_NEEDLE = "AI summary not yet available";
const isPending = (s) => {
  s = s || "";
  return s.startsWith(FB_JA_PREFIX) || s.includes(FB_JA_NEEDLE) || s.includes(FB_EN_NEEDLE);
};
const JA_TERMINAL = /[。．.!?！？」』）)\u2026]$/;
const EN_TERMINAL = /[.!?"'\u2019\u201d)\]\u2026]$/;
function isMasquerade(e) {
  const ja = e.summaryJa || "";
  const en = e.summaryEn || "";
  if (isPending(ja) || isPending(en)) return false;
  return (!!en && en === e.title) || (ja.length === 120 && !JA_TERMINAL.test(ja)) || (en.length === 200 && !EN_TERMINAL.test(en));
}

// ---- Anthropic call ----
async function generate(anthropicKey, entry) {
  const { text } = await callAnthropicText(
    {
      model: MODEL,
      systemPrompt:
        "あなたは技術記事を日本語と英語の両方で要約するエディターです。指示された JSON 形式のみを返してください。You are an editor who summarises tech articles in both Japanese and English. Return only the requested JSON.",
      userPrompt: buildSummaryPrompt(entry),
      maxTokens: MAX_TOKENS,
      effort: EFFORT,
    },
    { anthropicKey, timeoutMs: TIMEOUT_MS },
  );
  const parsed = parseResponse(text);
  if (!parsed.summaryJa.trim() || !parsed.summaryEn.trim()) {
    throw new Error("incomplete summary (empty summaryJa/summaryEn)");
  }
  return parsed;
}

// ---- main ----
const data = JSON.parse(fs.readFileSync(FILE, "utf8"));
const targets = data.entries.filter(isMasquerade);
console.log(`total entries: ${data.entries.length}`);
console.log(`masquerade targets: ${targets.length}`);
console.log(`model: anthropic:${MODEL} (effort ${EFFORT}) | concurrency: ${CONCURRENCY} | max_tokens: ${MAX_TOKENS}`);

if (DRY) {
  console.log("\n(dry run -- no API calls, no file written)");
  process.exit(0);
}

const anthropicKey = requireAnthropicKey(process.env);

const cache = fs.existsSync(CACHE_FILE) ? JSON.parse(fs.readFileSync(CACHE_FILE, "utf8")) : {};
const pending = targets.filter((e) => !cache[e.id]).slice(0, LIMIT === Infinity ? undefined : LIMIT);
console.log(`already cached: ${targets.length - targets.filter((e) => !cache[e.id]).length} | to generate now: ${pending.length}\n`);

let ok = 0;
let fail = 0;
let done = 0;
function saveCache() {
  fs.writeFileSync(CACHE_FILE, JSON.stringify(cache, null, 2) + "\n", "utf8");
}

async function worker(queue) {
  for (;;) {
    const entry = queue.shift();
    if (!entry) return;
    try {
      const parsed = await generate(anthropicKey, entry);
      cache[entry.id] = {
        titleJa: parsed.titleJa,
        summaryJa: parsed.summaryJa,
        summaryEn: parsed.summaryEn,
        importance: parsed.importance,
        extraTags: parsed.extraTags,
        model: MODEL,
        cachedAt: new Date().toISOString(),
      };
      ok++;
    } catch (err) {
      fail++;
      console.warn(`  FAIL ${entry.source} ${entry.id}: ${String(err.message || err).slice(0, 120)}`);
    }
    done++;
    if (done % 10 === 0) {
      saveCache();
      console.log(`  progress ${done}/${pending.length} (ok ${ok}, fail ${fail})`);
    }
  }
}

const queue = [...pending];
await Promise.all(Array.from({ length: Math.min(CONCURRENCY, queue.length) }, () => worker(queue)));
saveCache();

// Apply every cached summary back into index.json (covers this run + prior runs).
let applied = 0;
for (const e of data.entries) {
  const c = cache[e.id];
  if (!c) continue;
  e.summaryJa = c.summaryJa;
  e.summaryEn = c.summaryEn;
  if (c.titleJa && c.titleJa.trim()) e.titleJa = c.titleJa;
  if (Array.isArray(c.extraTags) && c.extraTags.length) {
    const tags = new Set([...(e.tags || []), ...c.extraTags]);
    e.tags = [...tags].slice(0, 12);
  }
  if (c.importance) e.importance = c.importance;
  applied++;
}
fs.writeFileSync(FILE, JSON.stringify(data, null, 2) + "\n", "utf8");

console.log(`\ndone: generated ok ${ok}, fail ${fail}`);
console.log(`applied ${applied} cached summaries to ${FILE}`);
if (fail > 0) console.log(`re-run to retry the ${fail} failures (cache resumes the successful ones).`);
