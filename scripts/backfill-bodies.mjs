/**
 * backfill-bodies.mjs — local bulk backfill of long-form article bodies.
 *
 * The cloud body pipeline (worker-body, LL-115/LL-116) generates ~20 bodies per
 * hourly cron, so draining a large backlog takes 1-2 days. This script does the
 * same generation LOCALLY with no cron / KV-write-budget / subrequest limits, so
 * the whole backlog finishes in ~1-2 hours.
 *
 * It reuses the EXACT worker contracts so the output is identical to the cloud
 * worker (no drift):
 *   - prompts:   worker/src/body-generate.ts (buildBodyPromptJa/En, cleanBodyText)
 *   - merge:     worker/src/bodies-file.ts   (mergeBodies, isRealBody, ...)
 *   - model:     Anthropic Claude API, claude-opus-5-5, effort=high (worker-body,
 *                worker/src/anthropic-client.ts)
 *
 * Safety vs the live collector (which commits data/bodies.json hourly):
 *   - Generated bodies are written incrementally to a gitignored cache
 *     (data/_body-backfill-cache.json) so an interrupted run resumes.
 *   - bodies.json is rebuilt by MERGING the cache into the CURRENT on-disk
 *     bodies.json (mergeBodies is additive + prunes only non-live ids), so a
 *     cloud-generated body landing during the run is preserved, not clobbered.
 *
 * Env (via `tsx --env-file-if-exists=.env.local`):
 *   ANTHROPIC_API_KEY  required, same as resummarize.mjs
 *   BODY_CONCURRENCY   default 6
 *   BODY_MAX_NEW       default Infinity (cap entries this run, for smoke tests)
 *   BODY_MODEL         default claude-opus-5-5 (claude-* only)
 *   BODY_EFFORT        default high (low | medium | high | xhigh | max)
 *   BODY_MAX_TOKENS    default 16000 (thinking + answer)
 *   BODY_TIMEOUT_MS    default 180000
 */
import fs from "node:fs";
import path from "node:path";
import {
  buildBodyPromptJa,
  buildBodyPromptEn,
  cleanBodyText,
} from "../worker/src/body-generate.ts";
import {
  parseBodies,
  serializeBodies,
  mergeBodies,
  isRealBody,
} from "../worker/src/bodies-file.ts";
import { hasSufficientBodySourceGrounding } from "../harness/pipeline/source-grounding.ts";
import {
  callAnthropicText,
  DEFAULT_ANTHROPIC_MODEL,
  parseAnthropicEffort,
  parseModelChain,
  parsePositiveInt,
  requireAnthropicKey,
} from "../worker/src/anthropic-client.ts";

const INDEX_PATH = "data/index.json";
const BODIES_PATH = "data/bodies.json";
const CACHE_PATH = "data/_body-backfill-cache.json";

const MODEL = parseModelChain(process.env.BODY_MODEL || DEFAULT_ANTHROPIC_MODEL, undefined)[0];
const EFFORT = parseAnthropicEffort(process.env.BODY_EFFORT, "high");
const MAX_TOKENS = parsePositiveInt(process.env.BODY_MAX_TOKENS, 16_000, "BODY_MAX_TOKENS");
const TIMEOUT_MS = Number(process.env.BODY_TIMEOUT_MS ?? 180_000);
const CONCURRENCY = Math.max(1, Number(process.env.BODY_CONCURRENCY ?? 6));
const MAX_NEW = Number(process.env.BODY_MAX_NEW ?? Infinity);
const MIN_BODY_CHARS = 120;
const FLUSH_EVERY = 8;

const BODY_SYSTEM_PROMPT =
  "You are a professional technology editor. Return only the plain-text body the user requested — no JSON, no code fences, no preamble.";

async function callBodyText(prompt) {
  const result = await callAnthropicText(
    { model: MODEL, systemPrompt: BODY_SYSTEM_PROMPT, userPrompt: prompt, maxTokens: MAX_TOKENS, effort: EFFORT },
    { anthropicKey: requireAnthropicKey(process.env), timeoutMs: TIMEOUT_MS },
  );
  const content = cleanBodyText(result.text);
  if (content.length < MIN_BODY_CHARS) throw new Error(`empty/short body (${content.length} chars)`);
  return content;
}

// --- cache (resume) ---------------------------------------------------------
function loadCache() {
  try {
    return JSON.parse(fs.readFileSync(CACHE_PATH, "utf8"));
  } catch {
    return {};
  }
}
function saveCache(cache) {
  const tmp = `${CACHE_PATH}.tmp`;
  fs.writeFileSync(tmp, JSON.stringify(cache, null, 2) + "\n");
  fs.renameSync(tmp, CACHE_PATH);
}

// --- bodies.json rebuild from cache (merge into CURRENT on-disk file) --------
function applyCacheToBodies(cache) {
  const existing = parseBodies(fs.readFileSync(BODIES_PATH, "utf8"));
  const idx = JSON.parse(fs.readFileSync(INDEX_PATH, "utf8"));
  const liveIds = new Set(idx.entries.map((e) => e.id));
  const newBodies = Object.entries(cache)
    .filter(([, b]) => isRealBody(b))
    .map(([id, b]) => ({ id, bodyJa: b.bodyJa, bodyEn: b.bodyEn, model: b.model, cachedAt: b.cachedAt }));
  const merged = mergeBodies(existing, newBodies, liveIds, new Date().toISOString());
  fs.writeFileSync(BODIES_PATH, serializeBodies(merged.payload));
  return merged;
}

// --- main -------------------------------------------------------------------
async function main() {
  // The historical backfill writes a single file directly and cannot commit
  // four related shards with the Publisher's main-SHA CAS.
  if (Array.from({ length: 4 }, (_, index) =>
    fs.existsSync(`data/bodies-${index}.json`)).some(Boolean)) {
    throw new Error("local single-file body backfill is retired after shard migration; use the Publisher body Queue");
  }
  const applyOnly = process.argv.includes("--apply-only");
  const cache = loadCache();

  if (applyOnly) {
    const merged = applyCacheToBodies(cache);
    console.log(`[backfill-bodies] apply-only: bodies.json now ${merged.payload.count} (added ${merged.added}, pruned ${merged.pruned})`);
    return;
  }

  const idx = JSON.parse(fs.readFileSync(INDEX_PATH, "utf8"));
  const existing = parseBodies(fs.readFileSync(BODIES_PATH, "utf8"));

  const isFallbackSummary = (s) => {
    const t = (s ?? "").trim();
    return !t || t.startsWith("このエントリは ") || t.includes("AI summary not yet available");
  };
  // Eligible = usable source excerpt AND real bilingual summary (publishable)
  // AND no real body yet AND not already cached this run.
  //
  // The excerpt gate is the same one worker-body applies before generation
  // (hasSufficientBodySourceGrounding). Without it this script generated a body
  // for every publishable entry, including the 53 with an entirely empty
  // contentSnippet, and wrote them straight into data/bodies.json.
  const eligible = idx.entries.filter((e) => {
    if (!hasSufficientBodySourceGrounding(e)) return false;
    if (isFallbackSummary(e.summaryJa) || isFallbackSummary(e.summaryEn)) return false;
    if (isRealBody(existing.bodies[e.id])) return false;
    if (isRealBody(cache[e.id])) return false;
    return true;
  });
  // Newest first so the visible top of the timeline fills in first.
  eligible.sort((a, b) => new Date(b.publishedAt || 0) - new Date(a.publishedAt || 0));
  const targets = Number.isFinite(MAX_NEW) ? eligible.slice(0, MAX_NEW) : eligible;

  console.log(
    `[backfill-bodies] live=${idx.entries.length} existing-bodies=${existing.count} cached=${Object.keys(cache).length} eligible=${eligible.length} targets=${targets.length}`,
  );
  // Fail fast with the explicit setup error before any work starts.
  requireAnthropicKey(process.env);
  console.log(`[backfill-bodies] model=${MODEL} effort=${EFFORT} concurrency=${CONCURRENCY} maxTokens=${MAX_TOKENS}`);
  if (targets.length === 0) {
    applyCacheToBodies(cache);
    console.log("[backfill-bodies] nothing to generate; bodies.json refreshed from cache");
    return;
  }

  let done = 0;
  let failed = 0;
  let sinceFlush = 0;
  const startMs = Date.now();
  let cursor = 0;

  async function worker(wid) {
    while (cursor < targets.length) {
      const i = cursor++;
      const e = targets[i];
      const promptEntry = {
        id: e.id,
        title: e.title,
        titleJa: e.titleJa,
        titleEn: e.titleEn,
        category: e.category,
        source: e.source,
        sourceType: e.sourceType,
        url: e.url,
        // contentSnippet is the ONLY factual source the prompt gets. Omitting it
        // (the previous behavior) meant contextLines() in
        // worker/src/body-generate.ts never emitted its
        // "収集元の抜粋 / Source excerpt:" line, so every body this script
        // produced was written from the title plus two AI summaries and nothing
        // else - the exact ungrounded-generation defect this cluster fixes.
        contentSnippet: e.contentSnippet,
        summaryJa: e.summaryJa,
        summaryEn: e.summaryEn,
        publishedAt: e.publishedAt,
        tags: e.tags,
      };
      try {
        const bodyJa = await callBodyText(buildBodyPromptJa(promptEntry));
        const bodyEn = await callBodyText(buildBodyPromptEn(promptEntry));
        if (bodyJa.trim().length < MIN_BODY_CHARS || bodyEn.trim().length < MIN_BODY_CHARS) {
          throw new Error(`incomplete (ja=${bodyJa.length}, en=${bodyEn.length})`);
        }
        cache[e.id] = { bodyJa, bodyEn, model: MODEL, cachedAt: new Date().toISOString() };
        done++;
        sinceFlush++;
      } catch (err) {
        failed++;
        console.warn(`[backfill-bodies] FAIL ${e.id} (${e.source}): ${err instanceof Error ? err.message : err}`);
      }
      if (sinceFlush >= FLUSH_EVERY) {
        sinceFlush = 0;
        saveCache(cache);
        const elapsed = (Date.now() - startMs) / 1000;
        const rate = done / Math.max(elapsed / 60, 0.001);
        const remain = targets.length - (done + failed);
        const eta = rate > 0 ? Math.ceil(remain / rate) : 0;
        console.log(`[backfill-bodies] progress ${done + failed}/${targets.length} (ok=${done} fail=${failed}) ~${rate.toFixed(1)}/min ETA~${eta}min`);
      }
    }
  }

  await Promise.all(Array.from({ length: CONCURRENCY }, (_, w) => worker(w)));
  saveCache(cache);
  const merged = applyCacheToBodies(cache);
  console.log(`[backfill-bodies] DONE ok=${done} fail=${failed} in ${((Date.now() - startMs) / 60000).toFixed(1)}min`);
  console.log(`[backfill-bodies] bodies.json now ${merged.payload.count} (added ${merged.added}, pruned ${merged.pruned})`);
}

main().catch((err) => {
  console.error("[backfill-bodies] fatal:", err);
  process.exit(1);
});
