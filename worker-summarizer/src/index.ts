/**
 * tech-dashboard-summarizer — Cloudflare Worker Queue consumer.
 *
 * Receives one entry per invocation from the SUMMARY_QUEUE producer
 * (tech-dashboard-harness). For each message:
 *   1. Call the Anthropic Claude API (worker/src/anthropic-client.ts) with the
 *      compact bilingual summary prompt. The key is the ANTHROPIC_API_KEY
 *      secret; without it the job fails with an explicit setup error.
 *   2. If the response is incomplete, retry once with a compact recovery prompt.
 *   3. Parse the response and write it into SUMMARY_CACHE KV under the entry URL.
 *
 * Why a separate Worker (LL-037):
 *   The harness Worker hit Cloudflare's default 30 s/invocation CPU cap when
 *   running summarize alongside publishHistoryFiles. Splitting summarize into
 *   its own Queue consumer gives summary generation an isolated retry and
 *   timeout budget.
 *
 * Failure handling:
 *   - API timeout / error / missing key: throw -> Cloudflare Queues retries
 *     with exponential backoff (max_retries=2, see wrangler.toml).
 *   - Incomplete model output: retry once in-process with a compact JSON-only
 *     prompt before handing the message back to Cloudflare Queues.
 *   - After max_retries, message goes to the DLQ for manual triage.
 */
import type { NormalizedEntry } from "../../harness/types.ts";
import { hasUsableGroundedBilingualSummary } from "../../harness/pipeline/summary-quality.ts";
import { hasSufficientSourceGrounding } from "../../harness/pipeline/source-grounding.ts";
import { buildSummaryPrompt, parseResponse } from "../../worker/src/prompt.ts";
import type { SummaryJob } from "../../worker/src/summary-queue.ts";
import {
  type CacheEntry,
  putCacheEntry,
  UNVERSIONED_JOB_FINGERPRINT,
} from "../../worker/src/kv-cache.ts";
import {
  callAnthropicText,
  DEFAULT_ANTHROPIC_MODEL,
  type AnthropicEffort,
  parseAnthropicEffort,
  parseModelChain,
  parsePositiveInt,
  readAnthropicKey,
  requireAnthropicKey,
  ANTHROPIC_API_KEY_MISSING,
} from "../../worker/src/anthropic-client.ts";

interface Env {
  SUMMARY_CACHE: KVNamespace;
  /** Secret (`wrangler secret put ANTHROPIC_API_KEY`). Required. */
  ANTHROPIC_API_KEY?: string;
  /** Claude model id (default claude-opus-5-5). */
  SUMMARIZE_MODEL?: string;
  /** Comma-separated ordered fallback Claude models tried when a call errors
   * or the output stays incomplete. See parseModelChain. */
  SUMMARIZE_MODEL_FALLBACKS?: string;
  /** low | medium | high | xhigh | max (default medium). */
  SUMMARIZE_EFFORT?: string;
  SUMMARIZE_TIMEOUT_MS?: string;
  /** Covers thinking + answer (default 16000). */
  SUMMARIZE_MAX_TOKENS?: string;
}

const SYSTEM_PROMPT =
  "You are a bilingual technical editor. Always return only the JSON object the user requested. Do not include code fences, prose preface, or commentary. Use natural Japanese for *Ja fields and native English for *En fields.";

export type { SummaryJob } from "../../worker/src/summary-queue.ts";

const DEFAULT_TIMEOUT_MS = 120_000;
// max_tokens covers thinking + the visible answer on current Claude models.
const DEFAULT_MAX_TOKENS = 16_000;
const DEFAULT_EFFORT: AnthropicEffort = "medium";
const ISSUE_KEY = "summarizer.issue.v1";
const ISSUE_TTL_SECONDS = 6 * 60 * 60;
const RECENT_ISSUE_MS = 60 * 60_000;
const ERROR_REPEAT_THRESHOLD = 3;
type IssueScope = "entry" | "runtime";
const RECOVERY_TIMEOUT_MS = 90_000;

async function callModel(
  config: SummaryLlmConfig,
  model: string,
  prompt: string,
  timeoutMs: number,
): Promise<CacheEntry> {
  const result = await callAnthropicText(
    {
      model,
      systemPrompt: SYSTEM_PROMPT,
      userPrompt: prompt,
      maxTokens: config.maxTokens,
      effort: config.effort,
    },
    { anthropicKey: config.anthropicKey, timeoutMs },
  );
  return {
    ...parseResponse(result.text),
    // The model that actually answered (differs after a refusal fallback).
    model: result.model,
    cachedAt: new Date().toISOString(),
  };
}

export function isCompleteCacheEntry(entry: CacheEntry): boolean {
  return Boolean(
    entry.titleJa.trim() &&
      entry.summaryJa.trim() &&
      entry.summaryEn.trim() &&
      entry.bodyJa.trim() &&
      entry.bodyEn.trim(),
  );
}

/**
 * The dashboard publishes a real (non-fallback) entry from the SUMMARY alone:
 * a Japanese title plus a JA and EN summary. The long bilingual body is shown
 * only on the article detail page and is acceptable as a deterministic fallback
 * until a real one is generated. So the queue consumer must persist an entry as
 * soon as the summary is complete — discarding a perfectly good summary because
 * the (longest, most truncation-prone) body field came back empty was the main
 * generation bottleneck: most fallback entries had NO KV summary at all and
 * stayed boilerplate forever (LL-104).
 */
export function isSummaryComplete(
  entry: CacheEntry,
  source: Partial<NormalizedEntry> = {},
): boolean {
  return Boolean(
    entry.titleJa.trim() &&
      hasUsableGroundedBilingualSummary(
        source,
        {
          ...entry,
          title: source.title,
          titleJa: entry.titleJa || source.titleJa,
          titleEn: source.titleEn,
        },
      ),
  );
}

export interface SummaryLlmConfig {
  anthropicKey: string;
  effort: AnthropicEffort;
  chain: string[];
  maxTokens: number;
}

/** Settings shared by the queue path and /health; never touches the key. */
function resolveSummarySettings(env: Env): Omit<SummaryLlmConfig, "anthropicKey"> {
  return {
    effort: parseAnthropicEffort(env.SUMMARIZE_EFFORT, DEFAULT_EFFORT),
    chain: parseModelChain(
      env.SUMMARIZE_MODEL || DEFAULT_ANTHROPIC_MODEL,
      env.SUMMARIZE_MODEL_FALLBACKS,
    ),
    maxTokens: parsePositiveInt(env.SUMMARIZE_MAX_TOKENS, DEFAULT_MAX_TOKENS, "SUMMARIZE_MAX_TOKENS"),
  };
}

/**
 * Claude API settings for one job. Throws the explicit setup error when the
 * ANTHROPIC_API_KEY secret is missing, and on any invalid model / effort /
 * token setting, so a misconfiguration fails visibly instead of silently.
 */
export function resolveSummaryLlmConfig(env: Env): SummaryLlmConfig {
  return { anthropicKey: requireAnthropicKey(env), ...resolveSummarySettings(env) };
}

async function processJob(env: Env, job: SummaryJob): Promise<void> {
  // Per-URL KV (LL-038): single KV.put per entry, no read-modify-write.
  // The producer (harness Worker) already filters out cached entries before
  // enqueuing, so duplicate work is rare. Each key is independent so parallel
  // consumer invocations can never clobber each other's writes.

  const config = resolveSummaryLlmConfig(env);
  const { chain } = config;
  const timeoutMs = Number(env.SUMMARIZE_TIMEOUT_MS ?? DEFAULT_TIMEOUT_MS);
  if (!hasSufficientSourceGrounding(job.entry)) {
    throw new Error(`insufficient source grounding for ${job.url}`);
  }

  // Summary-only prompt (LL-106). the Claude fallback emits opaque reasoning
  // tokens that count against max_tokens; asking for a long bilingual body in
  // the same call exhausts the budget and the chat endpoint returns
  // {"choices":[]} (empty) -> "incomplete summary" -> ZERO summaries written.
  // We request only title + JA/EN summary so reasoning + answer fit. Long-form
  // body generation is handled by the separate body-file pipeline.
  //
  // One attempt per model in the configured chain; a single-model chain keeps
  // the historical same-model retry so behavior without fallbacks is
  // unchanged. Later attempts use the tighter recovery timeout: they exist to
  // rescue the entry, not to double the invocation's worst-case wall time.
  const attempts = chain.length === 1 ? [chain[0]!, chain[0]!] : chain;
  let entry: CacheEntry | undefined;
  let lastError: unknown;
  for (const [attemptIndex, attemptModel] of attempts.entries()) {
    if (attemptIndex > 0) {
      console.warn(
        `[summarizer] attempt ${attemptIndex + 1}/${attempts.length} (${attemptModel}) for ${job.url}: previous output was incomplete or failed`,
      );
    }
    try {
      entry = await callModel(
        config,
        attemptModel,
        buildSummaryPrompt(job.entry),
        attemptIndex === 0 ? timeoutMs : Math.min(timeoutMs, RECOVERY_TIMEOUT_MS),
      );
      if (isSummaryComplete(entry, job.entry)) break;
    } catch (err) {
      // A per-model API failure (unsupported model, 4xx/5xx, timeout) advances
      // the chain instead of failing the job: that is what the fallbacks are
      // for. If every model errored, rethrow the last error below so Queue
      // retry semantics stay exactly as before.
      lastError = err;
      entry = undefined;
    }
  }
  if (!entry) throw lastError ?? new Error(`all summary models failed for ${job.url}`);

  // Persist as soon as the summary is complete, even if the long-form body is
  // still missing after all recovery attempts. The collector merges entries on
  // title+summaryJa+summaryEn and fills bodies deterministically; throwing away
  // a complete summary because the body truncated left most fallback entries
  // with no KV summary at all and was the real "stuck summaries" cause (LL-104).
  if (!isSummaryComplete(entry, job.entry)) {
    throw new Error(`incomplete summary for ${job.url}`);
  }

  await putCacheEntry(env.SUMMARY_CACHE, job.url, {
    ...entry,
    publisherContractFingerprint:
      job.publisherContractFingerprint ?? UNVERSIONED_JOB_FINGERPRINT,
  });
  await clearIssue(env, job.url);
}

function issueSummary(err: unknown): string {
  const text = err instanceof Error ? `${err.name}: ${err.message}` : String(err);
  return text.slice(0, 500);
}

function classifyIssueScope(err: unknown): IssueScope {
  return issueSummary(err).toLowerCase().includes("incomplete summary for ")
    ? "entry"
    : "runtime";
}

function issueScopeFromRecord(issue: Record<string, unknown> | null): IssueScope {
  if (issue?.scope === "entry" || issue?.scope === "runtime") return issue.scope;
  return classifyIssueScope(issue?.error ?? "");
}

async function writeIssue(
  env: Env,
  status: "retry" | "deferred",
  job: SummaryJob,
  err: unknown,
): Promise<void> {
  try {
    const existing = await env.SUMMARY_CACHE.get<Record<string, unknown>>(ISSUE_KEY, "json");
    const scope = classifyIssueScope(err);
    const sameIssue =
      existing?.status === status &&
      existing?.url === job.url &&
      issueScopeFromRecord(existing) === scope;
    const repeatCount =
      sameIssue && typeof existing?.repeatCount === "number" && Number.isFinite(existing.repeatCount)
        ? existing.repeatCount + 1
        : 1;
    await env.SUMMARY_CACHE.put(
      ISSUE_KEY,
      JSON.stringify({
        ok: status === "deferred" || scope === "entry" || repeatCount < ERROR_REPEAT_THRESHOLD,
        status,
        scope,
        at: new Date().toISOString(),
        url: job.url,
        source: job.entry.source,
        category: job.entry.category,
        repeatCount,
        error: issueSummary(err),
      }),
      { expirationTtl: ISSUE_TTL_SECONDS },
    );
  } catch (issueErr) {
    console.warn("[summarizer] issue heartbeat write failed:", issueErr);
  }
}

async function clearIssue(env: Env, url: string): Promise<void> {
  try {
    const existing = await env.SUMMARY_CACHE.get<Record<string, unknown>>(ISSUE_KEY, "json");
    if (existing?.url === url) {
      await env.SUMMARY_CACHE.delete(ISSUE_KEY);
    }
  } catch (issueErr) {
    console.warn("[summarizer] issue heartbeat clear failed:", issueErr);
  }
}

export default {
  async fetch(req: Request, env: Env): Promise<Response> {
    const url = new URL(req.url);
    if (url.pathname === "/health" && req.method === "GET") {
      const cache = env.SUMMARY_CACHE;
      const cacheBinding = Boolean(cache);
      const issue = cache
        ? await cache.get<Record<string, unknown>>(ISSUE_KEY, "json")
        : null;
      const issueAt = typeof issue?.at === "string" ? Date.parse(issue.at) : Number.NaN;
      const recentIssue = Number.isFinite(issueAt) && Date.now() - issueAt <= RECENT_ISSUE_MS;
      const issueStatus = typeof issue?.status === "string" ? issue.status : null;
      const issueScope = issueScopeFromRecord(issue);
      const repeatCount =
        typeof issue?.repeatCount === "number" && Number.isFinite(issue.repeatCount)
          ? issue.repeatCount
          : recentIssue && issueStatus === "retry"
          ? 1
          : 0;
      const issueSeverity =
        recentIssue &&
        issueStatus === "retry" &&
        issueScope === "runtime" &&
        repeatCount >= ERROR_REPEAT_THRESHOLD
          ? "error"
          : recentIssue
          ? "warn"
          : "ok";
      const anthropicSecretConfigured = readAnthropicKey(env) !== null;
      let settings: ReturnType<typeof resolveSummarySettings> | null = null;
      let configError: string | null = null;
      try {
        settings = resolveSummarySettings(env);
      } catch (err) {
        configError = err instanceof Error ? err.message : String(err);
      }
      const ok =
        cacheBinding && anthropicSecretConfigured && settings !== null && issueSeverity !== "error";
      return Response.json(
        {
          ok,
          role: "queue-consumer",
          provider: "anthropic",
          model: settings?.chain[0] ?? null,
          modelFallbacks: settings?.chain.slice(1) ?? [],
          effort: settings?.effort ?? null,
          timeoutMs: Number(env.SUMMARIZE_TIMEOUT_MS ?? DEFAULT_TIMEOUT_MS),
          maxTokens: settings?.maxTokens ?? null,
          cacheBinding,
          anthropicSecretConfigured,
          configError: configError ?? (anthropicSecretConfigured ? null : ANTHROPIC_API_KEY_MISSING),
          recentIssue,
          issueSeverity,
          issueScope,
          issue,
        },
        {
          status: ok ? 200 : 503,
          headers: {
            "access-control-allow-origin": "*",
            "cache-control": "no-store",
          },
        },
      );
    }
    return new Response("tech-dashboard summarizer worker. Queue consumer is active.", {
      status: 200,
      headers: { "content-type": "text/plain;charset=UTF-8" },
    });
  },

  async queue(batch: MessageBatch<SummaryJob>, env: Env): Promise<void> {
    for (const msg of batch.messages) {
      try {
        await processJob(env, msg.body);
        msg.ack();
      } catch (err) {
        const summary = err instanceof Error ? `${err.name}: ${err.message}` : String(err);
        // LL-043: KV daily write cap (1000/day on free tier) returns
        // "KV put() limit exceeded for the day". Retrying just burns through
        // max_retries and pushes the message to the DLQ — which is exactly
        // what we don't want for a transient daily-rate problem. Ack instead;
        // the next cron run after UTC midnight will re-enqueue any still-
        // fallback entry.
        if (summary.includes("KV put() limit exceeded")) {
          console.warn(`[summarizer] ack ${msg.body.url}: daily KV write cap reached, will retry tomorrow`);
          await writeIssue(env, "deferred", msg.body, summary);
          msg.ack();
        } else {
          console.warn(`[summarizer] retry ${msg.body.url}: ${summary}`);
          await writeIssue(env, "retry", msg.body, summary);
          msg.retry();
        }
      }
    }
  },
} satisfies ExportedHandler<Env, SummaryJob>;
