/**
 * tech-dashboard-body — Cloudflare Worker Queue consumer (body-file Phase B, LL-115).
 *
 * Receives one entry per invocation from the BODY_QUEUE producer
 * (tech-dashboard-harness). For each message:
 *   1. Generate the Japanese body and the English body as TWO SEPARATE
 *      plain-text calls to the Anthropic Claude API (claude-opus-5-5 by
 *      default, worker/src/anthropic-client.ts). The key is the
 *      ANTHROPIC_API_KEY secret; without it the job fails with an explicit
 *      setup error.
 *   2. Write { bodyJa, bodyEn, model, cachedAt } to the SUMMARY_CACHE KV
 *      namespace under a `b:` per-URL key.
 *
 * Why a separate Worker (mirrors the summarizer, LL-037/LL-108):
 *   Body generation is the heaviest LLM work (two long opus calls per entry).
 *   Keeping it in its own Queue consumer (a) leaves the finally-stable
 *   summarizer untouched, (b) gives body jobs an independent timeout/retry
 *   budget, and (c) isolates body throughput tuning from summaries.
 *
 * Why two single-language calls (LL-106/LL-115):
 *   Asking a reasoning model for both long bodies in one JSON response risks
 *   the reasoning-loop-empties failure. Two plain-text single-language calls
 *   each finish cleanly (verified by live API probe).
 *
 * Failure handling mirrors the summarizer: API timeout/error/missing key -> throw
 * -> Cloudflare Queues retries with backoff; KV daily write cap -> ack and
 * retry tomorrow (LL-043); persistent failure -> DLQ.
 */
import {
  type BodyJob,
  buildBodyPromptEn,
  buildBodyPromptJa,
  cleanBodyText,
} from "../../worker/src/body-generate.ts";
import {
  hasMaterialBodyGroundingConflict,
  hasSufficientBodySourceGrounding,
  type SourceGroundingInput,
} from "../../harness/pipeline/source-grounding.ts";
import {
  DEPLOYED_ARTICLE_CHAT_REVISION,
  isCurrentArticleChatJob,
  type BodyCacheEntry,
  putBodyCacheEntry,
} from "../../worker/src/body-cache.ts";
import { looksCompleteBodyText } from "../../worker/src/bodies-file.ts";
import { UNVERSIONED_JOB_FINGERPRINT } from "../../worker/src/kv-cache.ts";
import { DEPLOYED_PUBLISHER_FINGERPRINT } from "../../worker/src/publisher-contract.ts";
import {
  ANTHROPIC_API_KEY_MISSING,
  callAnthropicText,
  DEFAULT_ANTHROPIC_MODEL,
  type AnthropicEffort,
  parseAnthropicEffort,
  parseModelChain,
  parsePositiveInt,
  readAnthropicKey,
  requireAnthropicKey,
} from "../../worker/src/anthropic-client.ts";
import {
  buildArticleChatPrompt,
  chatGroundingText,
  parseArticleChat,
  validateArticleChat,
  type ArticleChatTurn,
} from "../../worker/src/article-chat.ts";

interface Env {
  SUMMARY_CACHE: KVNamespace;
  /** Secret (`wrangler secret put ANTHROPIC_API_KEY`). Required. */
  ANTHROPIC_API_KEY?: string;
  /** Claude model id (default claude-opus-5-5). */
  BODY_MODEL?: string;
  /** Comma-separated ordered fallback Claude models tried when a call errors. */
  BODY_MODEL_FALLBACKS?: string;
  /** low | medium | high | xhigh | max (default high). */
  BODY_EFFORT?: string;
  BODY_TIMEOUT_MS?: string;
  /** Covers thinking + answer (default 16000). */
  BODY_MAX_TOKENS?: string;
}

const BODY_SYSTEM_PROMPT =
  "You are a professional technology editor. Return only the plain-text body the user requested — no JSON, no code fences, no preamble.";

const DEFAULT_TIMEOUT_MS = 180_000;
// max_tokens covers thinking + the visible answer on current Claude models.
const DEFAULT_MAX_TOKENS = 16_000;
const DEFAULT_EFFORT: AnthropicEffort = "high";
const ISSUE_KEY = "body.issue.v1";
const ISSUE_TTL_SECONDS = 6 * 60 * 60;
const RECENT_ISSUE_MS = 60 * 60_000;
const ERROR_REPEAT_THRESHOLD = 3;
type IssueScope = "entry" | "runtime";
// A body must be at least this many characters to count as real (guards against
// a near-empty response slipping through as "complete").
const MIN_BODY_CHARS = 120;

/**
 * One plain-text Claude call. Returns the cleaned body string and throws on an
 * API error, refusal, truncation or a near-empty answer, so the model chain /
 * Cloudflare Queues retry takes over instead of persisting a bad body.
 */
async function callBodyModel(
  config: BodyLlmConfig,
  model: string,
  prompt: string,
  timeoutMs: number,
): Promise<string> {
  const result = await callAnthropicText(
    {
      model,
      systemPrompt: BODY_SYSTEM_PROMPT,
      userPrompt: prompt,
      maxTokens: config.maxTokens,
      effort: config.effort,
    },
    { anthropicKey: config.anthropicKey, timeoutMs },
  );
  const content = cleanBodyText(result.text);
  if (content.length < MIN_BODY_CHARS) {
    throw new Error(`empty/short body (${content.length} chars)`);
  }
  return content;
}

export function isBodyEntryComplete(
  entry: { bodyJa: string; bodyEn: string },
  source?: SourceGroundingInput,
): boolean {
  const structurallyComplete = Boolean(
    entry.bodyJa.trim().length >= MIN_BODY_CHARS &&
    entry.bodyEn.trim().length >= MIN_BODY_CHARS,
  );
  if (!structurallyComplete) return false;
  if (!source) return true;
  return (
    hasSufficientBodySourceGrounding(source) &&
    !hasMaterialBodyGroundingConflict(source, entry)
  );
}

export function buildBodyCacheEntry(
  job: BodyJob,
  bodyJa: string,
  bodyEn: string,
  model: string,
  cachedAt = new Date().toISOString(),
  chat?: ArticleChatTurn[],
): BodyCacheEntry {
  const acceptedChat = isCurrentArticleChatJob(job) ? validateArticleChat(chat) : null;
  return {
    bodyJa,
    bodyEn,
    ...(acceptedChat ? { chat: acceptedChat, articleChatRevision: DEPLOYED_ARTICLE_CHAT_REVISION } : {}),
    model,
    cachedAt,
    publisherContractFingerprint:
      job.publisherContractFingerprint ?? UNVERSIONED_JOB_FINGERPRINT,
  };
}

/**
 * Best-effort article chat (worker/src/article-chat.ts). One bilingual JSON
 * call on the SAME model that produced the bodies. Every failure mode —
 * API error, malformed JSON, wrong structure, grounding conflict — returns
 * undefined instead of throwing: the chat is presentation garnish and must
 * never fail, retry, or DLQ a job whose bodies already succeeded.
 */
async function generateArticleChat(
  config: BodyLlmConfig,
  model: string,
  job: BodyJob,
  timeoutMs: number,
): Promise<ArticleChatTurn[] | undefined> {
  try {
    const raw = await callBodyModel(config, model, buildArticleChatPrompt(job.entry), timeoutMs);
    const turns = parseArticleChat(raw);
    if (!turns) {
      console.warn(`[body] chat discarded for ${job.url}: structural contract not met`);
      return undefined;
    }
    const text = chatGroundingText(turns);
    if (hasMaterialBodyGroundingConflict(job.entry, { bodyJa: text.ja, bodyEn: text.en })) {
      console.warn(`[body] chat discarded for ${job.url}: grounding conflict`);
      return undefined;
    }
    return turns;
  } catch (err) {
    console.warn(
      `[body] chat generation failed for ${job.url}: ${err instanceof Error ? err.message : String(err)}`,
    );
    return undefined;
  }
}

export interface BodyLlmConfig {
  anthropicKey: string;
  effort: AnthropicEffort;
  chain: string[];
  maxTokens: number;
}

/** Settings shared by the queue path and /health; never touches the key. */
function resolveBodySettings(env: Env): Omit<BodyLlmConfig, "anthropicKey"> {
  return {
    effort: parseAnthropicEffort(env.BODY_EFFORT, DEFAULT_EFFORT),
    chain: parseModelChain(env.BODY_MODEL || DEFAULT_ANTHROPIC_MODEL, env.BODY_MODEL_FALLBACKS),
    maxTokens: parsePositiveInt(env.BODY_MAX_TOKENS, DEFAULT_MAX_TOKENS, "BODY_MAX_TOKENS"),
  };
}

/**
 * Claude API settings for one job. Throws the explicit setup error when the
 * ANTHROPIC_API_KEY secret is missing, and on any invalid model / effort /
 * token setting.
 */
export function resolveBodyLlmConfig(env: Env): BodyLlmConfig {
  return { anthropicKey: requireAnthropicKey(env), ...resolveBodySettings(env) };
}

async function processJob(env: Env, job: BodyJob): Promise<void> {
  const config = resolveBodyLlmConfig(env);
  const { chain } = config;
  const timeoutMs = Number(env.BODY_TIMEOUT_MS ?? DEFAULT_TIMEOUT_MS);
  if (!hasSufficientBodySourceGrounding(job.entry)) {
    throw new Error(`insufficient source grounding for ${job.url}`);
  }
  const canGenerateChat = isCurrentArticleChatJob(job);
  if (!canGenerateChat) {
    console.warn("[body] chat generation skipped: job fingerprint differs from compiled consumer");
  }

  // Two separate single-language calls (LL-115). Sequential so API pressure
  // stays moderate (max_concurrency also caps parallel invocations). Each call is I/O-bound (no Worker CPU cost, LL-115).
  //
  // Both languages come from the SAME model: mixing models inside one bilingual
  // entry would give the two halves different styles and failure modes. A
  // model-level failure on either call advances the whole pair to the next
  // model in the chain; if every model fails, the last error propagates so
  // Cloudflare Queues retry/DLQ semantics stay exactly as before.
  let entry: BodyCacheEntry | undefined;
  let lastError: unknown;
  for (const [attemptIndex, attemptModel] of chain.entries()) {
    if (attemptIndex > 0) {
      console.warn(
        `[body] fallback ${attemptIndex + 1}/${chain.length} (${attemptModel}) for ${job.url}: previous model failed`,
      );
    }
    try {
      const bodyJa = await callBodyModel(config, attemptModel, buildBodyPromptJa(job.entry), timeoutMs);
      const bodyEn = await callBodyModel(config, attemptModel, buildBodyPromptEn(job.entry), timeoutMs);
      // A max_tokens-truncated body must never be persisted: the 2026-08-29
      // audit found 16.9% of the indexed corpus cut mid-word by the old opus
      // config. Throwing advances the model chain / Queue retry instead.
      if (!looksCompleteBodyText(bodyJa, "ja")) throw new Error("truncated bodyJa (no terminal punctuation)");
      if (!looksCompleteBodyText(bodyEn, "en")) throw new Error("truncated bodyEn (no terminal punctuation)");
      const chat = canGenerateChat
        ? await generateArticleChat(config, attemptModel, job, timeoutMs)
        : undefined;
      entry = buildBodyCacheEntry(job, bodyJa, bodyEn, attemptModel, new Date().toISOString(), chat);
      break;
    } catch (err) {
      lastError = err;
    }
  }
  if (!entry) throw lastError ?? new Error(`all body models failed for ${job.url}`);
  if (!isBodyEntryComplete(entry, job.entry)) {
    throw new Error(`incomplete or ungrounded body for ${job.url}`);
  }
  await putBodyCacheEntry(env.SUMMARY_CACHE, job.url, entry);
  await clearIssue(env, job.url);
}

function issueSummary(err: unknown): string {
  const text = err instanceof Error ? `${err.name}: ${err.message}` : String(err);
  return text.slice(0, 500);
}

export function classifyBodyIssueScope(err: unknown): IssueScope {
  const summary = issueSummary(err).toLowerCase();
  return summary.includes("incomplete or ungrounded body for ")
    || summary.includes("insufficient source grounding for ")
    || /empty\/short body \(\d+ chars\)/.test(summary)
    ? "entry"
    : "runtime";
}

function issueScopeFromRecord(issue: Record<string, unknown> | null): IssueScope {
  if (issue?.scope === "entry" || issue?.scope === "runtime") return issue.scope;
  return classifyBodyIssueScope(issue?.error ?? "");
}

async function writeIssue(
  env: Env,
  status: "retry" | "deferred",
  job: BodyJob,
  err: unknown,
): Promise<void> {
  try {
    const existing = await env.SUMMARY_CACHE.get<Record<string, unknown>>(ISSUE_KEY, "json");
    const scope = classifyBodyIssueScope(err);
    const sameIssue =
      existing?.status === status
      && existing?.url === job.url
      && issueScopeFromRecord(existing) === scope;
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
    console.warn("[body] issue heartbeat write failed:", issueErr);
  }
}

async function clearIssue(env: Env, url: string): Promise<void> {
  try {
    const existing = await env.SUMMARY_CACHE.get<Record<string, unknown>>(ISSUE_KEY, "json");
    if (existing?.url === url) {
      await env.SUMMARY_CACHE.delete(ISSUE_KEY);
    }
  } catch (issueErr) {
    console.warn("[body] issue heartbeat clear failed:", issueErr);
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
        recentIssue
        && issueStatus === "retry"
        && issueScope === "runtime"
        && repeatCount >= ERROR_REPEAT_THRESHOLD
          ? "error"
          : recentIssue
          ? "warn"
          : "ok";
      const anthropicSecretConfigured = readAnthropicKey(env) !== null;
      let settings: ReturnType<typeof resolveBodySettings> | null = null;
      let configError: string | null = null;
      try {
        settings = resolveBodySettings(env);
      } catch (err) {
        configError = err instanceof Error ? err.message : String(err);
      }
      const ok =
        cacheBinding && anthropicSecretConfigured && settings !== null && issueSeverity !== "error";
      return Response.json(
        {
          ok,
          role: "body-queue-consumer",
          provider: "anthropic",
          model: settings?.chain[0] ?? null,
          modelFallbacks: settings?.chain.slice(1) ?? [],
          publisherContractFingerprint: DEPLOYED_PUBLISHER_FINGERPRINT,
          articleChatRevision: DEPLOYED_ARTICLE_CHAT_REVISION,
          effort: settings?.effort ?? null,
          timeoutMs: Number(env.BODY_TIMEOUT_MS ?? DEFAULT_TIMEOUT_MS),
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
          headers: { "access-control-allow-origin": "*", "cache-control": "no-store" },
        },
      );
    }
    return new Response("tech-dashboard body worker. Queue consumer is active.", {
      status: 200,
      headers: { "content-type": "text/plain;charset=UTF-8" },
    });
  },

  async queue(batch: MessageBatch<BodyJob>, env: Env): Promise<void> {
    for (const msg of batch.messages) {
      try {
        await processJob(env, msg.body);
        msg.ack();
      } catch (err) {
        const summary = err instanceof Error ? `${err.name}: ${err.message}` : String(err);
        // LL-043: KV daily write cap -> ack and let the next cron re-enqueue.
        if (summary.includes("KV put() limit exceeded")) {
          console.warn(`[body] ack ${msg.body.url}: daily KV write cap reached, will retry tomorrow`);
          await writeIssue(env, "deferred", msg.body, summary);
          msg.ack();
        } else {
          console.warn(`[body] retry ${msg.body.url}: ${summary}`);
          await writeIssue(env, "retry", msg.body, summary);
          msg.retry();
        }
      }
    }
  },
} satisfies ExportedHandler<Env, BodyJob>;
