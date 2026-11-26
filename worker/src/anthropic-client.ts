/**
 * Anthropic Claude API client: the only LLM backend of the summary and body
 * Queue consumers and the local harness summarizer (site-owner request
 * 2026-11-26: GitHub Copilot is no longer used).
 *
 * The key comes from the `ANTHROPIC_API_KEY` secret / env var. Without it a
 * consumer fails the job with an explicit error and reports `ok: false` on
 * `/health`; it never falls back to another provider.
 *
 * Request contract for the current models (claude-opus-5-5 by default):
 *   - Thinking is always on and cannot be disabled; depth is set with
 *     `output_config.effort`. `temperature` is not accepted, so it is never sent.
 *   - `max_tokens` covers thinking AND the visible answer. The Copilot-era
 *     budgets (1600 / 4000) were sized for chat models without that overhead,
 *     so this backend has its own, larger defaults (see the consumers).
 *   - Server-side refusal fallbacks (`fallbacks: "default"`) are enabled so a
 *     safety-classifier decline is re-run on a fallback model inside the same
 *     call. The model that actually answered is returned to the caller and is
 *     what gets recorded in the cache entry.
 *   - `stop_reason: "refusal"` (whole chain declined) and `"max_tokens"`
 *     (truncated answer) throw, so the callers' existing model-chain and
 *     Cloudflare Queues retry/DLQ contracts handle them instead of a truncated
 *     body or partial JSON being written to KV.
 *
 * SDK retries are disabled (maxRetries: 0): the consumers already retry through
 * their model chain and Queues' `retry_delay` backoff, and SDK retries would
 * multiply each attempt's wall time past the configured timeout.
 */
import Anthropic from "@anthropic-ai/sdk";

export const DEFAULT_ANTHROPIC_MODEL = "claude-opus-5-5";
export const ANTHROPIC_FALLBACK_BETA = "server-side-fallback-2026-07-01";

export const ANTHROPIC_API_KEY_MISSING =
  "ANTHROPIC_API_KEY is not configured. Register it with `npx wrangler secret put ANTHROPIC_API_KEY` in this Worker's directory (or set it in .env.local for local runs).";

export type AnthropicEffort = "low" | "medium" | "high" | "xhigh" | "max";
const EFFORTS: readonly AnthropicEffort[] = ["low", "medium", "high", "xhigh", "max"];

/** The configured key (trimmed), or null when it is missing or blank. */
export function readAnthropicKey(env: { ANTHROPIC_API_KEY?: string }): string | null {
  return env.ANTHROPIC_API_KEY?.trim() || null;
}

/** The configured key; throws the explicit setup error when it is missing. */
export function requireAnthropicKey(env: { ANTHROPIC_API_KEY?: string }): string {
  const key = readAnthropicKey(env);
  if (!key) throw new Error(ANTHROPIC_API_KEY_MISSING);
  return key;
}

/**
 * Ordered, de-duplicated model chain from a primary plus a comma-separated
 * fallback list. Blanks are dropped and the primary always leads. Every model
 * must be a Claude model id: a leftover non-Claude value (for example a model
 * name from the retired Copilot configuration) throws instead of sending a
 * request that can only fail.
 */
export function parseModelChain(primary: string, fallbacks: string | undefined): string[] {
  const chain: string[] = [];
  for (const candidate of [primary, ...(fallbacks ?? "").split(",")]) {
    const model = candidate.trim();
    if (model && !chain.includes(model)) chain.push(model);
  }
  const invalid = chain.filter((model) => !model.startsWith("claude-"));
  if (invalid.length > 0) {
    throw new Error(
      `Unsupported model "${invalid.join(", ")}": only Anthropic Claude model ids (claude-*) are accepted.`,
    );
  }
  return chain;
}

/** Positive integer setting from a Worker var / env var, or the fallback. */
export function parsePositiveInt(value: string | undefined, fallback: number, name: string): number {
  const trimmed = value?.trim();
  if (!trimmed) return fallback;
  const parsed = Number(trimmed);
  if (!Number.isInteger(parsed) || parsed <= 0) {
    throw new Error(`${name} must be a positive integer, got "${trimmed}".`);
  }
  return parsed;
}

/**
 * Validates an effort setting from a Worker var / env var. An unknown value
 * throws instead of silently falling back, so a typo in wrangler.toml surfaces
 * as a visible job failure rather than an unnoticed cost/quality change.
 */
export function parseAnthropicEffort(
  value: string | undefined,
  fallback: AnthropicEffort,
): AnthropicEffort {
  const trimmed = value?.trim();
  if (!trimmed) return fallback;
  if ((EFFORTS as readonly string[]).includes(trimmed)) return trimmed as AnthropicEffort;
  throw new Error(
    `Unsupported Anthropic effort "${trimmed}". Use one of: ${EFFORTS.join(", ")}.`,
  );
}

export interface AnthropicTextRequest {
  model: string;
  systemPrompt: string;
  userPrompt: string;
  maxTokens: number;
  effort: AnthropicEffort;
}

/** Request body for `client.beta.messages.create`. Pure and test-covered. */
export function buildAnthropicRequest(
  request: AnthropicTextRequest,
): Anthropic.Beta.Messages.MessageCreateParamsNonStreaming {
  return {
    model: request.model,
    max_tokens: request.maxTokens,
    system: request.systemPrompt,
    messages: [{ role: "user", content: request.userPrompt }],
    output_config: { effort: request.effort },
    betas: [ANTHROPIC_FALLBACK_BETA],
    fallbacks: "default",
  };
}

export interface AnthropicTextResult {
  text: string;
  /** The model that produced the answer (differs from the request after a fallback). */
  model: string;
}

/**
 * Visible text of a response. Thinking blocks and fallback markers are skipped.
 * Throws on a refusal or a truncated answer (see module comment).
 */
export function extractAnthropicResult(message: Anthropic.Beta.Messages.BetaMessage): AnthropicTextResult {
  if (message.stop_reason === "refusal") {
    const category = message.stop_details?.category ?? "unknown";
    throw new Error(`anthropic refusal (category=${category}) from ${message.model}`);
  }
  if (message.stop_reason === "max_tokens") {
    throw new Error(`anthropic output truncated at max_tokens (${message.model})`);
  }
  const text = message.content
    .map((block) => (block.type === "text" ? block.text : ""))
    .join("");
  return { text, model: message.model };
}

export interface AnthropicCallOptions {
  anthropicKey: string;
  timeoutMs: number;
  /** Injected in tests; production constructs a client per call. */
  client?: Pick<Anthropic, "beta">;
}

export async function callAnthropicText(
  request: AnthropicTextRequest,
  options: AnthropicCallOptions,
): Promise<AnthropicTextResult> {
  const key = options.anthropicKey;
  const client =
    options.client ??
    // authToken: null so an ambient ANTHROPIC_AUTH_TOKEN can never replace the
    // configured key in a local harness run.
    new Anthropic({
      apiKey: key,
      authToken: null,
      timeout: options.timeoutMs,
      maxRetries: 0,
    });
  const message = await client.beta.messages.create(buildAnthropicRequest(request), {
    timeout: options.timeoutMs,
    maxRetries: 0,
  });
  return extractAnthropicResult(message);
}
