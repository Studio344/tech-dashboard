/**
 * tests/worker-anthropic-client.test.ts
 *
 * Pins the Anthropic Claude API backend (worker/src/anthropic-client.ts):
 * provider selection by the ANTHROPIC_API_KEY secret, the request shape the
 * current models require (no temperature, effort in output_config, server-side
 * refusal fallbacks), and the stop_reason guards that keep refusals and
 * truncated answers out of KV. Also exercises the body Queue consumer end to
 * end on the Anthropic path with a stubbed fetch.
 */
import { describe, expect, it, vi } from "vitest";
import type Anthropic from "@anthropic-ai/sdk";
import {
  ANTHROPIC_FALLBACK_BETA,
  DEFAULT_ANTHROPIC_MODEL,
  buildAnthropicRequest,
  callAnthropicText,
  extractAnthropicResult,
  parseAnthropicEffort,
  resolveLlmProvider,
} from "../worker/src/anthropic-client.ts";
import bodyWorker, { resolveBodyLlmConfig } from "../worker-body/src/index.ts";
import summarizerWorker, { resolveSummaryLlmConfig } from "../worker-summarizer/src/index.ts";
import { DEPLOYED_PUBLISHER_FINGERPRINT } from "../worker/src/publisher-contract.ts";
import type { BodyCacheEntry } from "../worker/src/body-cache.ts";

function message(
  overrides: Partial<Anthropic.Beta.Messages.BetaMessage> = {},
): Anthropic.Beta.Messages.BetaMessage {
  return {
    id: "msg_test",
    type: "message",
    role: "assistant",
    model: DEFAULT_ANTHROPIC_MODEL,
    content: [
      { type: "thinking", thinking: "", signature: "sig" },
      { type: "text", text: "Hello ", citations: null },
      { type: "text", text: "world.", citations: null },
    ],
    stop_reason: "end_turn",
    stop_sequence: null,
    stop_details: null,
    usage: { input_tokens: 10, output_tokens: 5 },
    ...overrides,
  } as unknown as Anthropic.Beta.Messages.BetaMessage;
}

describe("resolveLlmProvider", () => {
  it("selects Anthropic only when the API key is present and non-blank", () => {
    expect(resolveLlmProvider({ ANTHROPIC_API_KEY: "sk-ant-test" })).toBe("anthropic");
    expect(resolveLlmProvider({ ANTHROPIC_API_KEY: "   " })).toBe("copilot");
    expect(resolveLlmProvider({ ANTHROPIC_API_KEY: "" })).toBe("copilot");
    expect(resolveLlmProvider({})).toBe("copilot");
  });
});

describe("parseAnthropicEffort", () => {
  it("uses the fallback when unset and rejects unknown values loudly", () => {
    expect(parseAnthropicEffort(undefined, "medium")).toBe("medium");
    expect(parseAnthropicEffort(" ", "high")).toBe("high");
    expect(parseAnthropicEffort("xhigh", "medium")).toBe("xhigh");
    expect(() => parseAnthropicEffort("maximum", "medium")).toThrow(/Unsupported Anthropic effort/);
  });
});

describe("buildAnthropicRequest", () => {
  it("sends effort, refusal fallbacks and no sampling parameters", () => {
    const body = buildAnthropicRequest({
      model: DEFAULT_ANTHROPIC_MODEL,
      systemPrompt: "sys",
      userPrompt: "user",
      maxTokens: 16000,
      effort: "medium",
    });
    expect(body).toEqual({
      model: "claude-opus-5-5",
      max_tokens: 16000,
      system: "sys",
      messages: [{ role: "user", content: "user" }],
      output_config: { effort: "medium" },
      betas: [ANTHROPIC_FALLBACK_BETA],
      fallbacks: "default",
    });
    expect(body).not.toHaveProperty("temperature");
    expect(body).not.toHaveProperty("thinking");
  });
});

describe("extractAnthropicResult", () => {
  it("joins visible text blocks and reports the answering model", () => {
    expect(extractAnthropicResult(message({ model: "claude-opus-4-8" }))).toEqual({
      text: "Hello world.",
      model: "claude-opus-4-8",
    });
  });

  it("throws on a refusal and on a max_tokens truncation", () => {
    expect(() =>
      extractAnthropicResult(
        message({
          stop_reason: "refusal",
          stop_details: { type: "refusal", category: "cyber", explanation: null },
        } as unknown as Partial<Anthropic.Beta.Messages.BetaMessage>),
      ),
    ).toThrow(/anthropic refusal \(category=cyber\)/);
    expect(() => extractAnthropicResult(message({ stop_reason: "max_tokens" }))).toThrow(
      /truncated at max_tokens/,
    );
  });
});

describe("callAnthropicText", () => {
  it("passes the built request and per-call timeout to the client", async () => {
    const create = vi.fn(async () => message());
    const client = { beta: { messages: { create } } } as unknown as Pick<Anthropic, "beta">;
    const result = await callAnthropicText(
      { model: "claude-opus-5-5", systemPrompt: "s", userPrompt: "u", maxTokens: 100, effort: "low" },
      { anthropicKey: "unused", timeoutMs: 1234, client },
    );
    expect(result.text).toBe("Hello world.");
    expect(create).toHaveBeenCalledWith(
      expect.objectContaining({ model: "claude-opus-5-5", output_config: { effort: "low" } }),
      { timeout: 1234, maxRetries: 0 },
    );
  });
});

describe("consumer LLM config", () => {
  const kv = { get: async () => null, put: async () => {} } as unknown as KVNamespace;

  it("keeps the Copilot chain unchanged when no Anthropic key is configured", () => {
    const summary = resolveSummaryLlmConfig({
      SUMMARY_CACHE: kv,
      COPILOT_PAT: "pat",
      SUMMARIZE_MODEL: "gpt-5.6-sol",
      SUMMARIZE_MODEL_FALLBACKS: "claude-sonnet-5",
    });
    expect(summary.provider).toBe("copilot");
    expect(summary.chain).toEqual(["gpt-5.6-sol", "claude-sonnet-5"]);
    expect(summary.maxTokens).toBe(1600);

    const body = resolveBodyLlmConfig({ SUMMARY_CACHE: kv, COPILOT_PAT: "pat", BODY_MAX_TOKENS: "4000" });
    expect(body.provider).toBe("copilot");
    expect(body.maxTokens).toBe(4000);
  });

  it("switches both consumers to Claude Opus 5.5 with their own budgets when the key is set", () => {
    const summary = resolveSummaryLlmConfig({
      SUMMARY_CACHE: kv,
      COPILOT_PAT: "",
      SUMMARIZE_MODEL: "gpt-5.6-sol",
      ANTHROPIC_API_KEY: "sk-ant-test",
    });
    expect(summary).toMatchObject({
      provider: "anthropic",
      chain: ["claude-opus-5-5"],
      maxTokens: 16000,
      backend: { provider: "anthropic", effort: "medium" },
    });

    const body = resolveBodyLlmConfig({
      SUMMARY_CACHE: kv,
      COPILOT_PAT: "",
      ANTHROPIC_API_KEY: "sk-ant-test",
      ANTHROPIC_BODY_EFFORT: "xhigh",
    });
    expect(body).toMatchObject({
      provider: "anthropic",
      chain: ["claude-opus-5-5"],
      maxTokens: 16000,
      backend: { provider: "anthropic", effort: "xhigh" },
    });
  });

  it("reports the Anthropic provider on /health without requiring COPILOT_PAT", async () => {
    const response = await summarizerWorker.fetch!(
      new Request("https://tech-dashboard-summarizer.example/health"),
      { SUMMARY_CACHE: kv, COPILOT_PAT: "", ANTHROPIC_API_KEY: "sk-ant-test" },
      {} as ExecutionContext,
    );
    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toMatchObject({
      ok: true,
      provider: "anthropic",
      model: "claude-opus-5-5",
      effort: "medium",
      anthropicSecretConfigured: true,
      copilotSecretConfigured: false,
    });
  });
});

describe("body Queue consumer on the Anthropic path", () => {
  const entry = {
    id: "anthropic-body-test",
    title: "GLM-5.2 on Cloudflare Workers AI",
    titleJa: "Cloudflare Workers AI で GLM-5.2 を試す",
    titleEn: "Trying GLM-5.2 on Cloudflare Workers AI",
    category: "local-llm",
    source: "qiita-llm",
    sourceType: "community",
    url: "https://example.com/glm-5-2",
    summaryJa: "Cloudflare Workers AI で GLM-5.2 を無料で試す手順を解説する。",
    summaryEn: "A walkthrough of running GLM-5.2 for free on Cloudflare Workers AI.",
    contentSnippet:
      "The source gives a step-by-step walkthrough for running GLM-5.2 on Cloudflare Workers AI, including setup and cost constraints.",
    tags: ["llm", "cloudflare", "glm"],
    publishedAt: "2026-06-27T10:00:00.000Z",
  };

  it("generates both bodies through the Messages API and never calls Copilot", async () => {
    const ack = vi.fn();
    const retry = vi.fn();
    const put = vi.fn(async (_key: string, _value: string) => {});
    const requests: Array<{ url: string; headers: Headers; body: Record<string, unknown> }> = [];
    const texts = [
      "GLM-5.2をCloudflare Workers AIで試す手順を記事に沿って説明する。".repeat(8),
      "The article walks through running GLM-5.2 on Cloudflare Workers AI. ".repeat(8),
    ];
    const fetchMock = vi.fn(async (input: string | URL | Request, init?: RequestInit) => {
      const request = input instanceof Request ? input : new Request(input, init);
      const body = (await request.json()) as Record<string, unknown>;
      requests.push({ url: request.url, headers: request.headers, body });
      const text = texts[requests.length - 1];
      if (!text) throw new Error("unexpected model request");
      return Response.json(message({ content: [{ type: "text", text, citations: null }] } as never));
    });
    const warning = vi.spyOn(console, "warn").mockImplementation(() => {});
    vi.stubGlobal("fetch", fetchMock);
    try {
      await bodyWorker.queue!(
        {
          messages: [{
            body: {
              url: entry.url,
              // An old fingerprint skips article chat, so exactly two calls run.
              publisherContractFingerprint: `sha256:${"e".repeat(64)}`,
              entry,
            },
            ack,
            retry,
          }],
        } as unknown as MessageBatch,
        {
          SUMMARY_CACHE: { get: async () => null, put } as unknown as KVNamespace,
          COPILOT_PAT: "",
          ANTHROPIC_API_KEY: "sk-ant-test",
        },
        {} as ExecutionContext,
      );
    } finally {
      vi.unstubAllGlobals();
      warning.mockRestore();
    }

    expect(retry).not.toHaveBeenCalled();
    expect(ack).toHaveBeenCalledOnce();
    expect(requests).toHaveLength(2);
    for (const request of requests) {
      expect(new URL(request.url).pathname).toBe("/v1/messages");
      expect(request.url).not.toContain("githubcopilot");
      // SDK 1.x sends the API key as a bearer credential.
      expect(request.headers.get("authorization")).toBe("Bearer sk-ant-test");
      expect(request.headers.get("anthropic-beta")).toContain(ANTHROPIC_FALLBACK_BETA);
      expect(request.body).toMatchObject({
        model: "claude-opus-5-5",
        max_tokens: 16000,
        output_config: { effort: "high" },
        fallbacks: "default",
      });
      expect(request.body).not.toHaveProperty("temperature");
    }
    const writes = put.mock.calls
      .filter(([key]) => key !== "body.issue.v1")
      .map(([, value]) => JSON.parse(value) as BodyCacheEntry);
    expect(writes).toHaveLength(1);
    expect(writes[0]?.model).toBe("claude-opus-5-5");
    expect(writes[0]?.bodyJa).toContain("GLM-5.2");
    expect(writes[0]?.bodyEn).toContain("GLM-5.2");
    expect(writes[0]?.publisherContractFingerprint).not.toBe(DEPLOYED_PUBLISHER_FINGERPRINT);
  });
});
