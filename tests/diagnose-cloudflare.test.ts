/**
 * tests/diagnose-cloudflare.test.ts
 *
 * Guards the read-only Cloudflare diagnosis (site-owner request 2026-11-27):
 * the workflow only runs by hand from main / develop with the protected
 * environment, the script only sends GET and read-only queries, secrets never
 * reach the output, and a failing section reports its error category instead
 * of a success-shaped default.
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
// @ts-expect-error plain ESM script without type declarations
import * as diag from "../scripts/diagnose-cloudflare.mjs";

const ROOT = join(import.meta.dirname, "..");
const workflow = readFileSync(join(ROOT, ".github/workflows/diagnose-cloudflare.yml"), "utf8");
const onSection = workflow.slice(workflow.indexOf("\non:\n"), workflow.indexOf("\npermissions:\n"));

const TOKEN = "cf-test-token-0123456789";
const ACCOUNT = "acct0123456789abcdef";
const NOW = Date.parse("2026-11-27T12:00:00Z");

type Call = { method: string; url: string; body?: unknown };

function fakeFetch(handler: (call: Call) => { status?: number; json: unknown }) {
  const calls: Call[] = [];
  const impl = async (url: string, init: { method: string; body?: string; headers: Record<string, string> }) => {
    const call = { method: init.method, url, body: init.body ? JSON.parse(init.body) : undefined };
    calls.push(call);
    const { status = 200, json } = handler(call);
    return { ok: status >= 200 && status < 300, status, json: async () => json };
  };
  return { calls, impl };
}

function healthyHandler(call: Call) {
  if (call.url.endsWith("/queues")) {
    return {
      json: {
        success: true,
        result: [
          {
            queue_id: "q-summary",
            queue_name: "tech-dashboard-summary",
            consumers: [{ script: "tech-dashboard-summarizer", settings: { batch_size: 1, max_retries: 2 } }],
            producers: [{ script: "tech-dashboard-harness" }],
          },
          { queue_id: "q-dlq", queue_name: "tech-dashboard-summary-dlq", consumers: [], producers: [] },
        ],
      },
    };
  }
  if (call.url.endsWith("/q-summary/metrics")) return { json: { success: true, result: { backlog_count: 0 } } };
  if (call.url.endsWith("/q-dlq/metrics")) return { json: { success: true, result: { backlog_count: 35 } } };
  if (call.url.endsWith("/graphql")) {
    const query = String((call.body as { query: string }).query);
    if (query.includes("KvAccountWrites")) {
      return {
        json: {
          data: { viewer: { accounts: [{ kvOperationsAdaptiveGroups: [
            { sum: { requests: 1000 }, dimensions: { date: "2026-11-27", actionType: "write", namespaceId: "ns" } },
            { sum: { requests: 40 }, dimensions: { date: "2026-11-26", actionType: "write", namespaceId: "ns" } },
            { sum: { requests: 9 }, dimensions: { date: "2026-11-27", actionType: "read", namespaceId: "ns" } },
          ] }] } },
        },
      };
    }
    if (query.includes("WorkerInvocations")) {
      return {
        json: {
          data: { viewer: { accounts: [{ workersInvocationsAdaptive: [
            { sum: { requests: 3, errors: 3, subrequests: 0 }, dimensions: { status: "scriptThrewException" } },
          ] }] } },
        },
      };
    }
    return { json: { data: { viewer: { accounts: [{}] } } } };
  }
  return { json: { success: true, result: { events: { events: [] } } } };
}

describe("Diagnose Cloudflare workflow", () => {
  it("only runs when dispatched by hand from main or develop", () => {
    expect(onSection).toContain("workflow_dispatch:");
    expect(onSection).not.toMatch(/^\s{2}(push|pull_request|pull_request_target|schedule|workflow_run|repository_dispatch):/m);
    expect(workflow).toContain("if: github.repository == 'Studio344/tech-dashboard'");
    expect(workflow).toContain("refs/heads/main | refs/heads/develop) ;;");
  });

  it("keeps the token in the protected environment and env only", () => {
    expect(workflow).toMatch(/^permissions:\n {2}contents: read$/m);
    expect(workflow).toMatch(/environment:\n\s+name: cloudflare-production/);
    expect(workflow).toContain("persist-credentials: false");
    const secrets = new Set([...workflow.matchAll(/\$\{\{\s*secrets\.([A-Z_]+)\s*\}\}/g)].map((m) => m[1]));
    expect(secrets).toEqual(new Set(["CLOUDFLARE_API_TOKEN", "CLOUDFLARE_ACCOUNT_ID"]));
    for (const line of workflow.split("\n").filter((l) => l.includes("${{ secrets."))) {
      expect(line, line).toMatch(/^\s+[A-Z_]+: \$\{\{ secrets\.[A-Z_]+ \}\}$/);
    }
    expect(workflow).not.toMatch(/wrangler|set -x|echo "\$\{?CLOUDFLARE/);
    // The dispatch input reaches the script through env, never inline in run.
    expect(workflow).not.toMatch(/run:.*\$\{\{\s*inputs\./);
  });
});

describe("diagnose-cloudflare script", () => {
  it("sends only GET requests and read-only POST queries", async () => {
    const { calls, impl } = fakeFetch(healthyHandler);
    const client = diag.createClient({ token: TOKEN, accountId: ACCOUNT, fetchImpl: impl });
    await diag.diagnose({ client, hours: 24, nowMs: NOW });
    for (const call of calls) {
      if (call.method === "GET") continue;
      expect(call.method).toBe("POST");
      expect(call.url.endsWith("/graphql") || call.url.endsWith("/workers/observability/telemetry/query"), call.url).toBe(true);
      if (call.url.endsWith("/graphql")) {
        expect(String((call.body as { query: string }).query)).not.toMatch(/\bmutation\b/);
      }
    }
  });

  it("names the failure modes from the evidence", async () => {
    const { impl } = fakeFetch(healthyHandler);
    const client = diag.createClient({ token: TOKEN, accountId: ACCOUNT, fetchImpl: impl });
    const result = await diag.diagnose({ client, hours: 24, nowMs: NOW });
    const verdicts = result.verdicts.join("\n");
    expect(verdicts).toContain("tech-dashboard-summarizer");
    expect(verdicts).toContain("Dead-letter queue: 35 message(s)");
    expect(verdicts).toContain("reached the free daily limit (1000) on 2026-11-27");
    expect(verdicts).toContain("scriptThrewException=3");
  });

  it("reports a missing consumer and zero invocations instead of guessing", () => {
    const verdicts = diag.buildVerdicts({ consumer: null, summarizerStatuses: [] }).join("\n");
    expect(verdicts).toContain("NONE attached");
    expect(verdicts).toContain("The consumer never ran.");
    expect(diag.buildVerdicts({}).join("\n")).toContain("Queue consumer: unknown");
  });

  it("keeps going section by section and categorizes a permission error", async () => {
    const { impl } = fakeFetch((call) => {
      if (call.url.endsWith("/graphql")) return { json: { errors: [{ message: "not authorized for that account" }] } };
      return healthyHandler(call);
    });
    const client = diag.createClient({ token: TOKEN, accountId: ACCOUNT, fetchImpl: impl });
    const result = await diag.diagnose({ client, hours: 24, nowMs: NOW });
    const failed = result.report.filter((s: { ok: boolean }) => !s.ok);
    expect(failed.length).toBeGreaterThan(0);
    expect(failed[0].lines[0]).toMatch(/^ERROR \(permission\): /);
    expect(result.report.find((s: { title: string }) => s.title === "Queues and consumers").ok).toBe(true);
  });

  it("never prints the token or account id, even when Cloudflare echoes them", async () => {
    const { impl } = fakeFetch((call) => ({
      status: 403,
      json: { success: false, errors: [{ code: 10000, message: `token ${TOKEN} has no access to ${ACCOUNT} (Bearer ${TOKEN})` }] },
    }));
    const client = diag.createClient({ token: TOKEN, accountId: ACCOUNT, fetchImpl: impl });
    const text = diag.formatReport(await diag.diagnose({ client, hours: 24, nowMs: NOW }));
    expect(text).not.toContain(TOKEN);
    expect(text).not.toContain(ACCOUNT);
    expect(text).toContain("ERROR (permission): HTTP 403: 10000: token *** has no access to ***");
  });

  it("rejects a missing token, a bad window and unknown arguments", () => {
    expect(() => diag.createClient({ token: "", accountId: ACCOUNT })).toThrow(/CLOUDFLARE_API_TOKEN/);
    expect(() => diag.createClient({ token: TOKEN, accountId: "" })).toThrow(/CLOUDFLARE_ACCOUNT_ID/);
    expect(diag.parseArgs(["--hours", "6"])).toEqual({ hours: 6 });
    expect(() => diag.parseArgs(["--hours", "0"])).toThrow(/1 to 72/);
    expect(() => diag.parseArgs(["--hours", "abc"])).toThrow(/1 to 72/);
    expect(() => diag.parseArgs(["--write"])).toThrow(/unknown argument/);
  });
});
