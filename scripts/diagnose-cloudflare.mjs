#!/usr/bin/env node

// Read-only Cloudflare diagnosis for the summary pipeline (site-owner request
// 2026-11-27). The Publisher sends summary jobs to the `tech-dashboard-summary`
// Queue, `tech-dashboard-summarizer` writes the result to the SUMMARY_CACHE KV
// namespace, and the next Publisher run merges it. When summaries stop landing
// while the summarizer `/health` stays clean, the cause is either that the
// consumer never processes the jobs, or that every KV write fails (which also
// stops the summarizer's own issue record). Neither is visible from GitHub, so
// this script reads the account's own numbers:
//
//   1. Queue list: queue ids, the consumer attached to each queue.
//   2. Realtime backlog of the summary queue and its dead-letter queue.
//   3. GraphQL: queue backlog over time, message writes / reads / deletes by
//      outcome (success, dlq, fail), consumer concurrency.
//   4. GraphQL: KV operations per day (reads, writes) against the free-tier
//      daily write limit.
//   5. GraphQL: Worker invocations by status for the summarizer and the bridge.
//   6. Workers Observability: the summarizer's recent log events (errors and
//      `[summarizer]` warnings), when the token may read them.
//
// Only GET requests and read-only queries (GraphQL, telemetry query) are sent.
// Nothing is written, purged, retried or redeployed. Each section reports its
// own result, so a missing token permission shows up as that section's error
// instead of stopping the whole run. The token and account id never appear in
// output: requests carry them in headers / paths only and every error message
// is passed through `redact`.
//
// Usage (GitHub Actions sets the env from the protected environment):
//   CLOUDFLARE_API_TOKEN=... CLOUDFLARE_ACCOUNT_ID=... node scripts/diagnose-cloudflare.mjs [--hours 24]

import { appendFileSync } from "node:fs";
import { pathToFileURL } from "node:url";

export const API_BASE = "https://api.cloudflare.com/client/v4";
export const SUMMARY_QUEUE = "tech-dashboard-summary";
export const SUMMARY_DLQ = "tech-dashboard-summary-dlq";
// Same namespace id as worker/wrangler.toml and worker-summarizer/wrangler.toml.
export const SUMMARY_KV_NAMESPACE = "6d67debb991742efadfec473a121f5fc";
export const SCRIPTS = ["tech-dashboard-summarizer", "tech-dashboard-harness"];
export const KV_FREE_DAILY_WRITE_LIMIT = 1000;
const MAX_TEXT = 300;
const MAX_EVENTS = 30;

export class DiagnosisError extends Error {
  constructor(category, message) {
    super(message);
    this.name = "DiagnosisError";
    this.category = category;
  }
}

/** Remove the secrets and anything token-shaped from text before printing it. */
export function redact(text, secrets = []) {
  let out = String(text ?? "");
  for (const secret of secrets) {
    if (secret && secret.length >= 4) out = out.split(secret).join("***");
  }
  out = out.replace(/Bearer\s+[A-Za-z0-9._~+/=-]+/gi, "Bearer ***");
  out = out.replace(/sk-ant-[A-Za-z0-9_-]+/g, "sk-ant-***");
  return out.length > MAX_TEXT ? `${out.slice(0, MAX_TEXT)}…` : out;
}

export function parseArgs(argv) {
  let hours = 24;
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === "--hours") {
      hours = Number(argv[++i]);
    } else {
      throw new DiagnosisError("usage", `unknown argument ${argv[i]}`);
    }
  }
  if (!Number.isInteger(hours) || hours < 1 || hours > 72) {
    throw new DiagnosisError("usage", "--hours must be an integer from 1 to 72");
  }
  return { hours };
}

function classifyHttp(status) {
  if (status === 401 || status === 403) return "permission";
  if (status === 404) return "not-found";
  if (status === 429) return "rate-limited";
  if (status >= 500) return "cloudflare-unavailable";
  return "request-rejected";
}

function apiErrorText(body) {
  const errors = Array.isArray(body?.errors) ? body.errors : [];
  return errors
    .map((e) => (e?.code ? `${e.code}: ${e.message ?? ""}` : String(e?.message ?? e)))
    .join("; ");
}

export function createClient({ token, accountId, fetchImpl = fetch }) {
  if (!token) throw new DiagnosisError("config", "CLOUDFLARE_API_TOKEN is not set");
  if (!accountId) throw new DiagnosisError("config", "CLOUDFLARE_ACCOUNT_ID is not set");
  const secrets = [token, accountId];
  const headers = {
    authorization: `Bearer ${token}`,
    "content-type": "application/json",
  };

  async function call(method, path, body) {
    let response;
    try {
      response = await fetchImpl(`${API_BASE}${path}`, {
        method,
        headers,
        body: body === undefined ? undefined : JSON.stringify(body),
      });
    } catch (err) {
      throw new DiagnosisError("network", redact(err instanceof Error ? err.message : err, secrets));
    }
    let parsed;
    try {
      parsed = await response.json();
    } catch {
      throw new DiagnosisError(classifyHttp(response.status), `HTTP ${response.status}, body is not JSON`);
    }
    if (!response.ok || parsed?.success === false) {
      const detail = apiErrorText(parsed) || `HTTP ${response.status}`;
      throw new DiagnosisError(
        response.ok ? "request-rejected" : classifyHttp(response.status),
        redact(`HTTP ${response.status}: ${detail}`, secrets),
      );
    }
    return parsed;
  }

  return {
    secrets,
    accountPath: `/accounts/${encodeURIComponent(accountId)}`,
    accountId,
    get: (path) => call("GET", path),
    post: (path, body) => call("POST", path, body),
    async graphql(query, variables) {
      const parsed = await call("POST", "/graphql", { query, variables });
      if (Array.isArray(parsed?.errors) && parsed.errors.length > 0) {
        const text = parsed.errors.map((e) => e?.message ?? String(e)).join("; ");
        const category = /authoriz|permission|access/i.test(text) ? "permission" : "query-rejected";
        throw new DiagnosisError(category, redact(text, secrets));
      }
      return parsed?.data?.viewer?.accounts?.[0] ?? {};
    },
  };
}

const QUEUE_BACKLOG_QUERY = `query QueueBacklog($accountTag: string!, $queueId: string!, $start: Time!, $end: Time!) {
  viewer { accounts(filter: { accountTag: $accountTag }) {
    queueBacklogAdaptiveGroups(limit: 10000, filter: { queueId: $queueId, datetime_geq: $start, datetime_leq: $end }, orderBy: [datetimeHour_ASC]) {
      avg { messages bytes }
      dimensions { datetimeHour }
    }
  } }
}`;

const QUEUE_OPERATIONS_QUERY = `query QueueOperations($accountTag: string!, $queueId: string!, $start: Time!, $end: Time!) {
  viewer { accounts(filter: { accountTag: $accountTag }) {
    queueMessageOperationsAdaptiveGroups(limit: 10000, filter: { queueId: $queueId, datetime_geq: $start, datetime_leq: $end }) {
      count
      avg { retryCount lagTime }
      dimensions { actionType consumerType outcome }
    }
  } }
}`;

const QUEUE_CONSUMER_QUERY = `query QueueConsumer($accountTag: string!, $queueId: string!, $start: Time!, $end: Time!) {
  viewer { accounts(filter: { accountTag: $accountTag }) {
    queueConsumerMetricsAdaptiveGroups(limit: 10000, filter: { queueId: $queueId, datetime_geq: $start, datetime_leq: $end }, orderBy: [datetimeHour_ASC]) {
      avg { concurrency }
      dimensions { datetimeHour }
    }
  } }
}`;

const KV_OPERATIONS_QUERY = `query KvOperations($accountTag: string!, $namespaceId: string!, $start: Date!, $end: Date!) {
  viewer { accounts(filter: { accountTag: $accountTag }) {
    kvOperationsAdaptiveGroups(limit: 10000, filter: { namespaceId: $namespaceId, date_geq: $start, date_leq: $end }, orderBy: [date_ASC]) {
      sum { requests }
      dimensions { date actionType }
    }
  } }
}`;

const KV_ACCOUNT_WRITES_QUERY = `query KvAccountWrites($accountTag: string!, $start: Date!, $end: Date!) {
  viewer { accounts(filter: { accountTag: $accountTag }) {
    kvOperationsAdaptiveGroups(limit: 10000, filter: { date_geq: $start, date_leq: $end }, orderBy: [date_ASC]) {
      sum { requests }
      dimensions { date actionType namespaceId }
    }
  } }
}`;

const WORKER_INVOCATIONS_QUERY = `query WorkerInvocations($accountTag: string!, $scriptName: string!, $start: string!, $end: string!) {
  viewer { accounts(filter: { accountTag: $accountTag }) {
    workersInvocationsAdaptive(limit: 10000, filter: { scriptName: $scriptName, datetime_geq: $start, datetime_leq: $end }) {
      sum { requests errors subrequests }
      dimensions { status }
    }
  } }
}`;

function isoDate(ms) {
  return new Date(ms).toISOString().slice(0, 10);
}

/** Sum rows by a key built from their dimensions. */
export function sumBy(rows, keyOf, valueOf) {
  const totals = new Map();
  for (const row of rows ?? []) {
    const key = keyOf(row);
    totals.set(key, (totals.get(key) ?? 0) + Number(valueOf(row) ?? 0));
  }
  return [...totals.entries()].sort(([a], [b]) => String(a).localeCompare(String(b)));
}

/**
 * Turn the gathered numbers into the few verdicts the owner needs. Every
 * verdict names the evidence; a missing section yields "unknown", never a
 * guess.
 */
export function buildVerdicts(facts) {
  const verdicts = [];
  if (facts.consumer === undefined) {
    verdicts.push("Queue consumer: unknown (queue list unavailable)");
  } else if (facts.consumer === null) {
    verdicts.push(`Queue consumer: NONE attached to ${SUMMARY_QUEUE}. Jobs are never processed.`);
  } else {
    verdicts.push(`Queue consumer: ${facts.consumer} is attached to ${SUMMARY_QUEUE}.`);
  }
  if (typeof facts.dlqBacklog === "number") {
    verdicts.push(
      facts.dlqBacklog > 0
        ? `Dead-letter queue: ${facts.dlqBacklog} message(s). Jobs reached the consumer and failed every retry.`
        : "Dead-letter queue: empty.",
    );
  }
  if (typeof facts.summaryBacklog === "number") {
    verdicts.push(`Summary queue backlog now: ${facts.summaryBacklog} message(s).`);
  }
  if (facts.kvWritesByDay) {
    const over = facts.kvWritesByDay.filter(([, n]) => n >= KV_FREE_DAILY_WRITE_LIMIT);
    verdicts.push(
      over.length > 0
        ? `KV writes reached the free daily limit (${KV_FREE_DAILY_WRITE_LIMIT}) on ${over.map(([d]) => d).join(", ")}. Writes after that fail, including the summarizer's issue record.`
        : `KV writes stayed under ${KV_FREE_DAILY_WRITE_LIMIT}/day in the window (account total).`,
    );
  }
  if (facts.summarizerStatuses) {
    const total = facts.summarizerStatuses.reduce((n, [, v]) => n + v, 0);
    verdicts.push(
      total === 0
        ? "Summarizer invocations: 0 in the window. The consumer never ran."
        : `Summarizer invocations by status: ${facts.summarizerStatuses.map(([s, v]) => `${s}=${v}`).join(", ")}.`,
    );
  }
  return verdicts;
}

async function section(report, title, fn) {
  try {
    const lines = await fn();
    report.push({ title, ok: true, lines });
  } catch (err) {
    const category = err instanceof DiagnosisError ? err.category : "unexpected";
    const message = err instanceof Error ? err.message : String(err);
    report.push({ title, ok: false, lines: [`ERROR (${category}): ${message}`] });
  }
}

export async function diagnose({ client, hours, nowMs = Date.now() }) {
  const end = new Date(nowMs).toISOString();
  const start = new Date(nowMs - hours * 3_600_000).toISOString();
  const startDate = isoDate(nowMs - hours * 3_600_000);
  const endDate = isoDate(nowMs);
  const accountTag = client.accountId;
  const facts = {};
  const report = [];
  let queues = [];

  await section(report, "Queues and consumers", async () => {
    const body = await client.get(`${client.accountPath}/queues`);
    queues = Array.isArray(body?.result) ? body.result : [];
    const lines = [];
    for (const name of [SUMMARY_QUEUE, SUMMARY_DLQ]) {
      const queue = queues.find((q) => q?.queue_name === name);
      if (!queue) {
        lines.push(`${name}: not found in this account`);
        if (name === SUMMARY_QUEUE) facts.consumer = null;
        continue;
      }
      const consumers = Array.isArray(queue.consumers) ? queue.consumers : [];
      const described = consumers.map((c) => {
        const target = c?.script ?? c?.service ?? c?.script_name ?? c?.type ?? "unknown";
        const s = c?.settings ?? {};
        return `${target} (batch_size=${s.batch_size ?? "?"}, max_retries=${s.max_retries ?? "?"}, max_concurrency=${s.max_concurrency ?? "?"}, retry_delay=${s.retry_delay ?? "?"}, dlq=${c?.dead_letter_queue ?? "?"})`;
      });
      if (name === SUMMARY_QUEUE) facts.consumer = described.length > 0 ? described.join("; ") : null;
      lines.push(`${name}: consumers=${described.length > 0 ? described.join("; ") : "none"}, producers=${Array.isArray(queue.producers) ? queue.producers.map((p) => p?.script ?? p?.type ?? "?").join(", ") || "none" : "?"}`);
    }
    return lines;
  });

  const queueId = (name) => queues.find((q) => q?.queue_name === name)?.queue_id;

  for (const name of [SUMMARY_QUEUE, SUMMARY_DLQ]) {
    await section(report, `Realtime backlog: ${name}`, async () => {
      const id = queueId(name);
      if (!id) throw new DiagnosisError("not-found", "queue id unknown (see the queue list section)");
      const body = await client.get(`${client.accountPath}/queues/${encodeURIComponent(id)}/metrics`);
      const r = body?.result ?? {};
      const backlog = Number(r.backlog_count);
      if (Number.isFinite(backlog)) {
        if (name === SUMMARY_QUEUE) facts.summaryBacklog = backlog;
        else facts.dlqBacklog = backlog;
      }
      const oldest = Number(r.oldest_message_timestamp_ms);
      return [
        `backlog_count=${r.backlog_count ?? "?"}, backlog_bytes=${r.backlog_bytes ?? "?"}, oldest_message=${Number.isFinite(oldest) && oldest > 0 ? new Date(oldest).toISOString() : "none"}`,
      ];
    });
  }

  await section(report, `Summary queue message operations (last ${hours}h)`, async () => {
    const id = queueId(SUMMARY_QUEUE);
    if (!id) throw new DiagnosisError("not-found", "queue id unknown");
    const data = await client.graphql(QUEUE_OPERATIONS_QUERY, { accountTag, queueId: id, start, end });
    const rows = data.queueMessageOperationsAdaptiveGroups ?? [];
    if (rows.length === 0) return ["no message operations recorded"];
    return rows.map((row) => {
      const d = row.dimensions ?? {};
      return `${d.actionType ?? "?"}${d.consumerType ? `/${d.consumerType}` : ""}${d.outcome ? ` outcome=${d.outcome}` : ""}: count=${row.count ?? 0}, avgRetry=${row.avg?.retryCount ?? "?"}, avgLagMs=${row.avg?.lagTime ?? "?"}`;
    });
  });

  await section(report, `Summary queue backlog by hour (last ${hours}h)`, async () => {
    const id = queueId(SUMMARY_QUEUE);
    if (!id) throw new DiagnosisError("not-found", "queue id unknown");
    const data = await client.graphql(QUEUE_BACKLOG_QUERY, { accountTag, queueId: id, start, end });
    const rows = data.queueBacklogAdaptiveGroups ?? [];
    if (rows.length === 0) return ["no backlog samples"];
    return rows.map((row) => `${row.dimensions?.datetimeHour ?? "?"}: messages=${row.avg?.messages ?? "?"}`);
  });

  await section(report, `Summary queue consumer concurrency by hour (last ${hours}h)`, async () => {
    const id = queueId(SUMMARY_QUEUE);
    if (!id) throw new DiagnosisError("not-found", "queue id unknown");
    const data = await client.graphql(QUEUE_CONSUMER_QUERY, { accountTag, queueId: id, start, end });
    const rows = data.queueConsumerMetricsAdaptiveGroups ?? [];
    if (rows.length === 0) return ["no consumer activity recorded"];
    return rows.map((row) => `${row.dimensions?.datetimeHour ?? "?"}: concurrency=${row.avg?.concurrency ?? "?"}`);
  });

  await section(report, `SUMMARY_CACHE KV operations per day (${startDate}..${endDate})`, async () => {
    const data = await client.graphql(KV_OPERATIONS_QUERY, {
      accountTag,
      namespaceId: SUMMARY_KV_NAMESPACE,
      start: startDate,
      end: endDate,
    });
    const rows = data.kvOperationsAdaptiveGroups ?? [];
    if (rows.length === 0) return ["no KV operations recorded"];
    return sumBy(rows, (r) => `${r.dimensions?.date ?? "?"} ${r.dimensions?.actionType ?? "?"}`, (r) => r.sum?.requests)
      .map(([key, n]) => `${key}: ${n}`);
  });

  await section(report, `KV writes per day, whole account (free limit ${KV_FREE_DAILY_WRITE_LIMIT}/day)`, async () => {
    const data = await client.graphql(KV_ACCOUNT_WRITES_QUERY, { accountTag, start: startDate, end: endDate });
    const rows = (data.kvOperationsAdaptiveGroups ?? []).filter((r) => r.dimensions?.actionType === "write");
    facts.kvWritesByDay = sumBy(rows, (r) => r.dimensions?.date ?? "?", (r) => r.sum?.requests);
    const byNamespace = sumBy(rows, (r) => `${r.dimensions?.date ?? "?"} ns=${r.dimensions?.namespaceId ?? "?"}`, (r) => r.sum?.requests);
    if (facts.kvWritesByDay.length === 0) return ["no KV writes recorded"];
    return [
      ...facts.kvWritesByDay.map(([date, n]) => `${date}: writes=${n}`),
      ...byNamespace.map(([key, n]) => `  ${key}: ${n}`),
    ];
  });

  for (const scriptName of SCRIPTS) {
    await section(report, `Worker invocations: ${scriptName} (last ${hours}h)`, async () => {
      const data = await client.graphql(WORKER_INVOCATIONS_QUERY, { accountTag, scriptName, start, end });
      const rows = data.workersInvocationsAdaptive ?? [];
      const statuses = sumBy(rows, (r) => r.dimensions?.status ?? "?", (r) => r.sum?.requests);
      if (scriptName === "tech-dashboard-summarizer") facts.summarizerStatuses = statuses;
      if (rows.length === 0) return ["no invocations recorded"];
      const errors = rows.reduce((n, r) => n + Number(r.sum?.errors ?? 0), 0);
      return [
        `by status: ${statuses.map(([s, n]) => `${s}=${n}`).join(", ")}`,
        `errors=${errors}`,
      ];
    });
  }

  await section(report, `Summarizer log events (last ${hours}h, newest ${MAX_EVENTS})`, async () => {
    const body = await client.post(`${client.accountPath}/workers/observability/telemetry/query`, {
      queryId: "techdb-summary-diagnosis",
      timeframe: { from: nowMs - hours * 3_600_000, to: nowMs },
      view: "events",
      limit: MAX_EVENTS,
      parameters: {
        filters: [
          { key: "$workers.scriptName", operation: "eq", type: "string", value: "tech-dashboard-summarizer" },
        ],
      },
    });
    const events = body?.result?.events?.events ?? body?.result?.events ?? [];
    if (!Array.isArray(events) || events.length === 0) return ["no log events returned"];
    return events.slice(0, MAX_EVENTS).map((e) => {
      const at = e?.timestamp ? new Date(Number(e.timestamp)).toISOString() : "?";
      const level = e?.$metadata?.level ?? e?.level ?? "?";
      const message = e?.$metadata?.message ?? e?.message ?? e?.$metadata?.error ?? JSON.stringify(e?.source ?? "").slice(0, MAX_TEXT);
      return `${at} [${level}] ${redact(message, client.secrets)}`;
    });
  });

  return { report, verdicts: buildVerdicts(facts), window: { start, end } };
}

export function formatReport({ report, verdicts, window }) {
  const out = [];
  out.push("## Cloudflare summary pipeline diagnosis");
  out.push("");
  out.push(`Window: ${window.start} .. ${window.end} (read-only)`);
  out.push("");
  out.push("### Verdicts");
  for (const v of verdicts) out.push(`- ${v}`);
  for (const s of report) {
    out.push("");
    out.push(`### ${s.ok ? "OK" : "FAILED"}: ${s.title}`);
    out.push("```");
    for (const line of s.lines) out.push(line);
    out.push("```");
  }
  return `${out.join("\n")}\n`;
}

async function main() {
  const { hours } = parseArgs(process.argv.slice(2));
  const client = createClient({
    token: process.env.CLOUDFLARE_API_TOKEN,
    accountId: process.env.CLOUDFLARE_ACCOUNT_ID,
  });
  const result = await diagnose({ client, hours });
  const text = formatReport(result);
  process.stdout.write(text);
  if (process.env.GITHUB_STEP_SUMMARY) appendFileSync(process.env.GITHUB_STEP_SUMMARY, text);
  if (result.report.every((s) => !s.ok)) {
    console.error("::error::every diagnosis section failed; check the token permissions shown above");
    process.exitCode = 1;
  }
}

if (import.meta.url === pathToFileURL(process.argv[1] ?? "").href) {
  main().catch((err) => {
    const category = err instanceof DiagnosisError ? err.category : "unexpected";
    console.error(`::error::diagnosis failed (${category}): ${redact(err instanceof Error ? err.message : err, [process.env.CLOUDFLARE_API_TOKEN, process.env.CLOUDFLARE_ACCOUNT_ID])}`);
    process.exitCode = 1;
  });
}
