/**
 * summarize.ts - Anthropic Claude API による summary-only 補完.
 *
 * GitHub Copilot は使わない (サイトオーナー要望 2026-11-26)。呼び出しは
 * worker/src/anthropic-client.ts を Queue consumer と共有する。
 *
 * 認証: ANTHROPIC_API_KEY (.env.local などで本人が設定)。未設定なら要約
 * フェーズを明示的な警告付きでスキップする (ローカル dev での軽量実行)。
 *
 * 環境変数:
 *   ANTHROPIC_API_KEY     … 必須 (未設定ならスキップ)
 *   SUMMARIZE_MODEL       … 既定 "claude-opus-5-5"。claude-* 以外は起動時にエラー
 *   SUMMARIZE_EFFORT      … 既定 "medium" (low/medium/high/xhigh/max)
 *   SUMMARIZE_MAX_NEW     … 1 ラン当たりの新規要約上限 (既定 15)
 *   SUMMARIZE_MAX_TOKENS  … thinking + 回答の合計上限 (既定 16000)
 *   SUMMARIZE_TIMEOUT_MS  … API 呼び出しのタイムアウト (既定 180000)
 *   SUMMARIZE_CONCURRENCY … API 呼び出しの並列数 (既定 4)
 *
 * キャッシュ: data/_summary-cache.json (URL キー)
 * 現行 contract は summary-only。bodyJa/bodyEn は legacy cache の保持用で、
 * 新規補完では空文字列のまま保存する。
 */
import { readFile, writeFile, mkdir } from "node:fs/promises";
import { dirname, join } from "node:path";
import type { NormalizedEntry } from "../types.ts";
import { buildSummaryPrompt, parseResponse } from "../../worker/src/prompt.ts";
import {
  hasUsableGroundedBilingualSummary,
  hasUsableBilingualSummary,
  needsSummaryGeneration,
  type SummaryQualityInput,
} from "./summary-quality.ts";
import { hasSufficientSourceGrounding } from "./source-grounding.ts";
import { normalizeTags } from "./tag.ts";
import {
  ANTHROPIC_API_KEY_MISSING,
  callAnthropicText,
  DEFAULT_ANTHROPIC_MODEL,
  parseAnthropicEffort,
  parseModelChain,
  parsePositiveInt,
  readAnthropicKey,
  type AnthropicEffort,
} from "../../worker/src/anthropic-client.ts";

interface CacheEntry {
  titleJa: string;
  summaryJa: string;
  summaryEn: string;
  bodyJa: string;
  bodyEn: string;
  importance: 1 | 2 | 3;
  extraTags: string[];
  model: string;
  cachedAt: string;
}

type Cache = Record<string, CacheEntry>;

/** The Claude model for summaries; non-Claude ids throw (see parseModelChain). */
export function resolveSummarizeModel(
  model = process.env.SUMMARIZE_MODEL,
): string {
  return parseModelChain(model?.trim() || DEFAULT_ANTHROPIC_MODEL, undefined)[0]!;
}

const MODEL = resolveSummarizeModel();
const MAX_NEW = Number(process.env.SUMMARIZE_MAX_NEW ?? "15");
const MAX_TOKENS = parsePositiveInt(process.env.SUMMARIZE_MAX_TOKENS, 16_000, "SUMMARIZE_MAX_TOKENS");
const REQUEST_TIMEOUT_MS = Number(process.env.SUMMARIZE_TIMEOUT_MS ?? "180000");
const CONCURRENCY = Number(process.env.SUMMARIZE_CONCURRENCY ?? "4");

const SUMMARY_SYSTEM_PROMPT =
  "あなたは技術記事を日本語と英語の両方で要約するエディターです。指示された JSON 形式のみを返してください。You are an editor who summarises tech articles in both Japanese and English. Return only the requested JSON.";

interface SummaryBackend {
  anthropicKey: string;
  effort: AnthropicEffort;
  model: string;
}
export function needsGeneratedContent(entry: SummaryQualityInput): boolean {
  return needsSummaryGeneration(entry);
}

async function loadCache(path: string): Promise<Cache> {
  try {
    const txt = await readFile(path, "utf8");
    return JSON.parse(txt) as Cache;
  } catch {
    return {};
  }
}

async function saveCache(path: string, cache: Cache): Promise<void> {
  await mkdir(dirname(path), { recursive: true });
  await writeFile(path, JSON.stringify(cache, null, 2) + "\n", "utf8");
}

function resolveSummaryBackend(): SummaryBackend | null {
  const anthropicKey = readAnthropicKey(process.env);
  if (!anthropicKey) return null;
  return {
    anthropicKey,
    effort: parseAnthropicEffort(process.env.SUMMARIZE_EFFORT, "medium"),
    model: MODEL,
  };
}

async function callSummaryModel(
  backend: SummaryBackend,
  entry: NormalizedEntry,
): Promise<CacheEntry> {
  const result = await callAnthropicText(
    {
      model: backend.model,
      systemPrompt: SUMMARY_SYSTEM_PROMPT,
      userPrompt: buildSummaryPrompt(entry),
      maxTokens: MAX_TOKENS,
      effort: backend.effort,
    },
    { anthropicKey: backend.anthropicKey, timeoutMs: REQUEST_TIMEOUT_MS },
  );
  return toSummaryCacheEntry(result.text, entry, result.model);
}

function toSummaryCacheEntry(
  text: string,
  entry: NormalizedEntry,
  model: string,
): CacheEntry {
  const parsed = parseModelResponse(text);
  if (!isCompleteSummaryResponse(parsed, entry)) {
    throw new Error("model response missing required clean summary fields");
  }
  return {
    titleJa: parsed.titleJa,
    summaryJa: parsed.summaryJa,
    summaryEn: parsed.summaryEn,
    bodyJa: "",
    bodyEn: "",
    importance: parsed.importance,
    extraTags: parsed.extraTags,
    model,
    cachedAt: new Date().toISOString(),
  };
}

export function parseModelResponse(text: string): {
  titleJa: string;
  summaryJa: string;
  summaryEn: string;
  bodyJa: string;
  bodyEn: string;
  importance: 1 | 2 | 3;
  extraTags: string[];
} {
  return parseResponse(text);
}

export function isCompleteSummaryResponse(
  parsed: ReturnType<typeof parseModelResponse>,
  source: SummaryQualityInput = {},
): boolean {
  return Boolean(
    parsed.titleJa
      && hasUsableBilingualSummary(
        parsed,
        [source.title, source.titleJa, source.titleEn],
      )
      && hasUsableGroundedBilingualSummary(source, parsed),
  );
}

async function runWithConcurrency<T, R>(
  items: T[],
  fn: (item: T) => Promise<R>,
  concurrency: number,
): Promise<R[]> {
  const results: R[] = new Array(items.length);
  let idx = 0;
  async function worker() {
    while (true) {
      const i = idx++;
      if (i >= items.length) return;
      const item = items[i]!;
      try {
        results[i] = await fn(item);
      } catch {
        await new Promise((r) => setTimeout(r, 2000));
        results[i] = await fn(item);
      }
    }
  }
  const workers = Array.from({ length: concurrency }, () => worker());
  await Promise.all(workers);
  return results;
}

export interface SummarizeResult {
  entries: NormalizedEntry[];
  stats: {
    cached: number;
    summarized: number;
    skipped: number;
    errors: number;
  };
}

export function stripIndexBodies(entry: NormalizedEntry): NormalizedEntry {
  return {
    ...entry,
    bodyJa: "",
    bodyEn: "",
  };
}

/**
 * Enhances entries with summaryJa/importance/extra tags via the Anthropic Claude API.
 * トークンが解決できない場合は no-op で透過 (ローカル dev を妨げない)。
 */
export async function summarize(
  entries: NormalizedEntry[],
  dataDir: string,
): Promise<SummarizeResult> {
  const stats = { cached: 0, summarized: 0, skipped: 0, errors: 0 };

  const cachePath = join(dataDir, "_summary-cache.json");
  const cache = await loadCache(cachePath);

  // 1) Apply cached summaries before token resolution so local preview keeps
  // Japanese titles/summaries even when the API key is temporarily unavailable.
  const needsSummary: NormalizedEntry[] = [];
  const out = entries.map((e) => {
    const hit = cache[e.url];
    const cachedTitleJa = hit?.titleJa || e.titleJa;
    const groundedCacheHit = Boolean(
      hit &&
      cachedTitleJa &&
      hit.model !== "deterministic-fallback" &&
      hasUsableGroundedBilingualSummary(
        e,
        {
          ...hit,
          title: e.title,
          titleJa: cachedTitleJa,
          titleEn: e.titleEn,
        },
      ),
    );
    if (hit && groundedCacheHit) {
      const cachedEntry = stripIndexBodies({
        ...e,
        titleJa: cachedTitleJa,
        summaryJa: hit.summaryJa,
        summaryEn: hit.summaryEn,
        importance: hit.importance,
        tags: dedupeTags([...e.tags, ...hit.extraTags]),
      });
      // Re-summarize only missing/fallback summaries. Legacy body fields may
      // exist in cache, but index entries must stay body-free (LL-115).
      if (needsGeneratedContent(cachedEntry)) {
        needsSummary.push(e);
        return stripIndexBodies(e);
      }
      stats.cached++;
      return cachedEntry;
    }
    if (
      (needsGeneratedContent(e) || Boolean(hit && !groundedCacheHit)) &&
      hasSufficientSourceGrounding(e)
    ) {
      needsSummary.push(e);
    }
    return stripIndexBodies(e);
  });

  const backend = resolveSummaryBackend();
  if (!backend) {
    console.warn(
      `[summarize] skipping ${needsSummary.length} entries: ${ANTHROPIC_API_KEY_MISSING}`,
    );
    stats.skipped = needsSummary.length;
    return { entries: out, stats };
  }

  // 2) Budget-cap newcomers.
  const toSummarize = needsSummary.slice(0, MAX_NEW);
  stats.skipped = needsSummary.length - toSummarize.length;

  if (toSummarize.length === 0) {
    return { entries: out, stats };
  }

  console.log(
    `[summarize] ${toSummarize.length} entries → anthropic:${backend.model} (cache=${stats.cached} skipped=${stats.skipped})`,
  );

  let cacheSaveChain = Promise.resolve();
  const queueCacheSave = () => {
    cacheSaveChain = cacheSaveChain.then(() => saveCache(cachePath, cache));
    return cacheSaveChain;
  };
  let completed = 0;

  const fetched = await runWithConcurrency(
    toSummarize,
    async (e) => {
      for (let attempt = 1; attempt <= 2; attempt++) {
        try {
          const r = await callSummaryModel(backend, e);
          cache[e.url] = r;
          stats.summarized++;
          completed++;
          console.log(`[summarize] ok ${completed}/${toSummarize.length} ${e.url}`);
          await queueCacheSave();
          return { url: e.url, entry: r, ok: true as const };
        } catch (err) {
          const msg = err instanceof Error ? err.message : String(err);
          if (attempt === 1) {
            console.warn(`[summarize] retry ${e.url}: ${msg}`);
            continue;
          }
          stats.errors++;
          completed++;
          console.warn(`[summarize] err ${completed}/${toSummarize.length} ${e.url}: ${msg}`);
          return { url: e.url, ok: false as const };
        }
      }
      return { url: e.url, ok: false as const };
    },
    CONCURRENCY,
  );
  await cacheSaveChain;

  // 3) Merge fetched summaries into output.
  const mergedByUrl = new Map(
    fetched.filter((f) => f.ok).map((f) => [f.url, f.entry] as const),
  );
  const finalEntries = out.map((e) => {
    const r = mergedByUrl.get(e.url);
    if (!r) return stripIndexBodies(e);
    return stripIndexBodies({
      ...e,
      titleJa: r.titleJa || e.titleJa,
      summaryJa: r.summaryJa,
      summaryEn: r.summaryEn || e.summaryEn,
      importance: r.importance,
      tags: dedupeTags([...e.tags, ...r.extraTags]),
    });
  });

  await saveCache(cachePath, cache);
  return { entries: finalEntries, stats };
}

function dedupeTags(tags: string[]): string[] {
  return normalizeTags(tags, 10);
}
