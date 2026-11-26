/**
 * tests/deploy-workers-workflow.test.ts
 *
 * Guards the manual Worker deploy workflow (R-001 exception, 2026-11-26).
 * The repository is public, so the workflow must never run from a fork PR or
 * an ordinary push, must keep its secrets in a protected environment, and
 * must hand the Anthropic key to wrangler on stdin so it never hits a log.
 */
import { readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

const ROOT = join(import.meta.dirname, "..");
const workflow = readFileSync(join(ROOT, ".github/workflows/deploy-workers.yml"), "utf8");
const onSection = workflow.slice(workflow.indexOf("\non:\n"), workflow.indexOf("\npermissions:\n"));

describe("Deploy Workers workflow", () => {
  it("only runs when dispatched by hand", () => {
    expect(onSection).toContain("workflow_dispatch:");
    expect(onSection).not.toMatch(/^\s{2}(push|pull_request|pull_request_target|schedule|workflow_run|repository_dispatch):/m);
    expect(workflow).toContain("if: github.repository == 'himiyosh/tech-dashboard'");
  });

  it("keeps secrets in the protected environment with read-only repository access", () => {
    expect(workflow).toMatch(/^permissions:\n {2}contents: read$/m);
    expect(workflow).not.toContain("id-token: write");
    expect(workflow).toMatch(/environment:\n\s+name: cloudflare-production/);
    expect(workflow).toContain("persist-credentials: false");
  });

  it("limits the bridge to main and consumers to main or develop", () => {
    expect(workflow).toContain(
      "consumers:refs/heads/main | consumers:refs/heads/develop | bridge:refs/heads/main) ;;",
    );
  });

  it("passes secrets through env and the API key through stdin only", () => {
    const secretUses = [...workflow.matchAll(/\$\{\{\s*secrets\.([A-Z_]+)\s*\}\}/g)];
    expect(new Set(secretUses.map((match) => match[1]))).toEqual(
      new Set(["CLOUDFLARE_API_TOKEN", "CLOUDFLARE_ACCOUNT_ID", "ANTHROPIC_API_KEY"]),
    );
    for (const line of workflow.split("\n").filter((l) => l.includes("${{ secrets."))) {
      expect(line, line).toMatch(/^\s+[A-Z_]+: \$\{\{ secrets\.[A-Z_]+ \}\}$/);
    }
    expect(workflow).toContain(
      `printf '%s' "\${ANTHROPIC_API_KEY}" | npx --no-install wrangler secret put ANTHROPIC_API_KEY`,
    );
    expect(workflow).not.toMatch(/secret put ANTHROPIC_API_KEY\s+["$]/);
    expect(workflow).not.toMatch(/set -x|echo "\$\{?ANTHROPIC_API_KEY/);
  });

  it("is the only workflow that deploys Workers", () => {
    const dir = join(ROOT, ".github/workflows");
    for (const file of readdirSync(dir).filter((name) => /\.ya?ml$/.test(name))) {
      if (file === "deploy-workers.yml") continue;
      expect(readFileSync(join(dir, file), "utf8"), file).not.toMatch(/wrangler (deploy|secret)/);
    }
  });
});
