/**
 * tests/workflow-lint.test.ts
 *
 * Guards the CI workflow lint job (added 2026-11-26). Workflow YAML used to be
 * checked only for syntax, so expression, action input and shell errors could
 * reach main. The job must stay pinned, run without secrets on a read-only
 * token (the repository is public and fork PRs run CI), and not wait on the
 * main data snapshot, so a stale index never hides a workflow error.
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

const ROOT = join(import.meta.dirname, "..");
const ci = readFileSync(join(ROOT, ".github/workflows/ci.yml"), "utf8");
const start = ci.indexOf("\n  workflow-lint:\n");
const end = ci.indexOf("\n  data-snapshot:\n");
const job = start >= 0 && end > start ? ci.slice(start, end) : "";

describe("CI workflow lint job", () => {
  it("exists in ci.yml", () => {
    expect(job.length).toBeGreaterThan(0);
  });

  it("runs on push and pull_request like the rest of CI", () => {
    const onSection = ci.slice(ci.indexOf("\non:\n"), ci.indexOf("\njobs:\n"));
    expect(onSection).toMatch(/^\s{2}push:/m);
    expect(onSection).toMatch(/^\s{2}pull_request:/m);
    expect(onSection).not.toContain("pull_request_target");
  });

  it("pins actionlint to an exact release", () => {
    expect(job).toMatch(/ACTIONLINT_VERSION: v\d+\.\d+\.\d+\n/);
    expect(job).toContain('go install "github.com/rhysd/actionlint/cmd/actionlint@${ACTIONLINT_VERSION}"');
    expect(job).not.toMatch(/actionlint@(latest|main|master)/);
  });

  it("uses no secrets and a read-only token", () => {
    expect(job).toMatch(/permissions:\n\s+contents: read\n/);
    expect(job).not.toMatch(/\$\{\{\s*secrets\./);
    expect(job).not.toContain("environment:");
    expect(job).toContain("persist-credentials: false");
  });

  it("does not depend on the main data snapshot", () => {
    expect(job).not.toMatch(/^\s+needs:/m);
  });

  it("fails when shellcheck is missing instead of silently skipping shell checks", () => {
    expect(job).toContain("command -v shellcheck");
  });
});
