import assert from "node:assert/strict";
import { mkdtemp, readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import type { Score } from "@metaharness/flywheel";
import { runRuvectorFlywheel, ruvectorPromotionRule } from "../src/flywheel.js";

/** Build a gate-visible score. Defaults keep every gate other than
 *  `hard_regression` silent, so a reason list is attributable to one check. */
function gateScore(overrides: {
  primarySamples?: number[];
  recallAt10?: number;
  costPerWin?: number;
  regressions?: string[];
  vetoReasons?: string[];
}): Score {
  const primarySamples = overrides.primarySamples ?? [0.9, 0.9, 0.9, 0.9, 0.9];
  const recallAt10 = overrides.recallAt10 ?? 0.9;
  const regressions = overrides.regressions ?? [];
  const vetoReasons = overrides.vetoReasons ?? [];
  return {
    primary: primarySamples[0]!,
    primarySamples,
    noopRate: 1 - recallAt10,
    recallAt10,
    qps: 2_000,
    memoryMb: 40,
    peakRssBytes: 0,
    costPerWin: overrides.costPerWin ?? 1,
    regressed: regressions.length > 0 || vetoReasons.length > 0,
    regressions,
    vetoReasons,
    observations: [],
  } as unknown as Score;
}

const hardRegressions = (reasons: string[]) =>
  reasons.filter((reason) => reason === "hard_regression" || reason.startsWith("hard_regression:"));

test("hard_regression stays silent when the candidate inherits the baseline's regressions", () => {
  const regressions = ["recall_below_0.95"];
  const { reasons } = ruvectorPromotionRule({
    baseline: gateScore({ regressions }),
    candidate: gateScore({ regressions }),
  });
  assert.deepEqual(hardRegressions(reasons), []);
});

test("hard_regression fires and names a regression the candidate adds to the baseline's", () => {
  const { reasons } = ruvectorPromotionRule({
    baseline: gateScore({ regressions: ["recall_below_0.95"] }),
    candidate: gateScore({ regressions: ["recall_below_0.95", "qps_below_baseline_floor"] }),
  });
  assert.deepEqual(hardRegressions(reasons), ["hard_regression:qps_below_baseline_floor"]);
});

test("hard_regression fires when the candidate trades one regression for a different one", () => {
  const { reasons } = ruvectorPromotionRule({
    baseline: gateScore({ regressions: ["recall_below_0.95"] }),
    candidate: gateScore({ regressions: ["qps_below_baseline_floor"] }),
  });
  assert.deepEqual(hardRegressions(reasons), ["hard_regression:qps_below_baseline_floor"]);
});

test("hard_regression fires when a clean baseline is replaced by a regressing candidate", () => {
  const { reasons } = ruvectorPromotionRule({
    baseline: gateScore({}),
    candidate: gateScore({ regressions: ["recall_below_0.95", "qps_below_baseline_floor"] }),
  });
  assert.deepEqual(hardRegressions(reasons), [
    "hard_regression:qps_below_baseline_floor",
    "hard_regression:recall_below_0.95",
  ]);
});

test("hard_regression stays silent when neither side regresses", () => {
  const { reasons } = ruvectorPromotionRule({ baseline: gateScore({}), candidate: gateScore({}) });
  assert.deepEqual(hardRegressions(reasons), []);
});

test("a score with no itemised regressions still fails closed on the boolean", () => {
  const legacy = gateScore({});
  delete (legacy as unknown as Record<string, unknown>).regressions;
  (legacy as unknown as Record<string, unknown>).regressed = true;
  const { reasons } = ruvectorPromotionRule({ baseline: gateScore({}), candidate: legacy });
  assert.deepEqual(hardRegressions(reasons), ["hard_regression"]);
});

test("vetoes remain absolute and are not diffed against the baseline", () => {
  const vetoReasons = ["redblue_live_credential_detected"];
  const { promote, reasons } = ruvectorPromotionRule({
    baseline: gateScore({ vetoReasons }),
    candidate: gateScore({ vetoReasons }),
  });
  assert.equal(promote, false);
  assert.ok(reasons.includes("redblue_live_credential_detected"));
});

// The real baseline/candidate pair from the 30-generation acceptance run in issue
// #920: `adapt m`, strictly better on primary, recall, qps, p99 and cost, rejected
// with `hard_regression` as its sole reason because both sides sat below the
// absolute 0.95 recall floor.
test("the issue #920 rejected candidate is no longer blocked by hard_regression", () => {
  const baseline = gateScore({
    primarySamples: [
      0.8781716212500001, 0.8597841212500001, 0.5462757812500001,
      0.8648141012499999, 0.6410117651373053,
    ],
    recallAt10: 0.888,
    costPerWin: 22.6502708788278,
    regressions: ["recall_below_0.95"],
  });
  const candidate = gateScore({
    primarySamples: [
      0.9473974412499999, 0.9617474412500001, 0.9590824612500001,
      0.9598274412500001, 0.9593016212500001,
    ],
    recallAt10: 0.9359999999999999,
    costPerWin: 0.693375479272705,
    regressions: ["recall_below_0.95"],
  });
  const { reasons } = ruvectorPromotionRule({ baseline, candidate });
  assert.deepEqual(hardRegressions(reasons), []);
});

test("flywheel emits a replay-verified bundle and cannot authorize a PR", async () => {
  const outputDir = await mkdtemp(join(tmpdir(), "ruvector-flywheel-test-"));
  const seenSuites = new Set<string>();
  const seenItems = new Set<string>();
  const result = await runRuvectorFlywheel({
    repoRoot: process.cwd(),
    outputDir,
    generations: 2,
    benchmark: async (policy, item, suite) => {
      seenSuites.add(suite.id);
      seenItems.add(`${item.dataset}:${item.seed}`);
      const ef = Number(policy.ef_search);
      return {
        scores: [{
          index: "fixture",
          dataset: "fixture",
          recall: { recall_at_10: 0.96 + Math.min(ef, 200) / 10_000 },
          qps: 2_000,
          memory_mb: 40,
          latency: { p99_us: 500 },
        }],
      };
    },
  });
  assert.equal(result.generationsRun, 2);
  assert.deepEqual([...seenSuites].sort(), ["ruvector-confirmation", "ruvector-frozen-anchor"]);
  assert.equal(seenItems.size, 10);
  const gateInput = JSON.parse(await readFile(join(outputDir, "research-gate-input.json"), "utf8")) as {
    replay_verified: boolean;
    pr_creation_authorized: boolean;
  };
  assert.equal(gateInput.replay_verified, true);
  assert.equal(gateInput.pr_creation_authorized, false);
});

test("overlapping holdout and anchor identities fail closed", async () => {
  const item = {
    seed: 7,
    dataset: "smoke-128",
    dataset_sha256: "a".repeat(64),
    kind: "smoke" as const,
  };
  await assert.rejects(() => runRuvectorFlywheel({
    repoRoot: process.cwd(),
    generations: 1,
    holdoutItems: [item],
    anchorItems: [item],
    benchmark: async () => ({ scores: [] }),
  }), /must be disjoint/);
});
