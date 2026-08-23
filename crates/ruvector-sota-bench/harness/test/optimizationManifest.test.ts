import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import test from "node:test";
import {
  assertOptimizationManifest,
  declaredLevers,
  declaredObjectives,
  deriveResearchManifestFields,
  loadOptimizationManifest,
  type OptimizationManifest,
} from "../src/optimizationManifest.js";
import { POLICY_LEVERS } from "../src/benchmark.js";

const repoRoot = resolve(import.meta.dirname, "../../../../..");
const FIXTURE = resolve(repoRoot, "schemas/fixtures/optimization-manifest-v1.json");
const SCHEMA = resolve(repoRoot, "schemas/optimization-manifest-v1.json");

async function fixture(): Promise<OptimizationManifest> {
  const manifest = JSON.parse(await readFile(FIXTURE, "utf8")) as OptimizationManifest;
  return manifest;
}

const SEEDS = { exploration: [2, 3, 5, 7, 11], confirmation: [13, 17, 19, 23, 29] };
const COMMIT = "a".repeat(40);

test("the shipped fixture manifest is accepted", async () => {
  const manifest = await loadOptimizationManifest(FIXTURE);
  assert.equal(manifest.repository, "ruvnet/ruvector");
  assert.deepEqual(declaredLevers(manifest), ["ef_search", "m", "ef_construction", "runner_set"]);
  assert.deepEqual(declaredObjectives(manifest).map((entry) => entry.name),
    ["primary", "costPerWin", "p99Us"]);
});

test("a lever the native runner would reject is refused", async () => {
  const manifest = await fixture();
  manifest.levers.push({ name: "temperature" });
  assert.throws(() => assertOptimizationManifest(manifest), /the runner would reject: temperature/);
});

test("the schema lever enum cannot drift from the runner's closed set", async () => {
  // Two copies of the lever list exist: the schema enum, for offline
  // validation, and POLICY_LEVERS, which normalizePolicy actually enforces.
  // This test is what keeps them the same list.
  const schema = JSON.parse(await readFile(SCHEMA, "utf8")) as {
    $defs: { lever_name: { enum: string[] } };
  };
  assert.deepEqual([...schema.$defs.lever_name.enum].sort(), [...POLICY_LEVERS].sort());
});

test("a missing objective is refused", async () => {
  const manifest = await fixture() as unknown as Record<string, unknown>;
  delete manifest.objective;
  assert.throws(() => assertOptimizationManifest(manifest), /objective must be an object/);
});

test("a non-positive minimum meaningful effect is refused", async () => {
  const manifest = await fixture();
  manifest.objective.minimum_meaningful_effect = 0;
  assert.throws(() => assertOptimizationManifest(manifest), /must exceed zero/);
});

test("an out-of-order lever bound is refused", async () => {
  const manifest = await fixture();
  manifest.levers[0] = { name: "ef_search", bounds: { minimum: 4096, maximum: 1 } };
  assert.throws(() => assertOptimizationManifest(manifest), /minimum above maximum/);
});

test("a manifest cannot switch the non-dominated requirement off", async () => {
  const manifest = await fixture() as OptimizationManifest;
  (manifest.promotion.pareto as { require_non_dominated: boolean }).require_non_dominated = false;
  assert.throws(() => assertOptimizationManifest(manifest), /cannot disable the non-dominated/);
});

test("a manifest cannot loosen alpha or the comparison", async () => {
  const loosened = await fixture() as OptimizationManifest;
  (loosened.promotion as { alpha: number }).alpha = 0.2;
  assert.throws(() => assertOptimizationManifest(loosened), /alpha must be 0.05/);
  const unpaired = await fixture() as OptimizationManifest;
  (unpaired.promotion as { comparison: string }).comparison = "unpaired";
  assert.throws(() => assertOptimizationManifest(unpaired), /must be paired/);
});

test("a non-finite budget ceiling is refused", async () => {
  const manifest = await fixture();
  manifest.budget.cost_ceiling_factor = Number.NaN;
  assert.throws(() => assertOptimizationManifest(manifest), /must be a finite number/);
});

test("an unknown runner set is refused", async () => {
  const manifest = await fixture();
  manifest.benchmark.runner_sets = ["core", "turbo"];
  assert.throws(() => assertOptimizationManifest(manifest), /unknown runner_set: turbo/);
});

test("derivation produces research manifest fields from the standing policy", async () => {
  const manifest = await fixture();
  const derived = deriveResearchManifestFields({
    manifest,
    independentVariable: "ef_search",
    commit: COMMIT,
    revision: 1,
    claim: "raising ef_search improves recall at fixed build cost",
    explorationSeeds: SEEDS.exploration,
    confirmationSeeds: SEEDS.confirmation,
  });
  assert.equal(derived.schema_version, 1);
  assert.equal(derived.phase, "confirmation");
  assert.equal(derived.independent_variable, "ef_search");
  const rule = derived.decision_rule as Record<string, unknown>;
  assert.equal(rule.primary_metric, "primary");
  assert.equal(rule.minimum_meaningful_effect, 0.005);
  assert.equal(rule.expected_direction, "greater");
  assert.equal(rule.alpha, 0.05);
  assert.equal(rule.comparison, "paired");
});

test("derivation refuses a variable the manifest never declared", async () => {
  const manifest = await fixture();
  assert.throws(() => deriveResearchManifestFields({
    manifest,
    independentVariable: "k",
    commit: COMMIT,
    revision: 1,
    claim: "c",
    explorationSeeds: SEEDS.exploration,
    confirmationSeeds: SEEDS.confirmation,
  }), /not a lever declared/);
});

test("derivation refuses overlapping exploration and confirmation seeds", async () => {
  const manifest = await fixture();
  assert.throws(() => deriveResearchManifestFields({
    manifest,
    independentVariable: "ef_search",
    commit: COMMIT,
    revision: 1,
    claim: "c",
    explorationSeeds: [2, 3, 5, 7, 11],
    confirmationSeeds: [11, 17, 19, 23, 29],
  }), /must be disjoint; shared: 11/);
});

test("derivation refuses a short commit or too few seeds", async () => {
  const manifest = await fixture();
  const base = {
    manifest,
    independentVariable: "ef_search",
    revision: 1,
    claim: "c",
    explorationSeeds: SEEDS.exploration,
    confirmationSeeds: SEEDS.confirmation,
  };
  assert.throws(() => deriveResearchManifestFields({ ...base, commit: "abc" }),
    /40-character commit/);
  assert.throws(
    () => deriveResearchManifestFields({ ...base, commit: COMMIT, confirmationSeeds: [1, 2] }),
    /at least five seeds/,
  );
});

test("a manifest that is not valid JSON reports the file", async () => {
  await assert.rejects(
    loadOptimizationManifest(resolve(repoRoot, "README.md")),
  );
});
