/**
 * AI4AI-Bench recursive-improvement adapter — first shippable slice
 * (PIR WP25, ADR-328).
 *
 * Source benchmark, cited in full on every reference (same naming discipline
 * as ADR-324 §5): AI4AI-Bench (arXiv:2608.20318, "AI4AI-Bench: Benchmarking
 * LLM Agents in Algorithmic Design for Recursive Self-Improvement";
 * github.com/Einsia/AI4AI-Bench, Apache-2.0). The benchmark freezes ten
 * research repositories spanning ten learning-algorithm families and asks an
 * agent to rewrite the learning algorithm itself; hyperparameter tuning and
 * data collection do not count as genuine improvement.
 *
 * WHAT THIS SLICE IS: the manifest/identity discipline, executor seam, score
 * ingest, and mutation-lineage chain that let MetaHarness treat AI4AI-Bench
 * (arXiv:2608.20318) tasks as an EXTERNAL published benchmark — exactly the
 * "published-benchmark" evidence kind ADR-324's external-grounding invariant
 * admits. Every candidate mutation this adapter observes carries reproducible
 * provenance (task identity → parent digest → diff digest → seed → executor
 * identity), so a run's full lineage can be replayed and audited.
 *
 * WHAT THIS SLICE IS NOT (honest scope): a real AI4AI-Bench run executes a
 * frozen repository's training pipeline inside the benchmark's own Docker
 * evaluator on datacenter-GPU hardware (the paper's official runs assume one
 * B300-class GPU). None of that happens here. The adapter takes a
 * CONSTRUCTOR-INJECTED executor (the same seam as
 * `RuvectorFlywheelOptions.benchmark` / `BenchmarkRunOptions.commandPrefixArgs`)
 * and this repo's CI exercises it only against a first-party fixture. The
 * command executor below spawns whatever evaluator entrypoint the caller
 * pins — locally that is a fixture; on real hardware it is the AI4AI-Bench
 * (arXiv:2608.20318) evaluator.
 *
 * PREPRINT-REPRODUCTION RULE (binding, unchanged from Waves 1–3): the paper's
 * published numbers — best 0.250, mean 0.166 across evaluated systems — are
 * HYPOTHESES and targets, never this program's acceptance bar. Promotion of
 * any mutation this adapter observes still routes exclusively through
 * statistics.ts `pairedBootstrapDecision` and the vetoes.ts conjunctive veto
 * algebra; an AI4AI score above 0.250 proves nothing by itself. The score
 * scale matters when reading results: 0.1 is the frozen repository's shipped
 * algorithm, so 0.250 closes less than a fifth of the ship-to-optimum
 * distance.
 *
 * FAIL-LOUD METRIC INTEGRITY (Wave-3 lesson, #888): an executor crash or a
 * non-finite / out-of-range score THROWS — it is never coerced into an
 * observation. NaN defeats every `<` comparison silently, so finiteness is
 * rejected at the choke point before any gate can see the value. Genuine-vs-
 * tuning classification is ingested only when the evaluator output carries
 * it; this adapter never fabricates the distinction.
 *
 * NO CACHING: a real AI4AI-Bench (arXiv:2608.20318) evaluation retrains a
 * stochastic pipeline; replaying a cached score as if it were a fresh run
 * would be evidence laundering. Every call to the executor is a real call.
 *
 * FROZEN WEIGHTS ARE STRUCTURAL: this file lives inside the mutation-surface
 * directory scripts/frozen-weights-check.mjs scans. It imports no weight-
 * updating API and names no model weight file. Rewriting a frozen BENCHMARK
 * repository's learning algorithm is the benchmark's task definition executed
 * by an external evaluator on external hardware — it is not a weight update
 * performed by this harness, and no path in this module persists weights.
 *
 * KNOWN LIMITATION (accepted, exact parity with benchmark.ts): the command
 * executor SIGKILLs only the spawned child, not its process group — an
 * evaluator that forks can leave orphans past the timeout. Inherited from
 * benchmark.ts's runner discipline; fixing it belongs to both call sites at
 * once, not to this adapter alone.
 */
import { spawn } from "node:child_process";
import { createHash, randomBytes } from "node:crypto";
import { mkdtemp, readFile, rm, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { pairedBootstrapDecision, type PairedDecision } from "./statistics.js";
import type { PromotionVetoProvider } from "./vetoes.js";

/** Exact citation for every reference (no bare "AI4AI" — naming discipline). */
export const AI4AI_BENCH_CITATION =
  "AI4AI-Bench (arXiv:2608.20318, github.com/Einsia/AI4AI-Bench)" as const;

/** The benchmark's score for the shipped algorithm — the floor, not zero. */
export const AI4AI_SHIPPED_ALGORITHM_SCORE = 0.1;

const HEX64 = /^[0-9a-f]{64}$/;
const SLUG = /^[a-z0-9][a-z0-9._-]{0,127}$/;

// ---------------------------------------------------------------------------
// FROZEN TASK IDENTITY — mirrors benchmark.ts's dataset-identity discipline.
// ---------------------------------------------------------------------------

/**
 * A pinned AI4AI-Bench (arXiv:2608.20318) task. `repoSnapshotSha256` is the
 * mandatory lowercase SHA-256 over the frozen repository snapshot the task
 * ships; `evaluatorSha256` pins the evaluator that scores submissions. A
 * manifest missing either hash is refused outright — the same posture
 * normalizeSuiteItem takes toward dataset identity: no hash, no run.
 */
export interface Ai4aiTaskManifest {
  /** Benchmark repo slug, e.g. "einsia-ai4ai-bench". */
  readonly benchmark: string;
  /** Task id inside the benchmark, e.g. "task-03-policy-gradient". */
  readonly taskId: string;
  /** The learning-algorithm family the frozen repo implements. */
  readonly algorithmFamily: string;
  /** Lowercase SHA-256 over the frozen repository snapshot. */
  readonly repoSnapshotSha256: string;
  /** Lowercase SHA-256 over the evaluator that scores submissions. */
  readonly evaluatorSha256: string;
  readonly citation: typeof AI4AI_BENCH_CITATION;
}

/** Refuse anything that is not a fully hash-pinned task manifest. */
export function assertAi4aiTaskManifest(value: unknown): asserts value is Ai4aiTaskManifest {
  if (!value || typeof value !== "object") throw new Error("ai4ai task manifest must be an object");
  const manifest = value as Partial<Ai4aiTaskManifest>;
  if (!manifest.benchmark || !SLUG.test(manifest.benchmark)) {
    throw new Error("ai4ai task manifest requires a benchmark slug");
  }
  if (!manifest.taskId || !SLUG.test(manifest.taskId)) {
    throw new Error("ai4ai task manifest requires a task id slug");
  }
  if (!manifest.algorithmFamily || typeof manifest.algorithmFamily !== "string") {
    throw new Error("ai4ai task manifest requires an algorithm family");
  }
  if (!manifest.repoSnapshotSha256 || !HEX64.test(manifest.repoSnapshotSha256)) {
    throw new Error("ai4ai task manifest requires a lowercase SHA-256 repo snapshot identity");
  }
  if (!manifest.evaluatorSha256 || !HEX64.test(manifest.evaluatorSha256)) {
    throw new Error("ai4ai task manifest requires a lowercase SHA-256 evaluator identity");
  }
  if (manifest.citation !== AI4AI_BENCH_CITATION) {
    throw new Error(`ai4ai task manifest must carry the exact citation "${AI4AI_BENCH_CITATION}"`);
  }
}

function canonical(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
  if (value && typeof value === "object") {
    return `{${Object.entries(value as Record<string, unknown>)
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([key, child]) => `${JSON.stringify(key)}:${canonical(child)}`).join(",")}}`;
  }
  return JSON.stringify(value);
}

/** The frozen task identity: SHA-256 over the canonical manifest encoding. */
export function ai4aiTaskIdentity(manifest: Ai4aiTaskManifest): string {
  assertAi4aiTaskManifest(manifest);
  return createHash("sha256")
    .update("ruvector.ai4ai.task-identity.v1\0")
    .update(canonical(manifest))
    .digest("hex");
}

// ---------------------------------------------------------------------------
// MUTATIONS + LINEAGE — reproducible provenance for every candidate.
// ---------------------------------------------------------------------------

/**
 * One candidate learning-algorithm mutation submitted against a task. The
 * mutation itself (the diff) lives with the executor; the adapter records its
 * digest so lineage is reproducible without the harness ever holding the
 * candidate code.
 */
export interface Ai4aiMutation {
  /** Stable id, e.g. "mut-007". */
  readonly id: string;
  /** Lowercase SHA-256 over the candidate diff against the parent state. */
  readonly diffSha256: string;
  /** The run seed the evaluator is asked to use. */
  readonly seed: number;
  /** One-line description of the algorithmic change (self-reported). */
  readonly description: string;
}

export function assertAi4aiMutation(value: unknown): asserts value is Ai4aiMutation {
  if (!value || typeof value !== "object") throw new Error("ai4ai mutation must be an object");
  const mutation = value as Partial<Ai4aiMutation>;
  if (!mutation.id || !SLUG.test(mutation.id)) throw new Error("ai4ai mutation requires an id slug");
  if (!mutation.diffSha256 || !HEX64.test(mutation.diffSha256)) {
    throw new Error("ai4ai mutation requires a lowercase SHA-256 diff digest");
  }
  if (!Number.isSafeInteger(mutation.seed) || (mutation.seed ?? -1) < 0) {
    throw new Error("ai4ai mutation seed must be a non-negative integer");
  }
  if (!mutation.description) throw new Error("ai4ai mutation requires a description");
}

/**
 * How the evaluator classified a submission. "unreported" means the evaluator
 * output carried no classification — the adapter NEVER fabricates one, per
 * the paper's own headline finding that only genuine algorithm changes count.
 */
export type Ai4aiSubmissionKind = "algorithm-change" | "tuning-only" | "unreported";

/** Parsed, validated evaluator output for one mutation. */
export interface Ai4aiEvaluation {
  /** Finite score in [0, 1]; 0.1 is the shipped algorithm. */
  readonly score: number;
  readonly submissionKind: Ai4aiSubmissionKind;
}

/**
 * Parse raw evaluator output. FAIL-LOUD: a missing, non-numeric, non-finite,
 * or out-of-range score THROWS — it never becomes an observation (Wave-3
 * lesson: NaN silently defeats every comparison-based gate downstream).
 */
export function parseAi4aiEvaluatorOutput(value: unknown): Ai4aiEvaluation {
  if (!value || typeof value !== "object") throw new Error("ai4ai evaluator output must be an object");
  const output = value as { score?: unknown; algorithm_changed?: unknown };
  if (typeof output.score !== "number" || !Number.isFinite(output.score)) {
    throw new Error("ai4ai evaluator score must be a finite number (fail-loud: refusing non-finite)");
  }
  if (output.score < 0 || output.score > 1) {
    throw new Error(`ai4ai evaluator score out of range [0, 1]: ${output.score}`);
  }
  let submissionKind: Ai4aiSubmissionKind = "unreported";
  if (output.algorithm_changed === true) submissionKind = "algorithm-change";
  else if (output.algorithm_changed === false) submissionKind = "tuning-only";
  else if (output.algorithm_changed !== undefined) {
    throw new Error("ai4ai evaluator algorithm_changed must be boolean when present");
  }
  return Object.freeze({ score: output.score, submissionKind });
}

/**
 * One link in the lineage chain. `parentDigest` is the previous record's
 * digest (the ROOT BINDING for the first record — see `ai4aiRunRoot`, which
 * mixes a per-run nonce into the task identity so a chain is bound to the run
 * that produced it and cannot be replayed as a fresh one). `digest` commits to
 * everything — task, parent, diff, seed, executor identity, whether the
 * executor was a declared wrapper, and the evaluation — so the chain is
 * tamper-evident and any run can be re-derived and audited.
 */
export interface Ai4aiLineageRecord {
  readonly taskIdentity: string;
  readonly parentDigest: string;
  readonly mutation: Ai4aiMutation;
  readonly executorIdentity: string;
  /**
   * True only when the caller EXPLICITLY declared that `executorIdentity`
   * hashes a wrapper rather than the evaluator itself. Default false, so the
   * evaluator-identity binding below is enforced unless someone opts out on
   * the record, in the open, where an auditor sees it.
   */
  readonly wrapperIndirection: boolean;
  readonly evaluation: Ai4aiEvaluation;
  readonly digest: string;
}

/**
 * The root binding a lineage chains from: the frozen task identity mixed with
 * a per-run nonce. Without this a chain is fully deterministic — the same
 * mutations against the same task always produce byte-identical records, so an
 * old chain could be presented as a fresh run. The nonce makes each run's
 * chain unique and unforgeable-as-fresh; it is carried alongside the records
 * (see `Ai4aiLineage`) so verification can re-derive the root.
 */
export function ai4aiRunRoot(manifest: Ai4aiTaskManifest, runNonce: string): string {
  if (!HEX64.test(runNonce)) throw new Error("ai4ai run nonce must be 64 lowercase hex chars");
  return createHash("sha256")
    .update("ruvector.ai4ai.run-root.v1\0")
    .update(`${ai4aiTaskIdentity(manifest)}\0${runNonce}`)
    .digest("hex");
}

/** Fresh per-run nonce. */
export function newAi4aiRunNonce(): string {
  return randomBytes(32).toString("hex");
}

/** A complete run: the nonce its chain is bound to, plus the records. */
export interface Ai4aiLineage {
  readonly runNonce: string;
  readonly records: readonly Ai4aiLineageRecord[];
}

function lineageDigest(record: Omit<Ai4aiLineageRecord, "digest">): string {
  return createHash("sha256")
    .update("ruvector.ai4ai.lineage.v1\0")
    .update(canonical(record))
    .digest("hex");
}

// ---------------------------------------------------------------------------
// EXECUTOR SEAM — injected, never run in-repo (no GPU, no Docker here).
// ---------------------------------------------------------------------------

/**
 * The injected executor: given a pinned task and one mutation, run the real
 * AI4AI-Bench (arXiv:2608.20318) evaluator somewhere it can actually run and
 * return its RAW output plus a stable identity for the executor itself (so
 * lineage records name what produced each score). A crash must propagate —
 * the adapter re-throws with context and never scores a crash as a result.
 */
export type Ai4aiExecutor = (
  manifest: Ai4aiTaskManifest,
  mutation: Ai4aiMutation,
) => Promise<{
  readonly raw: unknown;
  readonly executorIdentity: string;
  /**
   * Set true ONLY when `executorIdentity` deliberately hashes a wrapper rather
   * than the pinned evaluator itself. Absent/false means the identity IS
   * claimed to be the evaluator's, and `runAi4aiMutation` then ENFORCES it
   * against `manifest.evaluatorSha256`.
   */
  readonly wrapperIndirection?: boolean;
}>;

export interface Ai4aiCommandExecutorOptions {
  /** Absolute path to the evaluator entrypoint to spawn. */
  readonly executorPath: string;
  /** Trusted argv inserted before the entrypoint (primarily test wrappers). */
  readonly commandPrefixArgs?: readonly string[];
  readonly timeoutMs?: number;
  readonly maxOutputBytes?: number;
  /**
   * Declare that `executorPath` is a WRAPPER around the pinned evaluator, not
   * the evaluator itself. Opting out of the evaluator-identity binding must be
   * explicit and visible: it is recorded on every lineage record
   * (`wrapperIndirection: true`) and committed to by the record digest, so an
   * auditor sees exactly which runs waived the check. Default false.
   */
  readonly wrapperIndirection?: boolean;
}

/**
 * A command-spawning executor with benchmark.ts's isolation discipline: a
 * scrubbed environment (PATH only), a wall timeout, an output ceiling, and a
 * temp workspace removed afterward. The entrypoint receives
 * `--task <manifest.json> --mutation <mutation.json> --output <result.json>`
 * and must write its evaluation JSON to the output path. Executor identity is
 * the SHA-256 of the entrypoint file's bytes, so lineage pins WHAT ran — and
 * the entrypoint is RE-HASHED after exit, failing loudly if its bytes changed
 * between hashing and execution (time-of-check/time-of-use), so the recorded
 * identity always names the code that actually ran.
 */
export function commandAi4aiExecutor(options: Ai4aiCommandExecutorOptions): Ai4aiExecutor {
  return async (manifest, mutation) => {
    const executorPath = resolve(options.executorPath);
    const executorIdentity = createHash("sha256").update(await readFile(executorPath)).digest("hex");
    const directory = await mkdtemp(join(tmpdir(), "ruvector-ai4ai-"));
    try {
      const taskFile = join(directory, "task.json");
      const mutationFile = join(directory, "mutation.json");
      const outputFile = join(directory, "result.json");
      await writeFile(taskFile, `${JSON.stringify(manifest)}\n`, { mode: 0o600 });
      await writeFile(mutationFile, `${JSON.stringify(mutation)}\n`, { mode: 0o600 });
      const prefix = options.commandPrefixArgs ?? [];
      const [command, ...leading] = prefix.length > 0 ? prefix : [executorPath];
      const args = [
        ...leading,
        ...(prefix.length > 0 ? [executorPath] : []),
        "--task", taskFile,
        "--mutation", mutationFile,
        "--output", outputFile,
      ];
      await new Promise<void>((resolveRun, reject) => {
        const child = spawn(command!, args, {
          stdio: ["ignore", "pipe", "pipe"],
          env: { PATH: process.env.PATH ?? "" },
        });
        let stderr = "";
        let outputBytes = 0;
        let limitFailure = "";
        const consume = (chunk: Buffer, capture: boolean) => {
          outputBytes += chunk.length;
          if (capture) stderr = `${stderr}${chunk.toString("utf8")}`.slice(-2_000);
          if (outputBytes > (options.maxOutputBytes ?? 10 * 1024 * 1024)) {
            limitFailure = "ai4ai executor output limit exceeded";
            child.kill("SIGKILL");
          }
        };
        child.stdout.on("data", (chunk: Buffer) => consume(chunk, false));
        child.stderr.on("data", (chunk: Buffer) => consume(chunk, true));
        const timer = setTimeout(() => {
          child.kill("SIGKILL");
          reject(new Error(`ai4ai executor exceeded ${options.timeoutMs ?? 300_000}ms`));
        }, options.timeoutMs ?? 300_000);
        child.once("error", (error) => {
          clearTimeout(timer);
          reject(error);
        });
        child.once("exit", (code) => {
          clearTimeout(timer);
          code === 0 && !limitFailure
            ? resolveRun()
            : reject(new Error(limitFailure || `ai4ai executor exited ${code}: ${stderr}`));
        });
      });
      // TOCTOU: the entrypoint could have been swapped between hashing and
      // spawning. Re-hash and refuse rather than record an identity for code
      // that did not run.
      const afterIdentity = createHash("sha256").update(await readFile(executorPath)).digest("hex");
      if (afterIdentity !== executorIdentity) {
        throw new Error(
          `ai4ai executor bytes changed during the run (${executorIdentity} -> ${afterIdentity})`,
        );
      }
      // The stdout/stderr ceiling does not cover the result file the evaluator
      // writes; bound it before reading so an oversized artifact cannot be
      // pulled into memory.
      const maxOutputBytes = options.maxOutputBytes ?? 10 * 1024 * 1024;
      const outputStat = await stat(outputFile);
      if (outputStat.size > maxOutputBytes) {
        throw new Error(
          `ai4ai executor result exceeds output limit: ${outputStat.size} > ${maxOutputBytes} bytes`,
        );
      }
      return {
        raw: JSON.parse(await readFile(outputFile, "utf8")) as unknown,
        executorIdentity,
        wrapperIndirection: options.wrapperIndirection ?? false,
      };
    } finally {
      await rm(directory, { recursive: true, force: true });
    }
  };
}

// ---------------------------------------------------------------------------
// RUNNING — one mutation, or a lineage chain of them.
// ---------------------------------------------------------------------------

/**
 * Evaluate one mutation against a pinned task. Validates the manifest and
 * mutation, invokes the injected executor, and parses its output fail-loud.
 * An executor crash propagates as an error naming the mutation — it is NEVER
 * folded into a score (adapter-crash ≠ result, same discipline as the WP16
 * subprocess binder).
 *
 * EVALUATOR-IDENTITY BINDING: `manifest.evaluatorSha256` is not merely
 * format-checked — unless the executor EXPLICITLY declares wrapper
 * indirection, the identity of what actually ran must EQUAL it. Otherwise a
 * lineage could claim evaluator X while running Y, which would make the
 * manifest's evaluator pin a false safety claim rather than a real binding.
 */
export async function runAi4aiMutation(
  manifest: Ai4aiTaskManifest,
  mutation: Ai4aiMutation,
  executor: Ai4aiExecutor,
  parentDigest: string,
): Promise<Ai4aiLineageRecord> {
  assertAi4aiTaskManifest(manifest);
  assertAi4aiMutation(mutation);
  if (!HEX64.test(parentDigest)) {
    throw new Error("ai4ai parent digest must be 64 lowercase hex chars");
  }
  let outcome: Awaited<ReturnType<Ai4aiExecutor>>;
  try {
    outcome = await executor(manifest, mutation);
  } catch (error) {
    throw new Error(
      `ai4ai executor failed for mutation ${mutation.id} (crash is not a score): ${
        error instanceof Error ? error.message : String(error)}`,
    );
  }
  if (!HEX64.test(outcome.executorIdentity)) {
    throw new Error("ai4ai executor must report a lowercase SHA-256 identity");
  }
  const wrapperIndirection = outcome.wrapperIndirection ?? false;
  if (!wrapperIndirection && outcome.executorIdentity !== manifest.evaluatorSha256) {
    throw new Error(
      `ai4ai executor identity does not match the manifest's pinned evaluator ` +
      `(${outcome.executorIdentity} != ${manifest.evaluatorSha256}); declare ` +
      `wrapperIndirection explicitly if the entrypoint wraps the evaluator`,
    );
  }
  const evaluation = parseAi4aiEvaluatorOutput(outcome.raw);
  const body = {
    taskIdentity: ai4aiTaskIdentity(manifest),
    parentDigest,
    mutation,
    executorIdentity: outcome.executorIdentity,
    wrapperIndirection,
    evaluation,
  };
  return Object.freeze({ ...body, digest: lineageDigest(body) });
}

/**
 * Evaluate a sequence of mutations as one lineage chain: the first mutation's
 * parent is the frozen task identity, each subsequent mutation's parent is
 * the previous record's digest. The whole chain is tamper-evident; a single
 * failing evaluation aborts the chain loudly rather than recording a hole.
 */
export async function runAi4aiLineage(
  manifest: Ai4aiTaskManifest,
  mutations: readonly Ai4aiMutation[],
  executor: Ai4aiExecutor,
  runNonce: string = newAi4aiRunNonce(),
): Promise<Ai4aiLineage> {
  assertAi4aiTaskManifest(manifest);
  if (mutations.length === 0) throw new Error("ai4ai lineage requires at least one mutation");
  const records: Ai4aiLineageRecord[] = [];
  let parentDigest = ai4aiRunRoot(manifest, runNonce);
  for (const mutation of mutations) {
    const record = await runAi4aiMutation(manifest, mutation, executor, parentDigest);
    records.push(record);
    parentDigest = record.digest;
  }
  return Object.freeze({ runNonce, records: Object.freeze(records) });
}

/**
 * Re-verify a lineage: every record must re-derive its own digest, name the
 * same frozen task, chain to its predecessor from the run root (task identity
 * + this run's nonce), and — unless it explicitly declares wrapper
 * indirection — have run the evaluator the manifest pins. Returns veto-style
 * reasons (empty ⇒ intact). This is what the veto provider below consumes.
 */
export function verifyAi4aiLineage(
  manifest: Ai4aiTaskManifest,
  lineage: Ai4aiLineage,
): string[] {
  const reasons: string[] = [];
  const { runNonce, records } = lineage;
  if (records.length === 0) {
    reasons.push("ai4ai_lineage_empty");
    return reasons;
  }
  if (!HEX64.test(runNonce)) {
    reasons.push("ai4ai_lineage_run_nonce_invalid");
    return [...new Set(reasons)].sort();
  }
  const taskIdentity = ai4aiTaskIdentity(manifest);
  let expectedParent = ai4aiRunRoot(manifest, runNonce);
  for (const record of records) {
    if (record.taskIdentity !== taskIdentity) reasons.push("ai4ai_lineage_task_mismatch");
    if (record.parentDigest !== expectedParent) reasons.push("ai4ai_lineage_broken_chain");
    const { digest, ...body } = record;
    if (lineageDigest(body) !== digest) reasons.push("ai4ai_lineage_digest_mismatch");
    if (!record.wrapperIndirection && record.executorIdentity !== manifest.evaluatorSha256) {
      reasons.push("ai4ai_evaluator_identity_mismatch");
    }
    expectedParent = record.digest;
  }
  return [...new Set(reasons)].sort();
}

// ---------------------------------------------------------------------------
// PROMOTION — our own paired statistics, never the paper's number.
// ---------------------------------------------------------------------------

/**
 * Compare baseline vs candidate score samples with the EXISTING paired-
 * bootstrap authority. Finiteness is re-checked at this choke point even
 * though ingest already enforced it (defense-in-depth: a hypothetical bypass
 * still cannot reach the gate with a NaN). The paper's 0.250 best-system
 * score is a TARGET for reporting, never an acceptance bar — only this
 * decision, recomputed on our own paired runs, counts toward promotion.
 */
export function ai4aiPairedDecision(
  baseline: readonly number[],
  candidate: readonly number[],
  minimumEffect = 0.005,
): PairedDecision {
  for (const value of [...baseline, ...candidate]) {
    if (!Number.isFinite(value)) {
      throw new Error("ai4ai paired decision refuses non-finite samples (fail-loud)");
    }
  }
  return pairedBootstrapDecision(baseline, candidate, minimumEffect);
}

/**
 * A vetoes.ts-composable provider that REJECTS promotion when the lineage
 * chain behind a candidate does not verify. Conjunctive like every provider:
 * it can only object, never rescue. `lineageFor` maps the veto context to the
 * manifest + records under evaluation, mirroring the `candidateFor` /
 * `environmentFor` seams of the dream-machine and ADR-324 providers.
 */
export function ai4aiLineageVetoProvider(
  lineageFor: (context: unknown) => {
    manifest: Ai4aiTaskManifest;
    lineage: Ai4aiLineage;
  } | Promise<{ manifest: Ai4aiTaskManifest; lineage: Ai4aiLineage }>,
): PromotionVetoProvider {
  return async (context) => {
    const { manifest, lineage } = await lineageFor(context);
    return verifyAi4aiLineage(manifest, lineage);
  };
}
