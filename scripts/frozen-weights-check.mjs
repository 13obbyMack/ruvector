#!/usr/bin/env node
/**
 * frozen-weights-check.mjs — structural frozen-weights gate (PIR WP9, ADR-313, #841).
 *
 * Usage:
 *   node scripts/frozen-weights-check.mjs             # scan the mutation surfaces; exit 1 on any hit
 *   node scripts/frozen-weights-check.mjs --self-test # build an adversarial fixture tree in a temp
 *                                                     # dir (violations, comment-only mentions,
 *                                                     # symlinks) and assert the gate behaves
 *
 * ADR-313's central constraint is that foundation-model weights are frozen and
 * this is enforced STRUCTURALLY, not by policy: CI fails the build if any
 * mutation surface reachable from the promotion pipeline imports or invokes a
 * training / fine-tuning / weight-writing API, or references a model weight
 * file at all. This script is that CI check. The deny-list below is built from
 * what actually exists in this repo (grep of crates/ruvllm's training entry
 * points), not from hypothetical API names — every entry says why it is denied
 * and where the denied API lives.
 *
 * Security posture (style-matched to adr-index.mjs / workspace-check.mjs,
 * inheriting the PR #857 hardening lessons):
 *   - Symlinks are NEVER followed (file or directory) — skipped with a
 *     warning, so a committed symlink cannot smuggle content in or out of a
 *     mutation surface.
 *   - Every visited entry is belt-and-braces asserted (via realpath) to
 *     resolve inside its surface; anything that escapes is skipped.
 *   - Traversal is wrapped per-entry: broken symlinks, permission errors and
 *     cycles produce a warning + skip, never a stack-trace abort.
 *   - A MISSING mutation surface is a hard failure, not a skip — renaming a
 *     surface directory must not silently disable the gate.
 *
 * Known limitation (stated in ADR-313 Consequences): this is a static,
 * token-based check. It does not catch a sufficiently obfuscated or
 * dynamically-assembled fine-tuning path; it makes the honest path loud and
 * the dishonest path deliberate. Comments are stripped before matching so
 * that PROSE about training (e.g. darwin_guard.rs's train/eval-contamination
 * doc comments) does not false-positive — only code and string literals count.
 * The stripper is STRING-AWARE (PR #869 audit, MEDIUM finding): it is a
 * per-language character scan that tracks quote state, so a `//` inside a
 * string literal — "https://hf.co/repo/model.safetensors" — is content, not a
 * comment, and denied tokens in URLs are caught. It is also REGEX-AWARE (PR
 * #869 re-verify): a JS regex literal (`/…/`) has its interior preserved, so
 * a `//` or `[…]` inside a regex no longer eats a denied token later on the
 * line. Template literals are tracked including ${} nesting. The remaining
 * stripper edges are all in the false-POSITIVE (over-flag / safe) direction,
 * never hiding a denied token: a Rust char literal containing '"' or a raw
 * string r#"…"# may leave a comment unstripped, and a JS division mis-guessed
 * as a regex preserves its body as content. See stripComments() for the full
 * per-language contract and the residual-edge note.
 *
 * Accepted residuals (on record from the PR #869 audit, not closed here):
 *   - `fixtures/` and `node_modules/` under a surface stay unscanned by
 *     convention (committed `dist/` IS scanned — it is what executes).
 *   - shaperLoop.ts's capability gate fires on a SELF-DECLARED
 *     capabilityDelta until WP11 wires independent delta extraction from
 *     genome content (evaluation is still not promotion — human + vetoes own
 *     that).
 *   - FrozenModelRef.sha256 is record-only in this slice: witness-stamped as
 *     the claimed hash, but the day-0/day-30 re-hash that ENFORCES the freeze
 *     is WP12's operational harness.
 *
 * No dependencies beyond node >= 18.
 */

import { spawnSync } from 'node:child_process';
import {
  readdirSync, readFileSync, lstatSync, realpathSync,
  mkdirSync, mkdtempSync, rmSync, symlinkSync, copyFileSync, writeFileSync,
} from 'node:fs';
import { join, dirname, relative, sep } from 'node:path';
import { tmpdir } from 'node:os';
import { fileURLToPath } from 'node:url';

const SCRIPT_PATH = fileURLToPath(import.meta.url);
const REPO_ROOT = join(dirname(SCRIPT_PATH), '..');

/**
 * The evolution loop's mutation surfaces (ADR-313 §1, issue #841): the code
 * that Darwin/SHAPER generations are allowed to mutate or that orchestrates
 * mutation proposals reachable from the promotion pipeline. Only skills,
 * context and the execution harness evolve — so nothing under these paths may
 * touch a training API or a model weight file.
 */
const MUTATION_SURFACES = [
  // Darwin/GEPA/flywheel orchestration + dream-machine adapter + SHAPER loop.
  'crates/ruvector-sota-bench/harness/src',
  // scorePolicy mutation surface (issue #841 / ADR-313 map onto Darwin).
  'examples/mragent',
  // darwin_guard (ruvector ADR-271) — single-file surface.
  'crates/sona/src/darwin_guard.rs',
];

/** Source extensions worth scanning inside a surface. */
const SCAN_EXT = /\.(ts|mts|cts|js|mjs|cjs|rs|py|sh)$/;
/**
 * Directories that are never part of a surface's own source. `dist` is
 * deliberately NOT here (PR #869 audit): committed build output is what
 * actually executes, so it is scanned like source.
 */
const SKIP_DIRS = new Set(['node_modules', 'target', '.git', 'pkg', 'fixtures']);

/**
 * Deny-list. Each entry: a regex applied to comment-stripped source, why it is
 * denied, and where the denied API actually lives in this repo (the grep
 * anchor that put it on the list). Case-sensitive unless the API itself has
 * case variants in the tree.
 */
const DENY = [
  {
    id: 'ruvllm-training-module',
    re: /\bruvllm(::|\/)training\b/,
    why: 'gradient-descent fine-tuning module: RealTrainer::train(), GRPO, contrastive training',
    anchor: 'crates/ruvllm/src/training/{real_trainer,grpo,contrastive,mcp_tools}.rs',
  },
  {
    id: 'ruvllm-qat-module',
    re: /\bruvllm(::|\/)qat\b|\blora_qat\b|\btraining_loop\b/,
    why: 'quantization-aware training: weight updates via STE, LoRA-QAT',
    anchor: 'crates/ruvllm/src/qat/{training_loop,lora_qat}.rs',
  },
  {
    id: 'ruvllm-lora-module',
    re: /\bruvllm(::|\/)lora\b|\bmicro_?lora\b/i,
    why: 'LoRA adapter creation/training — adapters are weight deltas, out of bounds for a frozen-weights loop',
    anchor: 'crates/ruvllm/src/lora/{training.rs,adapters/trainer.rs,micro_lora.rs}',
  },
  {
    id: 'training-entry-points',
    re: /\btrain_step\b|\btrain_epoch\b|\btrain_on_trajectories\b|\btrain_buffered\b|\bRealTrainer\b/,
    why: 'concrete training entry-point invocations that update weights',
    anchor: 'grep "pub fn train*" over crates/ruvllm/src/{training,lora,qat}',
  },
  {
    id: 'generic-finetune',
    re: /\bfine[-_]?tune/i,
    why: 'any fine-tune token (fineTune / finetune / fine_tune) in code or strings, any language',
    anchor: 'ADR-313 Decision §2 — the acceptance test names fine-tuning APIs explicitly',
  },
  {
    id: 'pretrain-pipelines',
    re: /\bruvltra_pretrain\b|\bpretrain_pipeline\b/,
    why: 'pretraining pipelines — weight-producing by definition',
    anchor: 'crates/ruvllm/src/sona/ruvltra_pretrain.rs, crates/ruvllm/src/claude_flow/pretrain_pipeline.rs',
  },
  {
    id: 'weight-writing',
    re: /\bsave_checkpoint\b|\bsave_adapter\b|\bsave_weights\b|\bexport_weights\b/,
    why: 'weight/checkpoint writers — the loop must have no path that persists changed weights',
    anchor: 'crates/ruvllm/src/{qat/training_loop,lora/adapters/trainer,training/real_trainer}.rs',
  },
  {
    id: 'model-file-reference',
    re: /\.gguf\b|\.safetensors\b/i,
    why: 'model weight-file reference — mutation surfaces may not name model files at all (ADR-313: the weights path is simply absent)',
    anchor: 'GGUF/safetensors are the weight formats crates/ruvllm loads (crates/ruvllm/src/gguf)',
  },
  {
    id: 'mcp-weight-mutation-tools',
    re: /\bruvllm_microlora_(create|adapt)\b|\bruvllm_sona_(create|adapt)\b/,
    why: 'MCP tools that create/adapt LoRA/SONA weights from the TS side',
    anchor: 'claude-flow MCP surface (ruvllm_microlora_*, ruvllm_sona_*)',
  },
];

function warn(msg) {
  console.error(`frozen-weights-check: warning: ${msg}`);
}

/**
 * Strip comments so prose about training never false-positives; only code
 * and string literals are matched. STRING-AWARE (PR #869 audit): a
 * character scan tracks quote state per language, so `//` inside a string —
 * the URL form of a model-file or fine-tune reference — is content, never a
 * comment. Comment text is replaced with spaces (newlines kept).
 *
 * Per-language rules:
 *   - js-like (.ts/.mts/.cts/.js/.mjs/.cjs): `//` + `/* *​/` comments;
 *     `'`, `"`, and template-literal strings, with `${}` re-entry tracked
 *     (nested braces counted) and backslash escapes honored; and REGEX
 *     LITERALS (PR #869 re-verify): a `/` in value position opens a regex
 *     whose interior — including any `//` and `[...]` char classes — is
 *     content, never comment-stripped, so a denied token sharing a line with
 *     a regex is not silently dropped. `//` and `/* *​/` are still always
 *     comments (an empty `//` regex is illegal JS), so comment recognition
 *     wins first; only a single `/` not followed by `/` or `*` can open a
 *     regex. Value vs. operator (division) position is tracked by the last
 *     significant token; when uncertain the scan prefers the regex reading,
 *     which preserves content — the FALSE-POSITIVE (safe) direction.
 *   - rust (.rs): `//` + NESTED `/* *​/` comments; `"` strings only — `'` is
 *     deliberately not a string delimiter (lifetimes like `&'a str` would
 *     desynchronize the scan; char literals cannot hide a multi-char token).
 *   - python (.py): `#` comments; `'`/`"` and triple-quoted strings.
 *   - shell (.sh): `#` comments (only at line start or after whitespace, so
 *     `$#`/`${#x}` survive); `'` (no escapes) and `"` strings.
 *
 * Residual stripper edges, ALL in the false-POSITIVE (over-flag) direction —
 * they can leave a comment unstripped or preserve extra content, never hide a
 * denied token: a Rust char literal containing `"` or a raw string `r#"…"#`;
 * a js division mis-read as a regex when the value/operator heuristic guesses
 * wrong (preserves the "regex" body as content). The JS-regex FALSE-NEGATIVE
 * the PR #869 re-verify found (a regex's internal `//` eating a trailing
 * token) is closed by the regex-literal handling above.
 */
function stripComments(source, filename) {
  const lang = filename.endsWith('.rs') ? 'rs'
    : filename.endsWith('.py') ? 'py'
    : filename.endsWith('.sh') ? 'sh'
    : 'js';
  const slashComments = lang === 'js' || lang === 'rs';
  const hashComments = lang === 'py' || lang === 'sh';
  // Keywords after which a `/` still begins a VALUE (regex), not division.
  const VALUE_KEYWORDS = new Set([
    'return', 'typeof', 'instanceof', 'in', 'of', 'new', 'delete', 'void',
    'throw', 'else', 'do', 'yield', 'await', 'case',
  ]);
  const out = [];
  // For js template literals: one entry per open `${`, counting nested braces.
  const templateBraces = [];
  let mode = 'code'; // 'code' | 'line' | 'block' | 'string' | 'regex'
  let blockDepth = 0;
  let quote = ''; // ' " ` or ''' """ while mode === 'string'
  let regexInClass = false; // inside [...] while mode === 'regex'
  // js only: does the next `/` divide (true) or open a regex (false)?
  // Starts false — a `/` at the very start of a file opens a regex, not a
  // comment/division. Only a clear operand (identifier, number, `)`, `]`,
  // or a closed string/regex/template) sets it true.
  let expectOperand = false;
  let i = 0;
  while (i < source.length) {
    const ch = source[i];
    if (mode === 'line') {
      if (ch === '\n') { out.push('\n'); mode = 'code'; } else out.push(' ');
      i += 1; continue;
    }
    if (mode === 'block') {
      if (lang === 'rs' && source.startsWith('/*', i)) { blockDepth += 1; out.push('  '); i += 2; continue; }
      if (source.startsWith('*/', i)) {
        blockDepth -= 1; out.push('  '); i += 2;
        if (blockDepth === 0) mode = 'code';
        continue;
      }
      out.push(ch === '\n' ? '\n' : ' '); i += 1; continue;
    }
    if (mode === 'regex') {
      // Content-preserving: emit everything; a `//` or `[...]` here is regex
      // syntax, not a comment. Bail on a raw newline (unterminated regex).
      if (ch === '\\' && i + 1 < source.length) { out.push(source.slice(i, i + 2)); i += 2; continue; }
      if (ch === '\n') { out.push('\n'); mode = 'code'; expectOperand = false; i += 1; continue; }
      if (ch === '[') regexInClass = true;
      else if (ch === ']') regexInClass = false;
      else if (ch === '/' && !regexInClass) {
        out.push('/'); i += 1;
        while (i < source.length && /[a-z]/i.test(source[i])) { out.push(source[i]); i += 1; } // flags
        mode = 'code'; expectOperand = true; continue;
      }
      out.push(ch); i += 1; continue;
    }
    if (mode === 'string') {
      const noEscapes = quote.length === 3 || (lang === 'sh' && quote === "'");
      if (!noEscapes && ch === '\\') { out.push(source.slice(i, i + 2)); i += 2; continue; }
      if (quote === '`' && source.startsWith('${', i)) {
        templateBraces.push(0); mode = 'code'; expectOperand = false; out.push('${'); i += 2; continue;
      }
      if (quote.length === 3 ? source.startsWith(quote, i) : ch === quote) {
        out.push(quote); mode = 'code'; expectOperand = true; i += quote.length; continue;
      }
      out.push(ch); i += 1; continue;
    }
    // mode === 'code'
    if (templateBraces.length > 0 && (ch === '{' || ch === '}')) {
      const top = templateBraces.length - 1;
      if (ch === '{') { templateBraces[top] += 1; expectOperand = false; }
      else if (templateBraces[top] === 0) { templateBraces.pop(); mode = 'string'; quote = '`'; }
      else { templateBraces[top] -= 1; }
      out.push(ch); i += 1; continue;
    }
    if (slashComments && source.startsWith('//', i)) { mode = 'line'; out.push('  '); i += 2; continue; }
    if (slashComments && source.startsWith('/*', i)) { mode = 'block'; blockDepth = 1; out.push('  '); i += 2; continue; }
    // js regex-vs-division: a single `/` (not // or /*) in value position
    // opens a regex literal; in operator position it is division.
    if (lang === 'js' && ch === '/') {
      if (expectOperand) { out.push('/'); expectOperand = false; i += 1; continue; } // division
      mode = 'regex'; regexInClass = false; out.push('/'); i += 1; continue;
    }
    if (hashComments && ch === '#' && (lang === 'py' || i === 0 || /\s/.test(source[i - 1]))) {
      mode = 'line'; out.push(' '); i += 1; continue;
    }
    if (lang === 'py' && (source.startsWith('"""', i) || source.startsWith("'''", i))) {
      quote = source.slice(i, i + 3); mode = 'string'; out.push(quote); i += 3; continue;
    }
    if (ch === '"' || (ch === "'" && lang !== 'rs') || (ch === '`' && lang === 'js')) {
      quote = ch; mode = 'string'; out.push(ch); i += 1; continue;
    }
    // js value/operator tracking for the regex heuristic.
    if (lang === 'js') {
      if (/[A-Za-z0-9_$]/.test(ch)) {
        let j = i + 1;
        while (j < source.length && /[A-Za-z0-9_$]/.test(source[j])) j += 1;
        const word = source.slice(i, j);
        out.push(word);
        // A number or non-value-keyword identifier is an operand; a
        // value-keyword (return, typeof, …) still expects a value next.
        expectOperand = !VALUE_KEYWORDS.has(word);
        i = j; continue;
      }
      if (!/\s/.test(ch)) {
        // Punctuation: `)` and `]` close an operand; everything else
        // (`(`, `,`, `=`, `;`, `:`, `{`, operators …) expects a value next.
        expectOperand = ch === ')' || ch === ']';
      }
    }
    out.push(ch); i += 1;
  }
  return out.join('');
}

/**
 * Walk `dir` collecting scannable files. Symlinks are NEVER followed; every
 * kept entry must realpath-resolve under `rootReal`. Errors warn + skip.
 */
function walk(dir, rootReal, visited = new Set()) {
  const out = [];
  let dirReal;
  try {
    dirReal = realpathSync(dir);
  } catch (err) {
    warn(`cannot resolve directory ${dir}: ${err.message}`);
    return out;
  }
  if (visited.has(dirReal)) {
    warn(`directory cycle detected at ${dir} — skipping`);
    return out;
  }
  visited.add(dirReal);

  let dirents;
  try {
    dirents = readdirSync(dir, { withFileTypes: true });
  } catch (err) {
    warn(`cannot read directory ${dir}: ${err.message}`);
    return out;
  }
  for (const dirent of dirents) {
    const p = join(dir, dirent.name);
    if (dirent.isSymbolicLink()) {
      warn(`skipping symlink (not followed): ${p}`);
      continue;
    }
    if (dirent.isDirectory()) {
      if (!SKIP_DIRS.has(dirent.name)) out.push(...walk(p, rootReal, visited));
      continue;
    }
    if (!dirent.isFile() || !SCAN_EXT.test(dirent.name)) continue;
    try {
      const real = realpathSync(p);
      if (real !== rootReal && !real.startsWith(rootReal + sep)) {
        warn(`skipping entry that resolves outside its surface: ${p} -> ${real}`);
        continue;
      }
    } catch (err) {
      warn(`skipping unresolvable entry ${p}: ${err.message}`);
      continue;
    }
    out.push(p);
  }
  return out;
}

function surfaceFiles(surfaceRel, repoRoot) {
  const abs = join(repoRoot, surfaceRel);
  let st;
  try {
    st = lstatSync(abs);
  } catch {
    return { missing: true, files: [] };
  }
  if (st.isSymbolicLink()) {
    // A surface path replaced by a symlink is treated as missing: the gate
    // must not be redirected somewhere else.
    warn(`mutation surface is a symlink (not followed): ${surfaceRel}`);
    return { missing: true, files: [] };
  }
  if (st.isFile()) {
    return { missing: false, files: SCAN_EXT.test(abs) ? [abs] : [] };
  }
  let rootReal;
  try {
    rootReal = realpathSync(abs);
  } catch (err) {
    warn(`cannot resolve surface ${surfaceRel}: ${err.message}`);
    return { missing: true, files: [] };
  }
  return { missing: false, files: walk(abs, rootReal) };
}

function scanFile(abs, repoRoot) {
  const rel = relative(repoRoot, abs).split(sep).join('/');
  let source;
  try {
    source = readFileSync(abs, 'utf8');
  } catch (err) {
    warn(`cannot read ${rel}: ${err.message}`);
    return [];
  }
  const code = stripComments(source, abs);
  const hits = [];
  for (const entry of DENY) {
    const m = code.match(entry.re);
    if (m) hits.push({ rel, id: entry.id, token: m[0], why: entry.why, anchor: entry.anchor });
  }
  return hits;
}

function check(repoRoot = REPO_ROOT) {
  const missing = [];
  const violations = [];
  let scanned = 0;
  for (const surface of MUTATION_SURFACES) {
    const { missing: isMissing, files } = surfaceFiles(surface, repoRoot);
    if (isMissing) {
      missing.push(surface);
      continue;
    }
    for (const file of files) {
      scanned += 1;
      violations.push(...scanFile(file, repoRoot));
    }
  }

  if (missing.length > 0) {
    console.error(
      'frozen-weights-check: FAIL — missing mutation surface(s) (renaming a ' +
      'surface directory must not silently disable this gate):\n');
    for (const m of missing) console.error(`  ${m}`);
    console.error(
      '\nFix: restore the path, or update MUTATION_SURFACES in ' +
      'scripts/frozen-weights-check.mjs in the same commit that moves it.');
  }
  if (violations.length > 0) {
    console.error(
      `frozen-weights-check: FAIL — ${violations.length} training/weight-API ` +
      'reference(s) inside frozen-weights mutation surfaces (ADR-313):\n');
    for (const v of violations) {
      console.error(`  ${v.rel}: [${v.id}] matched "${v.token}"`);
      console.error(`    why denied: ${v.why}`);
      console.error(`    denied API lives at: ${v.anchor}`);
    }
    console.error(
      '\nThe SHAPER loop evolves skills, context and harness ONLY. If this hit ' +
      'is a false positive, narrow the deny-list entry in ' +
      'scripts/frozen-weights-check.mjs with a reviewable justification — do ' +
      'not move code out of the surface to dodge the gate.');
  }
  if (missing.length > 0 || violations.length > 0) process.exit(1);
  console.log(
    `frozen-weights-check: OK — ${scanned} files scanned across ` +
    `${MUTATION_SURFACES.length} mutation surfaces, ${DENY.length} deny-list ` +
    'entries, 0 training/weight-API references.');
}

/**
 * --self-test: build a fixture tree in a temp dir, copy this script into it
 * (REPO_ROOT derives from the script location, so the copy operates on the
 * fixture), and assert:
 *   1. a clean tree passes;
 *   2. training-import / MCP-tool / model-file violations each fail with the
 *      right deny-list id — including URL-form references
 *      ("https://…/model.safetensors", "file://…/x.gguf", "http://…/finetune",
 *      the PR #869 MEDIUM), regex literals whose internal `//` must not eat a
 *      trailing denied token (PR #869 re-verify PoCs), the single-slash
 *      "ruvllm/training" import form, .py/.sh helpers, and files under a
 *      committed dist/;
 *   3. comment-only mentions of denied tokens do NOT fail (plain comments,
 *      comments containing URLs, division-then-comment, and
 *      regex-then-comment lines alike);
 *   4. symlinks (including one pointing at an out-of-tree file full of
 *      violations) are skipped with a warning, never followed, no stack trace;
 *   5. a deleted mutation surface fails loudly (missing-surface hardening).
 */
function selfTest() {
  const failures = [];
  const ok = (cond, label) => {
    if (cond) console.log(`  PASS  ${label}`);
    else { console.error(`  FAIL  ${label}`); failures.push(label); }
  };

  const root = mkdtempSync(join(tmpdir(), 'frozen-weights-selftest-'));
  try {
    const harnessSrc = join(root, 'crates', 'ruvector-sota-bench', 'harness', 'src');
    const mragent = join(root, 'examples', 'mragent');
    const sonaSrc = join(root, 'crates', 'sona', 'src');
    for (const d of [harnessSrc, mragent, sonaSrc, join(root, 'scripts')]) {
      mkdirSync(d, { recursive: true });
    }
    copyFileSync(SCRIPT_PATH, join(root, 'scripts', 'frozen-weights-check.mjs'));
    writeFileSync(join(sonaSrc, 'darwin_guard.rs'),
      '// contamination guard: strict train/eval instance-ID separation\n' +
      'pub fn assert_train_eval_disjoint(train_ids: &[&str]) {}\n');
    writeFileSync(join(harnessSrc, 'clean.ts'), 'export const ok = 1;\n');
    writeFileSync(join(mragent, 'scorePolicy.mjs'), 'export const scorePolicy = () => 0;\n');

    const script = join(root, 'scripts', 'frozen-weights-check.mjs');
    const run = () => {
      const r = spawnSync(process.execPath, [script], { encoding: 'utf8', timeout: 30_000 });
      return { code: r.status ?? 1, stdout: r.stdout ?? '', stderr: r.stderr ?? '' };
    };

    // --- 1. clean tree passes; prose about training does not trip ---
    const clean = run();
    ok(clean.code === 0, 'clean fixture tree passes');
    ok(clean.stdout.includes('frozen-weights-check: OK'), 'clean run prints OK summary');

    // --- 3 + 4. comment-only mentions and symlinks are safe ---
    writeFileSync(join(harnessSrc, 'commented.ts'),
      '// mentions fine_tune only in a comment\n/* fineTune save_checkpoint */\nexport const x = 1;\n');
    // Comment containing a URL with a denied token: still a comment, still safe.
    writeFileSync(join(harnessSrc, 'comment-url.ts'),
      '// background reading: http://internal/finetune docs\nexport const y = 2;\n');
    // True division then a real trailing comment: the comment (with its denied
    // token) must still strip — division must not derail comment recognition.
    writeFileSync(join(harnessSrc, 'division-comment.ts'),
      'const q = a / b / c; // note: models/x.gguf\nexport {q};\n');
    // Regex literal then a real trailing comment: the comment's denied token
    // must NOT be flagged (the regex closes; `//` after it is a comment).
    writeFileSync(join(harnessSrc, 'regex-comment.ts'),
      'const re = /ab+c/; // mentions model.safetensors in prose\nexport {re};\n');
    writeFileSync(join(mragent, 'clean.sh'),
      '#!/bin/sh\n# fine_tune mentioned only in this comment\necho ok\n');
    writeFileSync(join(root, 'outside-violations.rs'),
      'use ruvllm::training::RealTrainer; // finetune everything\n');
    symlinkSync(join(root, 'outside-violations.rs'), join(harnessSrc, 'linked.rs'));
    symlinkSync('/nonexistent/target.rs', join(harnessSrc, 'broken.rs'));
    symlinkSync('.', join(harnessSrc, 'loop'));
    const benign = run();
    ok(benign.code === 0, 'comment-only mentions + symlinks still pass');
    ok(benign.stderr.includes('skipping symlink'), 'symlinks produce a skip warning');
    ok(!(benign.stdout + benign.stderr).includes('at walk'),
      'no stack trace on broken symlink / symlink loop');

    // --- 2. real violations fail with the right deny-list ids ---
    writeFileSync(join(harnessSrc, 'bad-train.rs'), 'use ruvllm::training::RealTrainer;\n');
    writeFileSync(join(mragent, 'bad-tool.ts'),
      'await call("ruvllm_microlora_adapt", {});\n');
    writeFileSync(join(harnessSrc, 'bad-model.ts'),
      'const weights = "models/llama.gguf";\n');
    // PR #869 MEDIUM: URL-form references — `//` inside a string is content.
    writeFileSync(join(harnessSrc, 'bad-url-hf.ts'),
      'const w = "https://hf.co/repo/model.safetensors";\n');
    writeFileSync(join(harnessSrc, 'bad-url-gguf.ts'),
      'loadWeights("file://models/x.gguf");\n');
    writeFileSync(join(harnessSrc, 'bad-url-finetune.ts'),
      'const u = "http://internal/finetune";\n');
    // Single-slash module path must keep flagging (auditor's regression case).
    writeFileSync(join(harnessSrc, 'bad-import-slash.ts'),
      'import { t } from "ruvllm/training";\n');
    // PR #869 re-verify: a `//` inside a JS regex must NOT eat the trailing
    // denied token on the same line. Both PoCs are live JS.
    writeFileSync(join(harnessSrc, 'bad-regex-class.ts'),
      'const re = /[a//]/; const w = "evil-model.gguf"; export {re, w};\n');
    writeFileSync(join(harnessSrc, 'bad-regex-escaped.ts'),
      'const re = /https:\\/\\//; loadWeights("m.safetensors"); export {re};\n');
    // Ordinary-code form: regex guard then a real model load on the same line.
    writeFileSync(join(harnessSrc, 'bad-regex-guard.ts'),
      'if (/https?:\\/\\//.test(u)) loadModel("x.gguf");\n');
    // New coverage: .py / .sh helpers and committed dist/ output are scanned.
    writeFileSync(join(mragent, 'bad.py'),
      'trainer.save_checkpoint("out")  # totally routine\n');
    writeFileSync(join(mragent, 'bad.sh'),
      '#!/bin/sh\npython finetune.py --epochs 3\n');
    mkdirSync(join(harnessSrc, 'dist'), { recursive: true });
    writeFileSync(join(harnessSrc, 'dist', 'bad-dist.mjs'), 'save_weights(model);\n');
    const dirty = run();
    ok(dirty.code === 1, 'violations fail the gate (exit 1)');
    ok(dirty.stderr.includes('bad-train.rs') && dirty.stderr.includes('ruvllm-training-module'),
      'training-module import is flagged with its deny-list id');
    ok(dirty.stderr.includes('bad-tool.ts') && dirty.stderr.includes('mcp-weight-mutation-tools'),
      'MCP weight-mutation tool call is flagged');
    ok(dirty.stderr.includes('bad-model.ts') && dirty.stderr.includes('model-file-reference'),
      'model weight-file reference is flagged');
    ok(dirty.stderr.includes('bad-url-hf.ts') && dirty.stderr.includes('model-file-reference'),
      'https URL to .safetensors is flagged (string-aware stripper)');
    ok(dirty.stderr.includes('bad-url-gguf.ts'),
      'file:// URL to .gguf is flagged (string-aware stripper)');
    ok(dirty.stderr.includes('bad-url-finetune.ts') && dirty.stderr.includes('generic-finetune'),
      'http URL to /finetune is flagged (string-aware stripper)');
    ok(dirty.stderr.includes('bad-import-slash.ts'),
      'single-slash ruvllm/training import still flags');
    ok(dirty.stderr.includes('bad-regex-class.ts') && dirty.stderr.includes('model-file-reference'),
      'regex with // in a char class does NOT eat the trailing .gguf (PoC a)');
    ok(dirty.stderr.includes('bad-regex-escaped.ts'),
      'regex with escaped // does NOT eat the trailing .safetensors (PoC b)');
    ok(dirty.stderr.includes('bad-regex-guard.ts'),
      'regex guard then loadModel(".gguf") on one line is flagged (ordinary code)');
    ok(dirty.stderr.includes('bad.py') && dirty.stderr.includes('weight-writing'),
      '.py helper invoking a weight writer is flagged');
    ok(dirty.stderr.includes('bad.sh') && dirty.stderr.includes('generic-finetune'),
      '.sh helper invoking finetune is flagged');
    ok(dirty.stderr.includes('dist/bad-dist.mjs'),
      'committed dist/ output is scanned and flagged');
    ok(!dirty.stderr.includes('commented.ts'), 'comment-only file is NOT flagged');
    ok(!dirty.stderr.includes('comment-url.ts'),
      'denied token in a genuine comment URL is NOT flagged');
    ok(!dirty.stderr.includes('clean.sh'),
      '.sh with denied token only in a comment is NOT flagged');
    ok(!dirty.stderr.includes('division-comment.ts'),
      'division then a real comment: comment token still stripped (not flagged)');
    ok(!dirty.stderr.includes('regex-comment.ts'),
      'regex then a real comment: comment token NOT flagged');
    ok(!dirty.stderr.includes('linked.rs: ['), 'symlinked violations are NOT followed/flagged');
    ok(!dirty.stderr.includes('darwin_guard.rs'),
      'train/eval-contamination guard prose does NOT false-positive');

    // --- 5. missing-surface hardening ---
    rmSync(mragent, { recursive: true, force: true });
    const missing = run();
    ok(missing.code === 1, 'deleted mutation surface fails the gate');
    ok(missing.stderr.includes('missing mutation surface'), 'missing surface is reported as such');
  } finally {
    rmSync(root, { recursive: true, force: true });
  }

  if (failures.length > 0) {
    console.error(`\nself-test FAILED: ${failures.length} assertion(s)`);
    process.exit(1);
  }
  console.log('\nself-test OK: all assertions passed');
}

try {
  if (process.argv.includes('--self-test')) selfTest();
  else check();
} catch (err) {
  console.error(`frozen-weights-check: fatal: ${err.message}`);
  process.exit(1);
}
