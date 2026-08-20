#!/usr/bin/env node
/**
 * adr-index.mjs — canonical ADR index generator + duplicate-number gate.
 *
 * Usage:
 *   node scripts/adr-index.mjs           # regenerate docs/adr/INDEX.md
 *   node scripts/adr-index.mjs --check   # exit 1 if a NEW duplicate ADR number
 *                                        # exists (one not in the frozen
 *                                        # historical list below); exit 0 otherwise
 *
 * Policy: see docs/adr/ADR-316-adr-numbering-hygiene.md.
 * Duplicate numbers that existed before ADR-316 are FROZEN historical
 * artifacts — they are never renamed, and are disambiguated in prose as
 * "ADR-NNN (slug)". This script's --check mode is the CI gate that keeps
 * the frozen list from growing.
 *
 * No dependencies beyond node >= 18 and git.
 */

import { execFileSync } from 'node:child_process';
import { readdirSync, readFileSync, writeFileSync, statSync } from 'node:fs';
import { join, relative, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const REPO_ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const ADR_DIR = join(REPO_ROOT, 'docs', 'adr');
const INDEX_PATH = join(ADR_DIR, 'INDEX.md');

/**
 * FROZEN historical duplicate list (verified against origin/main, 2026-08-19,
 * at the time ADR-316 was written). Key = plain ADR number (no namespace,
 * no a/b sub-ADR suffix), value = number of files historically sharing it.
 * Do NOT add entries here — new collisions must be renumbered before merge.
 */
const FROZEN_DUPLICATES = {
  272: 5,
  264: 3, 252: 3, 194: 3, 144: 3,
  268: 2, 266: 2, 260: 2, 258: 2, 256: 2, 254: 2,
  143: 2, 139: 2, 138: 2, 137: 2, 136: 2, 135: 2,
  134: 2, 133: 2, 132: 2, 117: 2,
  96: 2, 95: 2, 94: 2, 93: 2, 91: 2, 90: 2,
};

// Filename shapes handled:
//   ADR-016-slug.md                (plain, canonical counter)
//   ADR-040a-slug.md               (sub-ADR convention — distinct from ADR-040)
//   ADR-DB-001-slug.md             (namespaced family: DB, CE, QE, TTS, ...)
const FILE_RE = /^ADR-(?:([A-Z]{2,4})-)?(\d+)([a-z])?(?:[-_](.+?))?\.md$/;

function walk(dir) {
  const out = [];
  for (const entry of readdirSync(dir)) {
    const p = join(dir, entry);
    if (statSync(p).isDirectory()) out.push(...walk(p));
    else if (entry.endsWith('.md')) out.push(p);
  }
  return out;
}

function parseTitle(content, fallback) {
  const m = content.match(/^#\s+(.+?)\s*$/m);
  return m ? m[1].trim() : fallback;
}

function parseStatus(content) {
  // Frontmatter: status: Accepted
  let m = content.match(/^status:\s*(.+?)\s*$/im);
  if (m && content.slice(0, 4) === '---\n') return m[1];
  // Inline: **Status**: Proposed  /  **Status:** Proposed  /  - Status: Proposed
  m = content.match(/^[-*\s]*\*\*Status:?\*\*:?\s*(.+?)\s*$/im) ||
      content.match(/^[-*]\s*Status:\s*(.+?)\s*$/im);
  if (m) return m[1].replace(/\*+/g, '').trim();
  // Section: ## Status \n <first non-empty line>
  m = content.match(/^##\s*Status\s*\n+([^\n#]+)/im);
  if (m) return m[1].trim();
  return '';
}

function gitDate(relPath) {
  try {
    const out = execFileSync(
      'git', ['log', '--follow', '-1', '--format=%ci', '--', relPath],
      { cwd: REPO_ROOT, encoding: 'utf8' },
    ).trim();
    return out ? out.slice(0, 10) : '';
  } catch {
    return '';
  }
}

function collect({ withDates }) {
  const entries = [];
  for (const abs of walk(ADR_DIR)) {
    const rel = relative(REPO_ROOT, abs);
    const base = abs.split('/').pop();
    if (base === 'README.md' || base === 'INDEX.md') continue;
    const m = base.match(FILE_RE);
    if (!m) continue; // non-ADR markdown (design notes etc.)
    const [, ns, numStr, subSuffix, slug] = m;
    const num = parseInt(numStr, 10);
    let title = '', status = '';
    try {
      const content = readFileSync(abs, 'utf8');
      title = parseTitle(content, slug ?? base);
      status = parseStatus(content);
    } catch { /* unreadable file — keep filename-derived fields */ }
    entries.push({
      ns: ns ?? '',
      num,
      subSuffix: subSuffix ?? '',
      slug: slug ?? '',
      rel,
      title,
      status,
      date: withDates ? gitDate(rel) : '',
      // Duplicate-detection key. The a/b sub-ADR suffix makes a DISTINCT
      // key on purpose: ADR-040 / ADR-040a / ADR-040b never collide.
      key: `${ns ? ns + ':' : ''}${num}${subSuffix ?? ''}`,
    });
  }
  entries.sort((a, b) =>
    a.ns.localeCompare(b.ns) || a.num - b.num ||
    a.subSuffix.localeCompare(b.subSuffix) || a.rel.localeCompare(b.rel));
  return entries;
}

function groupDuplicates(entries) {
  const byKey = new Map();
  for (const e of entries) {
    if (!byKey.has(e.key)) byKey.set(e.key, []);
    byKey.get(e.key).push(e);
  }
  return byKey;
}

function check() {
  const entries = collect({ withDates: false });
  const byKey = groupDuplicates(entries);
  const violations = [];
  for (const [key, group] of byKey) {
    if (group.length < 2) continue;
    const e = group[0];
    const frozen = e.ns === '' && e.subSuffix === '' ? FROZEN_DUPLICATES[e.num] : undefined;
    if (frozen === undefined) {
      violations.push(`NEW duplicate ${key}: ${group.map(g => g.rel).join(', ')}`);
    } else if (group.length > frozen) {
      violations.push(
        `Duplicate count for ADR-${e.num} grew: ${group.length} files ` +
        `(frozen historical allowance is ${frozen}): ${group.map(g => g.rel).join(', ')}`);
    }
  }
  if (violations.length > 0) {
    console.error('ADR duplicate-number check FAILED:\n');
    for (const v of violations) console.error(`  - ${v}`);
    console.error('\nPick the next available number from docs/adr/INDEX.md ' +
      '(regenerate with `node scripts/adr-index.mjs`). Never reuse a number.');
    process.exit(1);
  }
  const maxNum = Math.max(...entries.filter(e => !e.ns).map(e => e.num));
  console.log(`ADR duplicate-number check OK: ${entries.length} ADR files, ` +
    `no duplicates outside the frozen historical list. Next available number: ${maxNum + 1}.`);
}

function mdCell(s) {
  return String(s ?? '').replace(/\|/g, '\\|').replace(/\s+/g, ' ').trim();
}

function generate() {
  const entries = collect({ withDates: true });
  const byKey = groupDuplicates(entries);
  const plain = entries.filter(e => !e.ns);
  const maxNum = Math.max(...plain.map(e => e.num));
  const next = maxNum + 1;
  const dupKeys = [...byKey.values()].filter(g => g.length > 1);
  const dupFileCount = dupKeys.reduce((n, g) => n + g.length, 0);

  const lines = [];
  lines.push('# ADR Index');
  lines.push('');
  lines.push(`**Next available ADR number: ${next}**`);
  lines.push('');
  lines.push('> Generated by `node scripts/adr-index.mjs` — do not edit by hand.');
  lines.push('> This file is the canonical allocation counter for new ADR numbers');
  lines.push('> (policy: `ADR-316`). Duplicate numbers listed below are frozen');
  lines.push('> historical artifacts and are cited as `ADR-NNN (slug)`.');
  lines.push('> CI gate: `node scripts/adr-index.mjs --check`.');
  lines.push('');
  lines.push(`- ADR files indexed: **${entries.length}** ` +
    `(${plain.length} on the canonical counter, ${entries.length - plain.length} in namespaced families)`);
  lines.push(`- Highest allocated number: **ADR-${maxNum}**`);
  lines.push(`- Frozen duplicate numbers: **${dupKeys.length}** ` +
    `(spanning ${dupFileCount} files)`);
  lines.push('');
  lines.push('| Number | Title | File | Last commit | Status | Duplicate |');
  lines.push('|---|---|---|---|---|---|');
  for (const e of entries) {
    const numLabel = `ADR-${e.ns ? e.ns + '-' : ''}${String(e.num).padStart(3, '0')}${e.subSuffix}`;
    const group = byKey.get(e.key);
    const dup = group.length > 1
      ? `DUPLICATE ×${group.length} — cite as \`ADR-${e.num} (${e.slug || 'untitled'})\``
      : '';
    lines.push(`| ${numLabel} | ${mdCell(e.title)} | [\`${e.rel.replace('docs/adr/', '')}\`](${e.rel.replace('docs/adr/', './')}) | ${e.date} | ${mdCell(e.status).slice(0, 80)} | ${dup} |`);
  }
  lines.push('');
  writeFileSync(INDEX_PATH, lines.join('\n'));
  console.log(`Wrote ${relative(REPO_ROOT, INDEX_PATH)}: ${entries.length} entries, ` +
    `${dupKeys.length} frozen duplicate numbers, next available ADR-${next}.`);
}

if (process.argv.includes('--check')) check();
else generate();
