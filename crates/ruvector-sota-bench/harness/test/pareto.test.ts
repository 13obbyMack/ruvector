import assert from "node:assert/strict";
import test from "node:test";
import {
  admitToFrontier,
  dominates,
  frontier,
  DEFAULT_OBJECTIVES,
  type ParetoPoint,
} from "../src/pareto.js";

const point = (id: string, primary: number, costPerWin: number, p99Us: number): ParetoPoint => ({
  id, values: { primary, costPerWin, p99Us },
});

test("dominance requires no worse on every objective and better on one", () => {
  const strong = point("strong", 0.9, 10, 100);
  const weak = point("weak", 0.8, 20, 200);
  assert.equal(dominates(strong, weak), true);
  assert.equal(dominates(weak, strong), false);
});

test("a scalar win bought with worse cost and latency does not dominate", () => {
  // The case the gate exists for: higher primary, but strictly worse on both
  // resource axes. That is a trade, not a win, so neither point dominates.
  const traded = point("traded", 0.95, 40, 400);
  const incumbent = point("incumbent", 0.90, 10, 100);
  assert.equal(dominates(traded, incumbent), false);
  assert.equal(dominates(incumbent, traded), false);
});

test("identical points do not dominate each other and both stay on the frontier", () => {
  const a = point("a", 0.9, 10, 100);
  const b = point("b", 0.9, 10, 100);
  assert.equal(dominates(a, b), false);
  assert.equal(dominates(b, a), false);
  assert.deepEqual(frontier([a, b]).map((entry) => entry.id), ["a", "b"]);
});

test("frontier keeps the non-dominated set in input order", () => {
  const points = [
    point("dominated", 0.5, 50, 500),
    point("fast", 0.8, 30, 50),
    point("accurate", 0.95, 60, 300),
    point("cheap", 0.7, 5, 400),
  ];
  assert.deepEqual(frontier(points).map((entry) => entry.id), ["fast", "accurate", "cheap"]);
});

test("non-finite objectives are refused rather than compared", () => {
  // NaN defeats both `<` and `>`, so a NaN candidate would read as
  // non-dominated against everything. Refuse it at the choke point instead.
  const sane = point("sane", 0.9, 10, 100);
  for (const bad of [Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY]) {
    const broken = point("broken", bad, 10, 100);
    assert.throws(() => dominates(broken, sane), /non-finite/);
    assert.throws(() => dominates(sane, broken), /non-finite/);
    assert.throws(() => frontier([sane, broken]), /non-finite/);
    assert.throws(() => admitToFrontier(broken, [sane]), /non-finite/);
  }
});

test("a missing objective is refused rather than treated as zero", () => {
  const partial: ParetoPoint = { id: "partial", values: { primary: 0.9 } };
  assert.throws(() => dominates(partial, point("other", 0.8, 10, 100)), /missing objective/);
});

test("admission blocks a dominated candidate and names the dominators", () => {
  const current = [point("incumbent", 0.9, 10, 100), point("other", 0.85, 8, 90)];
  const worse = point("worse", 0.5, 50, 500);
  const admission = admitToFrontier(worse, current);
  assert.equal(admission.admitted, false);
  assert.deepEqual(admission.dominatedBy, ["incumbent", "other"]);
});

test("an empty frontier admits, because absence of evidence is not domination", () => {
  assert.deepEqual(admitToFrontier(point("first", 0.1, 999, 999), []), {
    admitted: true, dominatedBy: [],
  });
});

test("objective specs are validated before any comparison", () => {
  const a = point("a", 0.9, 10, 100);
  const b = point("b", 0.8, 20, 200);
  assert.throws(() => dominates(a, b, []), /at least one objective/);
  assert.throws(
    () => dominates(a, b, [{ name: "primary", direction: "sideways" as never }]),
    /maximize or minimize/,
  );
  assert.throws(
    () => dominates(a, b, [...DEFAULT_OBJECTIVES, { name: "primary", direction: "maximize" }]),
    /duplicate/,
  );
});
