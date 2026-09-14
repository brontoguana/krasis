import test from "node:test";
import assert from "node:assert/strict";
import { pathToFileURL } from "node:url";
import path from "node:path";

const root = process.env.ADQ_TASK_ROOT;
const { reconcile } = await import(pathToFileURL(path.join(root, "src/ledger.js")));
const lines = (rows) => rows.map(JSON.stringify).join("\n");

test("out-of-order correction chains replace complete effects", () => {
  const input = lines([
    { id: "c", account: "cash", amount: 7, supersedes: "b" },
    { id: "other", account: "alpha", amount: 4 },
    { id: "a", account: "cash", amount: 10 },
    { id: "b", account: "cash", amount: 8, supersedes: "a" },
  ]);
  assert.deepEqual(reconcile(input), { alpha: 4, cash: 7 });
});

test("normalized semantic duplicates count once and declared conflicts fail", () => {
  const equivalentRawRecords = [
    '{"id":"a","account":"cash","amount":"2","note":"first"}',
    '{ "amount": 2.0, "account": "cash", "id": "a", "ignored": true }',
  ].join("\n");
  assert.deepEqual(reconcile(equivalentRawRecords), { cash: 2 });
  assert.throws(() => reconcile(lines([
    { id: "a", account: "cash", amount: 2 },
    { id: "a", account: "cash", amount: 3 },
  ])), /conflicting duplicate.*a/i);
  assert.throws(() => reconcile(lines([
    { id: "base", account: "cash", amount: 1 },
    { id: "a", account: "cash", amount: 2, supersedes: "base" },
    { id: "a", account: "cash", amount: 2 },
  ])), /conflicting duplicate.*a/i);
});

test("rejects missing targets, cycles, and account changes", () => {
  assert.throws(() => reconcile(lines([{ id: "b", account: "cash", amount: 2, supersedes: "missing" }])), /missing.*missing/i);
  assert.throws(() => reconcile(lines([
    { id: "a", account: "cash", amount: 1, supersedes: "b" },
    { id: "b", account: "cash", amount: 2, supersedes: "a" },
  ])), /cycle/i);
  assert.throws(() => reconcile(lines([
    { id: "a", account: "cash", amount: 1 },
    { id: "b", account: "other", amount: 2, supersedes: "a" },
  ])), /account/i);
});
