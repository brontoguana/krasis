import test from "node:test";
import assert from "node:assert/strict";
import { reconcile } from "../src/ledger.js";

const lines = (rows) => rows.map(JSON.stringify).join("\n");

test("sums ordinary events and sorts accounts", () => {
  const input = lines([
    { id: "a", account: "zeta", amount: 2 },
    { id: "b", account: "alpha", amount: "3.5" },
  ]);
  assert.deepEqual(reconcile(input), { alpha: 3.5, zeta: 2 });
});

test("uses normalized declared fields for duplicate identity", () => {
  const equivalentRawRecords = [
    '{"id":"a","account":"cash","amount":"2","note":"first"}',
    '{ "amount": 2.0, "account": "cash", "id": "a", "ignored": true }',
  ].join("\n");
  assert.deepEqual(reconcile(equivalentRawRecords), { cash: 2 });

  assert.throws(
    () => reconcile(lines([
      { id: "a", account: "cash", amount: 2 },
      { id: "a", account: "cash", amount: 3 },
    ])),
    /conflicting duplicate.*a/i,
  );
});
