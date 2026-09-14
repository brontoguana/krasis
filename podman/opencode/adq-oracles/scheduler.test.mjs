import test from "node:test";
import assert from "node:assert/strict";
import { pathToFileURL } from "node:url";
import path from "node:path";

const root = process.env.ADQ_TASK_ROOT;
const { Scheduler } = await import(pathToFileURL(path.join(root, "src/scheduler.js")));

test("replacement suppresses stale deadline", () => {
  let now = 1000;
  const scheduler = new Scheduler(() => now);
  scheduler.schedule("invoice-42", 5000);
  scheduler.schedule("invoice-42", 2000);
  now = 3000;
  assert.deepEqual(scheduler.due(), ["invoice-42"]);
  now = 6000;
  assert.deepEqual(scheduler.due(), []);
});

test("equal deadlines are stable by ID after replacement", () => {
  const scheduler = new Scheduler(() => 100);
  scheduler.schedule("z", 10);
  scheduler.schedule("a", 10);
  scheduler.schedule("z", 10);
  assert.deepEqual(scheduler.due(110), ["a", "z"]);
});
