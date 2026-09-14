import test from "node:test";
import assert from "node:assert/strict";
import { Scheduler } from "../src/scheduler.js";

test("returns due work in deadline order", () => {
  let now = 1000;
  const scheduler = new Scheduler(() => now);
  scheduler.schedule("later", 20);
  scheduler.schedule("first", 10);
  now = 1020;
  assert.deepEqual(scheduler.due(), ["first", "later"]);
  assert.deepEqual(scheduler.due(), []);
});
