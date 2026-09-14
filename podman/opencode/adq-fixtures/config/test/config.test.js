import test from "node:test";
import assert from "node:assert/strict";
import { cliConfig } from "../src/cli.js";
import { httpConfig } from "../src/http.js";

test("higher layers win", () => {
  const file = { retries: 1 };
  const env = { retries: 2 };
  const override = { retries: 4 };
  assert.equal(cliConfig(file, env, override).retries, 4);
  assert.deepEqual(cliConfig(file, env, override), httpConfig(file, env, override));
});
