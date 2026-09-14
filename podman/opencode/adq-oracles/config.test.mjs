import test from "node:test";
import assert from "node:assert/strict";
import { pathToFileURL } from "node:url";
import path from "node:path";

const root = process.env.ADQ_TASK_ROOT;
const { cliConfig } = await import(pathToFileURL(path.join(root, "src/cli.js")));
const { httpConfig } = await import(pathToFileURL(path.join(root, "src/http.js")));

test("falsey values are preserved at every precedence layer", () => {
  const expected = { enabled: false, retries: 0, label: "" };
  assert.deepEqual(cliConfig({}, {}, expected), expected);
  assert.deepEqual(httpConfig({}, {}, expected), expected);
  assert.deepEqual(cliConfig({}, expected, {}), expected);
  assert.deepEqual(cliConfig(expected, {}, {}), expected);
});

test("unknown and invalid values fail", () => {
  for (const fn of [cliConfig, httpConfig]) {
    assert.throws(() => fn({}, {}, { surprise: 1 }), /unknown.*surprise/i);
    assert.throws(() => fn({}, {}, { retries: -1 }), /retries/i);
    assert.throws(() => fn({}, { enabled: "false" }, {}), /enabled/i);
  }
});
