import test from "node:test";
import assert from "node:assert/strict";
import { pathToFileURL } from "node:url";
import path from "node:path";

const root = process.env.ADQ_TASK_ROOT;
if (!root) throw new Error("ADQ_TASK_ROOT is required");
const { normalizeRequest } = await import(pathToFileURL(path.join(root, "src/access.js")));

const defaults = {
  viewer: [" Public ", "reports", "public"],
  maintainer: ["Deploy", "audit"],
};

test("explicit empty resources override defaults", () => {
  assert.deepEqual(normalizeRequest({ role: "viewer", resources: [] }, defaults), {
    role: "viewer",
    resources: [],
  });
});

test("legacy operator role normalizes to maintainer", () => {
  assert.deepEqual(normalizeRequest({ role: "operator" }, defaults), {
    role: "maintainer",
    resources: ["audit", "deploy"],
  });
});

test("resource names are canonical and sorted", () => {
  assert.deepEqual(
    normalizeRequest({ role: "viewer", resources: [" Beta ", "alpha", "BETA"] }, defaults),
    { role: "viewer", resources: ["alpha", "beta"] },
  );
});

test("wildcard is exclusive", () => {
  assert.throws(
    () => normalizeRequest({ role: "viewer", resources: ["*", "reports"] }, defaults),
    /wildcard/i,
  );
  assert.deepEqual(normalizeRequest({ role: "viewer", resources: ["*"] }, defaults), {
    role: "viewer",
    resources: ["*"],
  });
});

test("unknown roles and malformed resources fail visibly", () => {
  assert.throws(() => normalizeRequest({ role: "owner" }, defaults), /unknown role.*owner/i);
  assert.throws(
    () => normalizeRequest({ role: "viewer", resources: ["ok", 7] }, defaults),
    /resource/i,
  );
});
