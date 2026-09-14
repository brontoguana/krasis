import test from "node:test";
import assert from "node:assert/strict";
import { normalizeRequest } from "../src/access.js";

test("returns an ordinary explicit request", () => {
  assert.deepEqual(
    normalizeRequest({ role: "viewer", resources: ["reports"] }, { viewer: ["public"] }),
    { role: "viewer", resources: ["reports"] },
  );
});
