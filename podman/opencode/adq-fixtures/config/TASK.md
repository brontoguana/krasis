# Centralize configuration resolution

CLI and HTTP entry points duplicated configuration precedence and incorrectly replace explicit `false`, `0`, and empty-string values because they use truthiness. Refactor them to one exported resolver in `src/resolve.js`.

Required precedence is CLI override, then environment, then file, then defaults. Only `undefined` means absent. Valid keys are `enabled`, `retries`, and `label`; unknown keys from any layer must throw an error naming the key. `enabled` must be boolean, `retries` a non-negative integer, and `label` a string (including the empty string). Both entry points must produce identical results and their existing public API signatures must remain stable.

Make the smallest multi-file refactor, add regression coverage for falsey values and an unknown key, run the tests, and preserve protected and unrelated files.
