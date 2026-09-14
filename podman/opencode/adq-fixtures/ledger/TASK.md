# Repair correction-chain reconciliation

Production sometimes receives correction events before the event they supersede. The current ledger incorrectly depends on input order and silently accepts conflicting duplicate IDs.

Implement order-independent reconciliation with these rules:

1. Parse non-empty JSONL records. `id` and `account` are non-empty strings, `amount` is a finite number or numeric string, and optional `supersedes` is a non-empty string.
2. Duplicate identity is defined after `parseEvent` normalization. Records with the same ID count once when their normalized `account`, numeric `amount`, and optional `supersedes` presence/value are all equal. Raw JSON whitespace, property order, numeric spelling, and undeclared fields do not affect duplicate identity. If any declared normalized field differs, throw an error mentioning `conflicting duplicate` and the ID.
3. A correction replaces the complete effect of the event it supersedes, including when the target arrives later. Correction chains are valid; cycles and missing targets must throw clear errors.
4. A correction must use the same account as its target. Balances are returned as an object whose keys are sorted lexicographically.
5. Add a focused regression test for the defect. Do not weaken existing tests.

Locate the defect, implement the smallest coherent repair, run the tests, and summarize the verified result.
