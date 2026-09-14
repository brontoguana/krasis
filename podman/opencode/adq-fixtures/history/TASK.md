# Repair access-policy normalization from the full issue history

Production requests are being normalized inconsistently. Read the complete attached `ISSUE_HISTORY.md`, recover the three current `ACTIVE POLICY` decisions distributed through it, and ignore nearby revoked proposals.

Implement those decisions in `src/access.js`, add focused regression coverage, and retain the existing exported `normalizeRequest(request, roleDefaults)` API. Make the smallest coherent repair. Run the tests and summarize both the policy you recovered and the verified result.
