# Diagnose retry duplication

Read `incident/production.log`, determine the invariant being violated, and repair the scheduler. A job ID may have only one live deadline. Rescheduling the same ID must replace its prior deadline. `due(now)` returns each due job once in deadline-then-ID order and removes it. A stale heap entry must never fire after replacement. Preserve the injected clock API and public method signatures.

Add regression coverage reproducing the incident, run the tests, and briefly explain the diagnosed cause in the final response.
