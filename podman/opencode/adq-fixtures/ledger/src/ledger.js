import { parseEvent } from "./parse.js";

export function reconcile(text) {
  const balances = new Map();
  const seen = new Set();
  for (const line of text.split("\n").filter(Boolean)) {
    const event = parseEvent(line);
    if (seen.has(event.id)) continue;
    seen.add(event.id);
    if (event.supersedes) {
      const old = balances.get(event.account) ?? 0;
      balances.set(event.account, old + event.amount);
    } else {
      balances.set(event.account, (balances.get(event.account) ?? 0) + event.amount);
    }
  }
  return Object.fromEntries([...balances.entries()].sort(([a], [b]) => a.localeCompare(b)));
}
