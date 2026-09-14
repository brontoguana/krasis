export function parseEvent(line) {
  const value = JSON.parse(line);
  if (typeof value.id !== "string" || value.id.length === 0) throw new Error("invalid id");
  if (typeof value.account !== "string" || value.account.length === 0) throw new Error("invalid account");
  const amount = Number(value.amount);
  if (!Number.isFinite(amount)) throw new Error("invalid amount");
  if (value.supersedes !== undefined && (typeof value.supersedes !== "string" || value.supersedes.length === 0)) {
    throw new Error("invalid supersedes");
  }
  return { id: value.id, account: value.account, amount, ...(value.supersedes ? { supersedes: value.supersedes } : {}) };
}
