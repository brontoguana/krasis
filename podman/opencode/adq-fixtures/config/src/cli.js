const defaults = { enabled: true, retries: 3, label: "service" };

export function cliConfig(file, env, cli) {
  return {
    enabled: cli.enabled || env.enabled || file.enabled || defaults.enabled,
    retries: cli.retries || env.retries || file.retries || defaults.retries,
    label: cli.label || env.label || file.label || defaults.label,
  };
}
