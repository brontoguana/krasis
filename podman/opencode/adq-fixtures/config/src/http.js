const defaults = { enabled: true, retries: 3, label: "service" };

export function httpConfig(file, env, requestOverrides) {
  return {
    enabled: requestOverrides.enabled || env.enabled || file.enabled || defaults.enabled,
    retries: requestOverrides.retries || env.retries || file.retries || defaults.retries,
    label: requestOverrides.label || env.label || file.label || defaults.label,
  };
}
