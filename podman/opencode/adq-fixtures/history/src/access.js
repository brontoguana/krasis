export function normalizeRequest(request, roleDefaults) {
  const role = request.role || "viewer";
  const resources = request.resources || roleDefaults[role] || [];
  return { role, resources };
}
