// Sealed complete storage binding witness; borrowed owners cannot be
// reconfigured during its lifetime. GPU input/state values may change.
// This is not a whole-model weight/graph epoch witness.
struct ModelBindingProof<'a> {
    bindings: &'a [ModelFamilyBinding<'a>],
    identity: serde_json::Value,
}
impl<'a> ModelBindingProof<'a> {
    fn checked(
        bindings: &'a [ModelFamilyBinding<'a>],
        expected: &serde_json::Value,
    ) -> Result<Self, String> {
        let identity = model_family_identity(bindings)?;
        if &identity != expected {
            return Err("complete graph family identity or inventory incomplete".into());
        }
        Ok(Self { bindings, identity })
    }
}
