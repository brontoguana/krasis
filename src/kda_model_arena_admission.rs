// Private recurrence-arena admission. This does not admit producer workspaces
// or replace event ordering across different families or CUDA devices.
struct ModelArenaAdmission<'a> {
    families: ModelFamilyAdmission<'a>,
    owners: Vec<&'a PreparedOwner>,
    arenas: Vec<EnqueueLease<'a>>,
}
fn model_arena_lease<'a, 'b>(
    owner: &PreparedOwner,
    leases: &'b [EnqueueLease<'a>],
) -> Result<&'b EnqueueLease<'a>, String> {
    leases
        .iter()
        .find(|lease| std::ptr::eq(lease.arena, owner.arena.as_ref()))
        .ok_or_else(|| "complete graph recurrence arena missing".into())
}
impl<'a> ModelArenaAdmission<'a> {
    fn checked(
        bindings: &'a [ModelFamilyBinding<'a>],
        stream: &'a NativeStream,
        expected: &serde_json::Value,
    ) -> Result<Self, String> {
        let families = ModelFamilyAdmission::checked(bindings, stream, expected)?;
        Self::acquire(families, bindings)
    }
    fn from_proof(proof: &ModelBindingProof<'a>, stream: &'a NativeStream) -> Result<Self, String> {
        let families = ModelFamilyAdmission::from_proof(proof, stream)?;
        Self::acquire(families, proof.bindings)
    }
    fn acquire(
        families: ModelFamilyAdmission<'a>,
        bindings: &'a [ModelFamilyBinding<'a>],
    ) -> Result<Self, String> {
        let mut arenas: Vec<EnqueueLease<'a>> = Vec::new();
        let mut owners = Vec::with_capacity(bindings.len());
        for b in bindings {
            if !arenas
                .iter()
                .any(|lease| std::ptr::eq(lease.arena, b.owner.arena.as_ref()))
            {
                // Nonblocking acquisition in canonical family order. Partial
                // rejection drops all preceding guards before returning.
                arenas.push(b.owner.arena.lease()?);
            }
            b.owner
                .validate_lease(model_arena_lease(b.owner, &arenas)?)?;
            owners.push(b.owner);
        }
        Ok(Self {
            families,
            owners,
            arenas,
        })
    }
    fn execute<T>(
        &mut self,
        work: impl FnOnce(&[EnqueueLease<'a>]) -> Result<T, String>,
    ) -> Result<T, String> {
        self.families.validate_ready()?;
        for owner in &self.owners {
            owner.validate_lease(model_arena_lease(owner, &self.arenas)?)?;
        }
        let arenas = &self.arenas;
        self.families.execute(|| work(arenas))
    }
}
