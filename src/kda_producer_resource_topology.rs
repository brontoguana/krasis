// Cold requested-span topology only. This neither owns GPU memory nor admits
// producer execution. Locks, completion events and model epochs are separate.
#[derive(Clone, Debug, PartialEq, Eq)]
struct ProducerWorkspaceSpans {
    context: usize,
    spans: Vec<(u64, usize)>,
}
#[derive(Debug, PartialEq, Eq)]
struct ProducerResourceTopology {
    groups: Vec<Vec<usize>>,
}
impl ProducerResourceTopology {
    fn checked(workspaces: &[ProducerWorkspaceSpans]) -> Result<Self, String> {
        if workspaces.is_empty() {
            return Err("producer resource topology requires complete workspaces".into());
        }
        let mut intervals = Vec::new();
        for (index, workspace) in workspaces.iter().enumerate() {
            if workspace.context == 0 || workspace.spans.is_empty() {
                return Err("producer resource workspace context or extent absent".into());
            }
            for &(start, bytes) in &workspace.spans {
                if start == 0 || bytes == 0 {
                    return Err("producer resource span is empty/null".into());
                }
                let end = start
                    .checked_add(
                        u64::try_from(bytes).map_err(|_| "producer resource size overflow")?,
                    )
                    .ok_or("producer resource pointer overflow")?;
                intervals.push((workspace.context, start, end, index));
            }
        }
        intervals.sort_unstable();
        let mut parent = (0..workspaces.len()).collect::<Vec<_>>();
        fn root(parent: &mut [usize], mut i: usize) -> usize {
            while parent[i] != i {
                parent[i] = parent[parent[i]];
                i = parent[i];
            }
            i
        }
        // An interval connected component has a continuous union. Retaining
        // its furthest endpoint suffices to join every overlapping interval.
        let mut active: Option<(usize, u64, usize)> = None;
        for (context, start, end, index) in intervals {
            if let Some((previous_context, limit, representative)) = active {
                if context == previous_context && start < limit {
                    let a = root(&mut parent, representative);
                    let b = root(&mut parent, index);
                    parent[b] = a;
                    active = Some((context, limit.max(end), representative));
                    continue;
                }
            }
            active = Some((context, end, index));
        }
        let mut components = std::collections::BTreeMap::<usize, Vec<usize>>::new();
        for i in 0..workspaces.len() {
            components.entry(root(&mut parent, i)).or_default().push(i);
        }
        let mut groups = components.into_values().collect::<Vec<_>>();
        groups.sort_unstable_by_key(|group| group[0]);
        Ok(Self { groups })
    }
}
