// Cold, one-shot test control on the serialized real model worker. This names
// an engine error at generation entry; it does not simulate a failed CUDA copy
// or qualify resource completion, native poison, or numerical recovery.
use std::cell::Cell;
thread_local! { static ARMED: Cell<bool> = const { Cell::new(false) }; }
pub(crate) fn action(action: &str) -> Result<bool, String> {
    ARMED.with(|armed| match action {
        "arm" if !armed.get() => {
            armed.set(true);
            Ok(true)
        }
        "arm" => Err("publication failure control is already armed".into()),
        "clear" => {
            armed.set(false);
            Ok(false)
        }
        _ => Err("publication failure action must be arm or clear".into()),
    })
}
pub(crate) fn take() -> Option<String> {
    ARMED.with(|armed| {
        armed
            .replace(false)
            .then(|| "explicit publication control: generation entry rejected".into())
    })
}

// A separate one-shot control is consumed only after a genuine prompt-state
// destination copy. Native LA invokes this inside its real completion guard;
// this deliberately returns Err after enqueue, without fabricating a CUDA error.
thread_local! { static AUXILIARY: Cell<Option<&'static str>> = const { Cell::new(None) }; }
pub(crate) fn auxiliary_action(action: &str) -> Result<bool, String> {
    AUXILIARY.with(|armed| {
        if action == "clear" {
            armed.set(None);
            return Ok(false);
        }
        let stage = match action {
            "arm-kv" => "kv",
            "arm-native-la" => "native-la",
            _ => {
                return Err(
                    "auxiliary failure action must be arm-kv, arm-native-la or clear".into(),
                )
            }
        };
        if armed.get().is_some() {
            return Err("auxiliary failure control is already armed".into());
        }
        armed.set(Some(stage));
        Ok(true)
    })
}
pub(crate) fn after_auxiliary_copy(stage: &str) -> Result<(), String> {
    AUXILIARY.with(|armed| {
        if armed.get() == Some(stage) {
            armed.set(None);
            Err(format!(
                "explicit auxiliary control: {stage} destination copy completed/enqueued"
            ))
        } else {
            Ok(())
        }
    })
}
