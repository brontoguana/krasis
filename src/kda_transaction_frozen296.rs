// cfgtest-only exact frozen296 control; never selected in serving.
impl ModelFamilyAdmission<'_> {
    fn capture_frozen296<T>(&mut self, work: impl FnOnce() -> Result<T, String>) -> Result<T, String> {
        // No event wait/record or host phase mutation belongs inside capture.
        // All admitted family mutexes remain held; arena leases are taken by work.
        let before = self
            .slots
            .iter()
            .map(|s| {
                let t = s.as_ref().expect("checked family");
                (t.phases, t.recorded, t.poisoned)
            })
            .collect::<Vec<_>>();
        let result = work();
        for (slot, want) in self.slots.iter().zip(before) {
            let t = slot.as_ref().ok_or("capture removed timeline")?;
            if (t.phases, t.recorded, t.poisoned) != want {
                return Err("capture changed committed host family timeline".into());
            }
        }
        result
    }
    fn execute_frozen296<T>(&mut self, work: impl FnOnce() -> Result<T, String>) -> Result<T, String> {
        // All families have passed admission before any GPU submission.
        // Mark the entire transaction poisoned until every completion records.
        for slot in &mut self.slots {
            slot.as_mut().ok_or("family vanished")?.poisoned = true;
        }
        for slot in &self.slots {
            let t = slot.as_ref().ok_or("family vanished")?;
            if t.recorded {
                check(
                    unsafe { sys::lib().cuStreamWaitEvent(self.stream.handle, t.event, 0) },
                    "complete graph preceding family wait",
                )?;
            }
        }
        let value = work();
        let mut failure = None;
        for slot in &mut self.slots {
            let t = slot.as_mut().ok_or("family vanished")?;
            match check(
                unsafe { sys::lib().cuEventRecord(t.event, self.stream.handle) },
                "complete graph family completion record",
            ) {
                Ok(()) => t.recorded = true,
                Err(error) => {
                    if failure.is_none() {
                        failure = Some(error);
                    }
                }
            }
        }
        if let Some(error) = failure {
            return Err(error);
        }
        match value {
            Ok(value) => {
                for slot in &mut self.slots {
                    let t = slot.as_mut().ok_or("family vanished")?;
                    t.phases = t
                        .phases
                        .checked_add(1)
                        .ok_or("complete graph phase overflow")?;
                    t.poisoned = false;
                }
                Ok(value)
            }
            Err(error) => Err(error),
        }
    }
}
