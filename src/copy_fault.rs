// Compiled only for explicit installed lifecycle diagnostics.
use cudarc::driver::sys;
use std::cell::RefCell;
#[derive(Clone, Copy, Debug)]
pub enum Fault {
    EventError,
    Panic,
}
#[derive(Default)]
struct State {
    armed: Option<(usize, Fault)>,
    enqueues: usize,
    before_return: Option<String>,
}
thread_local! { static STATE: RefCell<State> = RefCell::new(State::default()); }
pub fn arm(index: usize, fault: Fault) {
    STATE.with(|state| {
        *state.borrow_mut() = State {
            armed: Some((index, fault)),
            ..State::default()
        }
    });
}
pub fn take() -> (usize, Option<String>) {
    STATE.with(|state| {
        let old = std::mem::take(&mut *state.borrow_mut());
        (old.enqueues, old.before_return)
    })
}
pub fn after_enqueue(stream: sys::CUstream) -> Result<(), String> {
    let fault = STATE.with(|state| {
        let mut state = state.borrow_mut();
        let index = state.enqueues;
        state.enqueues += 1;
        let fault = state
            .armed
            .filter(|(target, _)| *target == index)
            .map(|(_, fault)| fault);
        if fault.is_some() {
            state.before_return =
                Some(format!("{:?}", unsafe { sys::lib().cuStreamQuery(stream) }));
            state.armed = None;
        }
        fault
    });
    match fault {
        None => Ok(()),
        Some(Fault::Panic) => panic!("installed copy lifecycle: post-enqueue unwind"),
        Some(Fault::EventError) => {
            // A real driver error at the event boundary, after the real DMA.
            let result = unsafe { sys::lib().cuEventRecord(std::ptr::null_mut(), stream) };
            if result == sys::CUresult::CUDA_SUCCESS {
                return Err("null event unexpectedly accepted by CUDA".into());
            }
            Err(format!(
                "installed copy lifecycle: post-enqueue event error: {result:?}"
            ))
        }
    }
}
