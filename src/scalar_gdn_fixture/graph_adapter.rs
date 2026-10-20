//! Borrowed graph execution; all operand/module/stream owners outlive destruction.
use super::{fail,InputStorage};
use crate::{
    scalar_gdn_owner::{drain_or_fatal, Owner},
    shared_gdn_workspace::Workspace,
};
use cudarc::driver::{sys, CudaSlice};

pub struct Graph<'a> {
    owner: &'a mut Owner,
    slots: &'a InputStorage,
    position: &'a CudaSlice<i32>,
    workspace: &'a Workspace,
    graph: sys::CUgraph,
    exec: sys::CUgraphExec,
    nodes: usize,
}
impl<'a> Graph<'a> {
    pub fn capture(
        owner: &'a mut Owner,
        slots: &'a InputStorage,
        position: &'a CudaSlice<i32>,
        workspace: &'a Workspace,
        inject_after_enqueue: bool,
    ) -> Result<Self, String> {
        let lease = workspace.lease()?;
        let mut graph = Self {
            owner,
            slots,
            position,
            workspace,
            graph: std::ptr::null_mut(),
            exec: std::ptr::null_mut(),
            nodes: 0,
        };
        unsafe {
            sys::lib().cuStreamBeginCapture_v2(
                *workspace.program.device.cu_stream(),
                sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
            )
        }
        .result()
        .map_err(fail)?;
        let enqueue = (|| {
            let view = graph.slots.view(0, 1)?;
            unsafe { graph.owner.enqueue_decode(&lease, &view, graph.position) }
        })();
        // Always close capture, including CPU errors after kernels were captured.
        // The closure also routes input-view errors through EndCapture.
        let end = unsafe {
            sys::lib().cuStreamEndCapture(*workspace.program.device.cu_stream(), &mut graph.graph)
        }
        .result()
        .map_err(fail);
        enqueue?;
        end?;
        if inject_after_enqueue {
            return Err("explicit CPU injection after successful captured enqueue".into());
        }
        if graph.graph.is_null() {
            return Err("capture ended without an actual graph".into());
        }
        unsafe { sys::lib().cuGraphGetNodes(graph.graph, std::ptr::null_mut(), &mut graph.nodes) }
            .result()
            .map_err(fail)?;
        if graph.nodes == 0 {
            return Err("captured graph contains no actual nodes".into());
        }
        unsafe { sys::lib().cuGraphInstantiateWithFlags(&mut graph.exec, graph.graph, 0) }
            .result()
            .map_err(fail)?;
        Ok(graph)
    }
    pub fn replay(&mut self) -> Result<(), String> {
        let _lease = self.workspace.lease()?;
        unsafe { sys::lib().cuGraphLaunch(self.exec, *self.workspace.program.device.cu_stream()) }
            .result()
            .map_err(fail)
    }
    pub fn owner(&self) -> &Owner {
        self.owner
    }
    pub fn nodes(&self) -> usize {
        self.nodes
    }
}
impl Drop for Graph<'_> {
    fn drop(&mut self) {
        drain_or_fatal(&self.workspace.program.device, "scalar graph teardown");
        let outcome = (|| -> Result<(), String> {
            if !self.exec.is_null() {
                unsafe { sys::lib().cuGraphExecDestroy(self.exec) }
                    .result()
                    .map_err(fail)?;
            }
            if !self.graph.is_null() {
                unsafe { sys::lib().cuGraphDestroy(self.graph) }
                    .result()
                    .map_err(fail)?;
            }
            Ok(())
        })();
        if let Err(e) = outcome {
            crate::vram_monitor::fatal_cuda_context_error(
                "scalar graph destroy completion unknown",
                &e,
            );
        }
    }
}
