pub mod backend;
pub mod candidate;
pub mod planner;
pub mod policy_apply;
pub mod trace;
pub mod traits;
pub mod types;

pub use traits::{PdPlanner, PolicySource, WorkerSource};
pub use types::{
    AdapterError, PlacementError, PlacementPlan, PlacementTrace, Protocol, RequestDescriptor,
};

#[cfg(test)]
pub(crate) mod test_support;

#[cfg(test)]
mod tests_candidate;
#[cfg(test)]
mod tests_policy_apply;
#[cfg(test)]
mod tests_regular_planning;
#[cfg(test)]
mod tests_pd_planning;
#[cfg(test)]
mod tests_adapter;
#[cfg(test)]
mod tests_trace;
#[cfg(test)]
mod tests_error;
#[cfg(test)]
mod tests_integration;
