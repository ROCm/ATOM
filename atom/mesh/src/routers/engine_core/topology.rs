use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EngineCoreEndpoint {
    pub engine_rank: usize,
    pub dp_rank: usize,
    pub pp_rank: usize,
    pub input_address: String,
    pub control_address: String,
    pub output_address: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EngineCoreEndpointTopology {
    pub endpoints: Vec<EngineCoreEndpoint>,
}

impl EngineCoreEndpointTopology {
    pub fn new(mut endpoints: Vec<EngineCoreEndpoint>) -> Result<Self, String> {
        if endpoints.is_empty() {
            return Err("at least one EngineCore endpoint is required".to_string());
        }

        let mut engine_ranks = BTreeSet::new();
        let mut stage_ranks = BTreeSet::new();
        let mut pipeline_stage_ranks: BTreeMap<usize, BTreeSet<usize>> = BTreeMap::new();
        let mut addresses = BTreeSet::new();
        for endpoint in &endpoints {
            if !engine_ranks.insert(endpoint.engine_rank) {
                return Err(format!(
                    "duplicate EngineCore endpoint for engine_rank {}",
                    endpoint.engine_rank
                ));
            }
            if !stage_ranks.insert((endpoint.dp_rank, endpoint.pp_rank)) {
                return Err(format!(
                    "duplicate EngineCore endpoint for dp_rank {}, pp_rank {}",
                    endpoint.dp_rank, endpoint.pp_rank
                ));
            }
            pipeline_stage_ranks
                .entry(endpoint.dp_rank)
                .or_default()
                .insert(endpoint.pp_rank);
            for (name, address) in [
                ("input_address", &endpoint.input_address),
                ("control_address", &endpoint.control_address),
                ("output_address", &endpoint.output_address),
            ] {
                if address.is_empty() {
                    return Err(format!(
                        "{name} is empty for EngineCore engine_rank {}",
                        endpoint.engine_rank
                    ));
                }
                if !addresses.insert(address.clone()) {
                    return Err(format!("duplicate EngineCore ZMQ address {address}"));
                }
            }
        }
        for (dp_rank, pp_ranks) in pipeline_stage_ranks {
            let expected = (0..pp_ranks.len()).collect::<BTreeSet<_>>();
            if pp_ranks != expected {
                return Err(format!(
                    "EngineCore pipeline dp_rank {dp_rank} must contain contiguous PP stages \
                     starting at 0; got {pp_ranks:?}"
                ));
            }
        }

        endpoints.sort_unstable_by_key(|endpoint| endpoint.engine_rank);
        Ok(Self { endpoints })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn endpoint(dp_rank: usize) -> EngineCoreEndpoint {
        EngineCoreEndpoint {
            engine_rank: dp_rank,
            dp_rank,
            pp_rank: 0,
            input_address: format!("ipc:///tmp/atom-input-{dp_rank}"),
            control_address: format!("ipc:///tmp/atom-control-{dp_rank}"),
            output_address: format!("ipc:///tmp/atom-output-{dp_rank}"),
        }
    }

    #[test]
    fn validates_and_sorts_endpoints() {
        let topology = EngineCoreEndpointTopology::new(vec![endpoint(2), endpoint(0)]).unwrap();
        assert_eq!(
            topology
                .endpoints
                .iter()
                .map(|endpoint| endpoint.dp_rank)
                .collect::<Vec<_>>(),
            vec![0, 2]
        );

        let mut second_stage = endpoint(1);
        second_stage.dp_rank = 0;
        second_stage.pp_rank = 1;
        let pp_topology = EngineCoreEndpointTopology::new(vec![second_stage, endpoint(0)]).unwrap();
        assert_eq!(
            pp_topology
                .endpoints
                .iter()
                .map(|endpoint| (endpoint.engine_rank, endpoint.pp_rank))
                .collect::<Vec<_>>(),
            vec![(0, 0), (1, 1)]
        );
    }

    #[test]
    fn rejects_duplicate_ranks_and_addresses() {
        assert!(EngineCoreEndpointTopology::new(vec![endpoint(0), endpoint(0)]).is_err());

        let mut duplicate_address = endpoint(1);
        duplicate_address.input_address = endpoint(0).output_address;
        assert!(EngineCoreEndpointTopology::new(vec![endpoint(0), duplicate_address]).is_err());

        let mut duplicate_stage = endpoint(1);
        duplicate_stage.dp_rank = 0;
        assert!(EngineCoreEndpointTopology::new(vec![endpoint(0), duplicate_stage]).is_err());

        let mut missing_head = endpoint(1);
        missing_head.pp_rank = 1;
        assert!(EngineCoreEndpointTopology::new(vec![missing_head]).is_err());

        let mut stage_with_gap = endpoint(2);
        stage_with_gap.dp_rank = 0;
        stage_with_gap.pp_rank = 2;
        assert!(EngineCoreEndpointTopology::new(vec![endpoint(0), stage_with_gap]).is_err());
    }
}
