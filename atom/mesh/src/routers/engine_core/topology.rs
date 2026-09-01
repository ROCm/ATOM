use std::collections::BTreeSet;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EngineCoreEndpoint {
    pub dp_rank: usize,
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

        let mut ranks = BTreeSet::new();
        let mut addresses = BTreeSet::new();
        for endpoint in &endpoints {
            if !ranks.insert(endpoint.dp_rank) {
                return Err(format!(
                    "duplicate EngineCore endpoint for dp_rank {}",
                    endpoint.dp_rank
                ));
            }
            for (name, address) in [
                ("input_address", &endpoint.input_address),
                ("control_address", &endpoint.control_address),
                ("output_address", &endpoint.output_address),
            ] {
                if address.is_empty() {
                    return Err(format!(
                        "{name} is empty for EngineCore dp_rank {}",
                        endpoint.dp_rank
                    ));
                }
                if !addresses.insert(address.clone()) {
                    return Err(format!("duplicate EngineCore ZMQ address {address}"));
                }
            }
        }

        endpoints.sort_unstable_by_key(|endpoint| endpoint.dp_rank);
        Ok(Self { endpoints })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn endpoint(dp_rank: usize) -> EngineCoreEndpoint {
        EngineCoreEndpoint {
            dp_rank,
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
    }

    #[test]
    fn rejects_duplicate_ranks_and_addresses() {
        assert!(EngineCoreEndpointTopology::new(vec![endpoint(0), endpoint(0)]).is_err());

        let mut duplicate_address = endpoint(1);
        duplicate_address.input_address = endpoint(0).output_address;
        assert!(EngineCoreEndpointTopology::new(vec![endpoint(0), duplicate_address]).is_err());
    }
}
