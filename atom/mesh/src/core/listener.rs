use std::net::SocketAddr;

/// Port zero asks the OS to choose a port and never conflicts by configuration.
pub fn overlaps(a: SocketAddr, b: SocketAddr) -> bool {
    a.port() != 0
        && a.port() == b.port()
        && (a.ip() == b.ip()
        || (a.is_ipv4() == b.is_ipv4() && (a.ip().is_unspecified() || b.ip().is_unspecified()))
        // An IPv6 wildcard may also accept IPv4 on dual-stack hosts.
        || (a.is_ipv6() && a.ip().is_unspecified())
        || (b.is_ipv6() && b.ip().is_unspecified()))
}

pub fn validate(listeners: &[(&str, SocketAddr)]) -> Result<(), std::io::Error> {
    for (i, (name, address)) in listeners.iter().enumerate() {
        for (other, other_address) in &listeners[..i] {
            if overlaps(*address, *other_address) {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::AddrInUse,
                    format!("{name} listener {address} overlaps {other} listener {other_address}"),
                ));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn collision_checks_cover_wildcards_families_and_ephemeral_ports() {
        for (a, b, conflict) in [
            ("127.0.0.1:9002", "127.0.0.1:9002", true),
            ("0.0.0.0:9002", "127.0.0.1:9002", true),
            ("[::]:9002", "127.0.0.1:9002", true),
            ("[::]:9002", "[::1]:9002", true),
            ("127.0.0.1:9002", "127.0.0.2:9002", false),
            ("127.0.0.1:9002", "[::1]:9002", false),
            ("127.0.0.1:9002", "127.0.0.1:9003", false),
            ("0.0.0.0:0", "127.0.0.1:0", false),
        ] {
            let (a, b) = (a.parse().unwrap(), b.parse().unwrap());
            assert_eq!(overlaps(a, b), conflict);
            assert_eq!(overlaps(b, a), conflict);
            let result = validate(&[("HTTP", a), ("metrics", b)]);
            assert_eq!(result.is_err(), conflict);
            if let Err(error) = result {
                assert!(
                    error.to_string().contains("HTTP") && error.to_string().contains("metrics")
                );
            }
        }
    }
}
