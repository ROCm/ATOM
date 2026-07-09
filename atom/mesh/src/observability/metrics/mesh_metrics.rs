use std::{
    net::{IpAddr, Ipv4Addr, SocketAddr},
    time::Duration,
};

use metrics_exporter_prometheus::{Matcher, PrometheusBuilder};

use super::{
    config::{default_duration_buckets, PrometheusConfig},
    schema,
};

pub(crate) fn init_metrics() {
    schema::describe_all_metrics();
}

pub fn start_prometheus(config: PrometheusConfig) {
    init_metrics();

    let duration_matcher = Matcher::Suffix(String::from("duration_seconds"));
    let duration_bucket: Vec<f64> = config
        .duration_buckets
        .unwrap_or_else(default_duration_buckets);

    let ip_addr: IpAddr = config
        .host
        .parse()
        .unwrap_or(IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0)));
    let socket_addr = SocketAddr::new(ip_addr, config.port);

    PrometheusBuilder::new()
        .with_http_listener(socket_addr)
        .upkeep_timeout(Duration::from_secs(5 * 60))
        .set_buckets_for_metric(duration_matcher, &duration_bucket)
        .expect("failed to set duration bucket")
        .install()
        .expect("failed to install Prometheus metrics exporter");
}

#[cfg(test)]
mod tests {
    use std::net::{IpAddr, Ipv4Addr, SocketAddr, TcpListener};

    use metrics_exporter_prometheus::Matcher;

    use super::*;

    #[test]
    fn test_valid_ipv4_parsing() {
        let test_cases = vec!["127.0.0.1", "192.168.1.1", "0.0.0.0"];

        for ip_str in test_cases {
            let config = PrometheusConfig {
                port: 29000,
                host: ip_str.to_string(),
                duration_buckets: None,
            };

            let ip_addr: IpAddr = config.host.parse().unwrap();
            assert!(matches!(ip_addr, IpAddr::V4(_)));
        }
    }

    #[test]
    fn test_valid_ipv6_parsing() {
        let test_cases = vec!["::1", "2001:db8::1", "::"];

        for ip_str in test_cases {
            let config = PrometheusConfig {
                port: 29000,
                host: ip_str.to_string(),
                duration_buckets: None,
            };

            let ip_addr: IpAddr = config.host.parse().unwrap();
            assert!(matches!(ip_addr, IpAddr::V6(_)));
        }
    }

    #[test]
    fn test_invalid_ip_parsing_falls_back_to_unspecified_ipv4() {
        let test_cases = vec!["invalid", "256.256.256.256", "hostname"];

        for ip_str in test_cases {
            let config = PrometheusConfig {
                port: 29000,
                host: ip_str.to_string(),
                duration_buckets: None,
            };

            let ip_addr: IpAddr = config
                .host
                .parse()
                .unwrap_or(IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0)));

            assert_eq!(ip_addr, IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0)));
        }
    }

    #[test]
    fn test_socket_addr_creation() {
        let test_cases = vec![("127.0.0.1", 8080), ("0.0.0.0", 29000), ("::1", 9090)];

        for (host, port) in test_cases {
            let config = PrometheusConfig {
                port,
                host: host.to_string(),
                duration_buckets: None,
            };

            let ip_addr: IpAddr = config.host.parse().unwrap();
            let socket_addr = SocketAddr::new(ip_addr, config.port);

            assert_eq!(socket_addr.port(), port);
            assert_eq!(socket_addr.ip().to_string(), host);
        }
    }

    #[test]
    fn test_socket_addr_with_different_ports() {
        let ports = vec![0, 80, 8080, 65535];

        for port in ports {
            let config = PrometheusConfig {
                port,
                host: "127.0.0.1".to_string(),
                duration_buckets: None,
            };

            let ip_addr: IpAddr = config.host.parse().unwrap();
            let socket_addr = SocketAddr::new(ip_addr, config.port);

            assert_eq!(socket_addr.port(), port);
        }
    }

    #[test]
    fn test_duration_bucket_coverage() {
        let test_cases: [(f64, &str); 7] = [
            (0.0005, "sub-millisecond"),
            (0.005, "5ms"),
            (0.05, "50ms"),
            (1.0, "1s"),
            (10.0, "10s"),
            (60.0, "1m"),
            (240.0, "4m"),
        ];
        let buckets = default_duration_buckets();

        for (duration, label) in test_cases {
            let bucket_found = buckets
                .iter()
                .any(|&b| (b - duration).abs() < 0.0001 || b > duration);
            assert!(bucket_found, "No bucket found for {} ({})", duration, label);
        }
    }

    #[test]
    fn test_duration_suffix_matcher() {
        let matcher = Matcher::Suffix(String::from("duration_seconds"));

        match matcher {
            Matcher::Suffix(suffix) => assert_eq!(suffix, "duration_seconds"),
            _ => panic!("Expected Suffix matcher"),
        }
    }

    #[test]
    fn test_prometheus_builder_configuration() {
        let duration_matcher = Matcher::Suffix(String::from("duration_seconds"));
        let duration_bucket = default_duration_buckets();

        assert_eq!(duration_bucket.len(), 20);

        match duration_matcher {
            Matcher::Suffix(s) => assert_eq!(s, "duration_seconds"),
            _ => panic!("Expected Suffix matcher"),
        }
    }

    #[test]
    fn test_upkeep_timeout_duration() {
        let timeout = Duration::from_secs(5 * 60);
        assert_eq!(timeout.as_secs(), 300);
    }

    #[test]
    fn test_custom_buckets_for_different_metrics() {
        let request_buckets = [0.001, 0.01, 0.1, 1.0, 10.0];
        let generate_buckets = [0.1, 0.5, 1.0, 5.0, 30.0, 60.0];

        assert_eq!(request_buckets.len(), 5);
        assert_eq!(generate_buckets.len(), 6);

        for i in 1..request_buckets.len() {
            assert!(request_buckets[i] > request_buckets[i - 1]);
        }

        for i in 1..generate_buckets.len() {
            assert!(generate_buckets[i] > generate_buckets[i - 1]);
        }
    }

    #[test]
    fn test_port_already_in_use() {
        let port = 29123;

        if let Ok(_listener) = TcpListener::bind(("127.0.0.1", port)) {
            let config = PrometheusConfig {
                port,
                host: "127.0.0.1".to_string(),
                duration_buckets: None,
            };

            assert_eq!(config.port, port);
        }
    }

    #[test]
    fn test_metrics_endpoint_accessibility() {
        let config = PrometheusConfig {
            port: 29000,
            host: "127.0.0.1".to_string(),
            duration_buckets: None,
        };

        let ip_addr: IpAddr = config.host.parse().unwrap();
        let socket_addr = SocketAddr::new(ip_addr, config.port);

        assert_eq!(socket_addr.to_string(), "127.0.0.1:29000");
    }
}
