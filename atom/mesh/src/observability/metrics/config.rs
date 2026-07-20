#[derive(Debug, Clone)]
pub struct PrometheusConfig {
    pub port: u16,
    pub host: String,
    pub duration_buckets: Option<Vec<f64>>,
}

impl Default for PrometheusConfig {
    fn default() -> Self {
        Self {
            port: 29000,
            host: "0.0.0.0".to_string(),
            duration_buckets: None,
        }
    }
}

pub fn default_duration_buckets() -> Vec<f64> {
    vec![
        0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 15.0, 30.0, 45.0,
        60.0, 90.0, 120.0, 180.0, 240.0,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prometheus_config_default() {
        let config = PrometheusConfig::default();
        assert_eq!(config.port, 29000);
        assert_eq!(config.host, "0.0.0.0");
        assert!(config.duration_buckets.is_none());
    }

    #[test]
    fn test_prometheus_config_custom() {
        let config = PrometheusConfig {
            port: 8080,
            host: "127.0.0.1".to_string(),
            duration_buckets: None,
        };
        assert_eq!(config.port, 8080);
        assert_eq!(config.host, "127.0.0.1");
    }

    #[test]
    fn test_prometheus_config_clone() {
        let config = PrometheusConfig {
            port: 9090,
            host: "192.168.1.1".to_string(),
            duration_buckets: None,
        };
        let cloned = config.clone();
        assert_eq!(cloned.port, config.port);
        assert_eq!(cloned.host, config.host);
    }

    #[test]
    fn test_default_duration_buckets() {
        let buckets = default_duration_buckets();
        assert_eq!(buckets.len(), 20);

        for i in 1..buckets.len() {
            assert!(buckets[i] > buckets[i - 1]);
        }
    }
}
