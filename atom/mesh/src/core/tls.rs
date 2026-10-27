use rustls::pki_types::{pem::PemObject, CertificateDer, PrivateKeyDer};
use std::{io, path::Path};

pub struct TlsMaterial {
    pub certificate: Vec<u8>,
    pub private_key: Vec<u8>,
}

async fn read(path: &Path, kind: &str) -> io::Result<Vec<u8>> {
    let metadata = tokio::fs::metadata(path).await.map_err(|error| {
        io::Error::new(
            error.kind(),
            format!("TLS {kind} '{}': {error}", path.display()),
        )
    })?;
    if !metadata.is_file() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("TLS {kind} '{}' is not a file", path.display()),
        ));
    }
    tokio::fs::read(path).await.map_err(|error| {
        io::Error::new(
            error.kind(),
            format!("TLS {kind} '{}': {error}", path.display()),
        )
    })
}

pub async fn certificate(path: &Path) -> io::Result<Vec<u8>> {
    let bytes = read(path, "certificate").await?;
    let certs = CertificateDer::pem_slice_iter(&bytes)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("TLS certificate '{}': {error}", path.display()),
            )
        })?;
    if certs.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "TLS certificate '{}' contains no certificates",
                path.display()
            ),
        ));
    }
    Ok(bytes)
}

impl TlsMaterial {
    pub async fn load(cert: &Path, key: &Path) -> io::Result<Self> {
        let certificate = certificate(cert).await?;
        let private_key = read(key, "private key").await?;
        PrivateKeyDer::from_pem_slice(&private_key).map_err(|error| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("TLS private key '{}': {error}", key.display()),
            )
        })?;
        let _ = rustls::crypto::ring::default_provider().install_default();
        Ok(Self {
            certificate,
            private_key,
        })
    }
}
