import os  
import socket  
import warnings  
  
def get_ip() -> str:  
    # Check environment variable first  
    host_ip = os.environ.get("ATOM_HOST_IP")  
    if "HOST_IP" in os.environ and "ATOM_HOST_IP" not in os.environ:  
        warnings.warn(  
            "The environment variable HOST_IP is deprecated and ignored, as"  
            " it is often used by Docker and other software to"  
            " interact with the container's network stack. Please "  
            "use ATOM_HOST_IP instead to set the IP address for ATOM processes"  
            " to communicate with each other."  
        )  
    if host_ip:  
        return host_ip  
  
    # Try IPv4 by connecting to Google's DNS  
    try:  
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:  
            s.connect(("8.8.8.8", 80))  # Doesn't need to be reachable  
            return s.getsockname()[0]  
    except Exception:  
        pass  
  
    # Try IPv6 by connecting to Google's IPv6 DNS  
    try:  
        with socket.socket(socket.AF_INET6, socket.SOCK_DGRAM) as s:  
            s.connect(("2001:4860:4860::8888", 80))  # Doesn't need to be reachable  
            return s.getsockname()[0]  
    except Exception:  
        pass  
  
    # Fallback to 0.0.0.0 if all methods fail  
    warnings.warn(  
        "Failed to get the IP address, using 0.0.0.0 by default."  
        "The value can be set by the environment variable"  
        " ATOM_HOST_IP or HOST_IP.",  
        stacklevel=2,  
    )  
    return "0.0.0.0"