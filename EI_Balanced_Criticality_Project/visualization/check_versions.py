import brainpy
import jax
import sys
import platform

try:
    import brainevent
    brainevent_version = brainevent.__version__
except ImportError:
    brainevent_version = "Not Installed"

def check_env():
    print("="*60)
    print("Environment Version Check")
    print("="*60)
    print(f"Python:     {sys.version.split()[0]}")
    print(f"Platform:   {platform.platform()}")
    print("-" * 60)
    print(f"BrainPy:    {brainpy.__version__}")
    print(f"JAX:        {jax.__version__}")
    print(f"Brainevent: {brainevent_version}")
    print("-" * 60)
    
    # Check JAX Backend
    try:
        devices = jax.devices()
        backend = jax.lib.xla_bridge.get_backend().platform
        print(f"JAX Backend: {backend}")
        print(f"Devices:     {devices}")
    except Exception as e:
        print(f"JAX Backend Error: {e}")

    print("="*60)

if __name__ == "__main__":
    check_env()