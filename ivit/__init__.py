"""
iVIT 2.0 SDK Version Information
"""

__version__ = "2.0.2"
__author__ = "iVIT Development Team"
__email__ = "mannywang@innodisk.com"
__description__ = "Advanced AI Vision Training and Deployment SDK"
__url__ = "https://github.com/innodisk-mannywang/ivit-sdk"

# Release information
RELEASE_DATE = "2025-09-10"
VERSION_INFO = {
    "major": 2,
    "minor": 0, 
    "patch": 2,
    "release": "beta"
}

# Feature support matrix
FEATURES = {
    "classification": True,
    "detection": True,
    "segmentation": True,
    "smart_recommendation": True,
    "inference": True,
    "multi_gpu": True,
    "deployment": False, # Coming soon
}

# Supported frameworks
FRAMEWORKS = {
    "pytorch": ">=2.0.0",
    "ultralytics": ">=8.0.0",
    "torchvision": ">=0.15.0",
}

def get_version():
    """Get current version string."""
    return __version__

def get_version_info():
    """Get detailed version information."""
    return VERSION_INFO

def print_info():
    """Print SDK information."""
    print(f"iVIT 2.0 SDK v{__version__}")
    print(f"Released: {RELEASE_DATE}")
    print(f"Author: {__author__}")
    print(f"Description: {__description__}")
    print("\nSupported Tasks:")
    for task, supported in FEATURES.items():
        status = "✅" if supported else "🚧"
        print(f"  {status} {task}")

if __name__ == "__main__":
    print_info()
