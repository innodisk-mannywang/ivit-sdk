"""
iVIT 2.0 SDK - AI Vision Training & Deployment Toolkit
=====================================================

A comprehensive SDK for AI vision model training and deployment,
featuring intelligent parameter recommendations and multi-platform support.
"""

from setuptools import setup, find_packages
import os

# Read README file
current_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(current_dir, "README.md"), "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open(os.path.join(current_dir, "requirements.txt"), "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ivit",
    version="2.0.0",
    author="iVIT Team",
    author_email="support@ivit.ai",
    description="AI Vision Training & Deployment SDK with intelligent recommendations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/ivit-2.0-sdk",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": ["pytest>=7.0.0", "black>=22.0.0", "flake8>=5.0.0"],
    },
    include_package_data=True,
    zip_safe=False,
)
