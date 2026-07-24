from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).parent


def read_requirements():
    requirements = []
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-"):
            continue
        if line.startswith("pip"):
            continue
        requirements.append(line)
    return requirements


setup(
    name="cellforge",
    version="0.1.0",
    packages=find_packages(include=["cellforge", "cellforge.*"]),
    py_modules=["main"],
    entry_points={"console_scripts": ["cellforge=main:cli_entrypoint"]},
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.1.0",
            "isort>=5.12.0",
            "mypy>=1.3.0",
        ],
    },
    python_requires=">=3.9,<3.13",
    author="CellForge authors",
    description="Open-ended autonomous design of computational methods for single-cell omics via multi-agent collaboration",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/GabbyKoki/cellforge-new",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)
