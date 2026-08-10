"""Setup script for the pinnkit package."""

from setuptools import setup, find_packages

install_requires = [
    "torch>=1.9.0",
    "numpy>=1.20.0",
    "PyYAML>=6.0.0",
    "tqdm>=4.60.0",
]

extras_require = {
    "viz": ["matplotlib>=3.3.0", "seaborn>=0.11.0", "plotly>=5.0.0", "scipy>=1.7.0"],
    "dev": [
        "pytest>=6.0.0",
        "jupyter>=1.0.0",
        "ipykernel>=6.0.0",
        "black==26.5.1",
        "mypy",
    ],
    "gpu": ["cupy-cuda11x"],
    "deploy": ["fastapi>=0.95", "uvicorn>=0.20", "grpcio>=1.54"],
}

if __name__ == "__main__":
    setup(
        name="pinnkit",
        version="0.2.0",
        author="Diogo Ribeiro",
        author_email="dfr@esmad.ipp.pt",
        description="Physics-Informed Neural Network implementations",
        packages=find_packages("src"),
        package_dir={"": "src"},
        python_requires=">=3.10",
        install_requires=install_requires,
        extras_require=extras_require,
        entry_points={
            "console_scripts": [
                "pinnkit=pinnkit.cli:main",
            ]
        },
    )
