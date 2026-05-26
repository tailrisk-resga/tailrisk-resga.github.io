from setuptools import find_packages, setup


setup(
    name="es_prediction",
    version="0.1.0",
    description="Reproducible code for expected shortfall prediction experiments.",
    packages=find_packages("src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "statsmodels",
        "torch",
        "tqdm",
        "joblib",
        "matplotlib",
        "seaborn",
        "python-dateutil",
        "einops",
        "PyYAML",
    ],
    extras_require={
        "baselines": ["rpy2", "numba"],
        "logging": ["wandb"],
        "dev": ["pytest", "ruff"],
    },
)
