from setuptools import setup, find_packages

setup(
    name="feature_fabrica",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "pyyaml",
        "numpy",
    ],
    entry_points={
        "console_scripts": [
            "feature-fabrica=feature_fabrica.cli:main",
        ],
    },
    include_package_data=True,
    description="Feature engineering framework for transparency and scalability",
    author="Chingis Oinar",
    license="MIT",
)
