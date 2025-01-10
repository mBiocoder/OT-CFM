from setuptools import setup, find_packages

setup(
    name="ot_cfm",
    version="0.1.0",
    description="Optimal Transport Conditional Flow Matching for Batch Correction",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "scanpy",
        "scprep",
        "matplotlib",
        "torch",
        "torchsde",
        "torchdyn",
        "tqdm",
        "torchcfm",
    ],
)
