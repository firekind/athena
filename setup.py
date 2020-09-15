import setuptools

setuptools.setup(
    name="athena",
    version="0.0.1",
    author="Shyamant Achar",
    packages=setuptools.find_packages(),
    python_requires='>=3.6.8',
    install_requires=[
        "pkbar==0.4",
        "torchsummary",
        "tqdm",
        "matplotlib",
        "tensorboard"
    ]
)