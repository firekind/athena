import setuptools

setuptools.setup(
    name="athena",
    version="0.0.2",
    author="Shyamant Achar",
    packages=setuptools.find_packages(),
    python_requires='>=3.6.8',
    install_requires=[
        "tqdm~=4.48.2",
        "matplotlib~=3.2.2",
        "tensorboard~=2.2.0",
        "albumentations~=0.4.6",
        "opencv-python~=4.4.0.42",
        "pytorch-lightning~=1.0.0",
        "ipympl~=0.5.8"
    ]
)