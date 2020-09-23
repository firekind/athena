import setuptools

setuptools.setup(
    name="athena",
    version="1.0.0",
    author="Shyamant Achar",
    packages=setuptools.find_packages(),
    python_requires='>=3.6.8',
    install_requires=[
        "tqdm~=4.48.2",
        "matplotlib~=3.3.1",
        "tensorboard~=2.3.0",
        "albumentations~=0.4.6",
        "opencv-python~=4.4.0.42"
    ]
)