import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

# Load required packages from requirements.txt
with open('requirements.txt') as f:
    required = f.read().splitlines()

setuptools.setup(
    name="naturalexperiments",
    version="0.2.0",
    author="R. Teal Witter",
    author_email="rtealwitter@gmail.com",
    description="Estimators and datasets for treatment effect estimation in natural experiments.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rtealwitter/naturalexperiments",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=required,
)
