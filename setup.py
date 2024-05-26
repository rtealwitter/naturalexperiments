import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="naturalexperiments",
    version="0.1.4",
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
)
