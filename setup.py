from setuptools import setup

with open("README.md", "r") as arq:
    readme = arq.read()

setup(
    name="tinyshift",
    version="0.0.1",
    license="MIT License",
    author="Lucas Leão",
    long_description=readme,
    long_description_content_type="text/markdown",
    author_email="heylucasleao@gmail.com",
    description="A small toolbox for mlops",
    packages=["tinyshift"],
    install_requires=["pandas", "scipy", "plotly", "scikit-learn", "numpy"],
)
