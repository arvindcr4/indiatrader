from setuptools import setup, find_packages

setup(
    name="indiatrader",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.2.0",
        "torch>=2.0.0",
        "transformers>=4.38.0",
        "pyarrow>=14.0.0",
        "fastapi>=0.105.0",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="AI-Driven Intraday Trading Platform for Indian Stock Markets",
    keywords="trading, machine learning, finance, stock market, india",
    url="https://github.com/yourusername/indiatrader",
    python_requires=">=3.10",
)