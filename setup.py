from setuptools import setup, find_packages

setup(
    name="multilingual-sentiment-api",
    version="0.1.0",
    description="Multilingual sentiment analysis API using XLM-RoBERTa",
    author="Bakhtiar Rasheed",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "datasets>=2.14.0",
        "fastapi>=0.100.0",
        "uvicorn[standard]>=0.23.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "python-dotenv>=1.0.0",
    ],
)
