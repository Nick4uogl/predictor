from setuptools import setup, find_packages

setup(
    name="environmental_predictor",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "scikit-learn>=0.24.0",
        "sqlalchemy>=1.4.0",
        "joblib>=1.0.0",
        "python-dotenv>=0.19.0",
        "seaborn>=0.12.0",
        "tensorflow-cpu>=2.15.0",
    ],
    python_requires=">=3.8",
) 