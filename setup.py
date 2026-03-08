from setuptools import setup, find_packages


setup(
    name="aizenx-xai",
    version="0.1.0",
    description="AizenX - Explainable AI Toolkit for model interpretability, fairness and debugging",
    author="Bhageeratha Reddy",
    packages=find_packages(),

    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "plotly",
        "fastapi",
        "uvicorn",
        "python-multipart",
        "streamlit",
        "networkx"
    ],

    entry_points={
        "console_scripts": [
            "aizenx=aizenx.cli:main"
        ]
    },

    python_requires=">=3.9",

    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)