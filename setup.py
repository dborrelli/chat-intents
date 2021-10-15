import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="chatintents",
    version="0.0.1",
    packages=["chatintents"],
    author="David Borrelli",
    description="ChatIntents automatically clusters and labels short text intent messages.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dborrelli/chat-intents",
    keywords="nlp clustering document embeddings unsupervised intents",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy >= 1.19.2',
        'pandas >= 1.2.0',
        'umap-learn >= 0.5.1',
        'hdbscan >= 0.8.27',
        'hyperopt >= 0.2.5',
        'matplotlib >= 3.4.2',
        'spacy >= 3.1.2',
        'ipywidgets >= 7.6.3',
    ],
    python_requires=">=3.9",

)
