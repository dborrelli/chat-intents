import setuptools

setuptools.setup(
    name="chatintents",
    version="0.1",
    packages=setuptools.find_packages(exclude=['data', 'images', 'notebooks']),
)