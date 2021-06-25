import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="df3dbehav",
    version="0.1",
    author="Semih Gunel",
    packages=["df3dbehav"],
    entry_points={"console_scripts": ["df3dbehav = df3dbehav.df3dbehav:main"]},
    description="Behavior Estimation on DeepFly3D annotations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/semihgunel/Df3dBehav"
)