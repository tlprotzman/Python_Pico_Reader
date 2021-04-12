from setuptools import setup

setup(
    name="pico_reader",
    version="0.1",
    description="A reader for STAR's PicoDST file format",
    packages=["pico_reader"],
    install_requires=["numpy", "awkward", "matplotlib", "uproot"]
)