from setuptools import setup, find_packages

setup(
    name="SFQCcontrol",
    version="1.0",
    author="Chenyin Sun",
    author_email="sunchenyin@mail.ustc.edu.cn",
    description="SFQ control package",
    packages=["SFQControl"],
    install_requires=['numpy', 'progressbar']
)
