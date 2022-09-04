"""Setup."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2022(18588220928@163.com), All Rights Reserved.
# ***
# ***    File Author: Dell, 2022年 09月 04日 星期日 16:41:59 CST
# ***
# ************************************************************************************/
#

from setuptools import setup

with open("README.md", "r") as file:
    long_description = file.read()

setup(
    name="image_icolor",
    version="1.0.0",
    author="Dell Du",
    author_email="18588220928@163.com",
    description="image icolor package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/delldu/iColoriT.git",
    packages=["image_icolor"],
    package_data={"image_icolor": ["models/image_icolor.pth"]},
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "torch >= 1.9.0",
        "torchvision >= 0.10.0",
        "Pillow >= 7.2.0",
        "numpy >= 1.19.5",
        "einops >= 0.3.0",
        "redos >= 1.0.0",
        "todos >= 1.0.0",
    ],
)
