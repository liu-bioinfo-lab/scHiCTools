import setuptools

with open("README.md", "rb") as fh:
    long_description = fh.read()


def get_requirements(path):
    with open(path, "r") as fh:
        content = fh.read()
    return [
        req
        for req in content.split("\n")
        if req != '' and not req.startswith('#')
    ]

install_requires = get_requirements('requirements.txt')


setuptools.setup(
    name="scHiCTools",
    version="0.0.1",
    author="Fan Feng",
    author_email="fanfeng@umich.edu",
    description="A user-friendly package for processing single cell HiC data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/liu-bioinfo-lab/scHiCTool",
    packages=setuptools.find_packages(exclude=['test']),
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
