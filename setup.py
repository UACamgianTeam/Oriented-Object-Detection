import setuptools

setuptools.setup(
    name="oriented-object-detection",
    version="0.0.1a",
    author="Alexander Lay, Ethan Mines",
    packages=["ood","ood.utils","ood.evaluate","ood.preprocess"],
    install_requires=[
        "pycocotools",
        "object-detection",
        "tensorflow",
        "nms",
        "matplotlib",
        "shapely"
    ]
)
