# Oriented Object Detection

Framework for processing oriented object detection (OOD) data and evaluating OOD models

This repository can be installed like so:
```bash
git clone https://github.com/UACamgianTeam/Oriented-Object-Detection.git ood_repo/
cd ood_repo/
pip install .
```
After that, one can import it in other projects via `import ood`.

## Dependencies
- TensorFlow 2
- TensorFlow Object Detection API (compiled from source)
- nms
- Shapely
- Matplotlib

## Do Commit to this Repository
Preprocessing and evaluation code (for oriented or regular object detection)

## Don't Commit to this Repository
- Actual ML code. This repository is really meant to house utility functions.
- Actual image data or labels
- Code that isn't strictly related object detection. If you write something cool, make a new repo.

