# Master-Thesis
![alt text](http://url/to/img.png)
This repository represents my master thesis for the spring of 2021.
The python version required for this repo is `>=3.5, <3.8`, and version used is `3.7.9`

## Setup
### Zivid SDK
To use the Zivid camera you need to download and install the "Zivid core" package. Zivid SDK Version
Follow the guide here [zivid-python](https://github.com/zivid/zivid-python) to install the SDK and zivid-pyhton.

Then clone this repo.

### Package Dependicies
To install the package dependicies
```python

pip3 install -r requirements.txt
```

I did have issue with installing the OpenExr package, so the package is build manually from the the folder `whl/whl/OpenEXR-1.3.2-cp37-cp37m-win_amd64.whl` as in `requirements.txt`. Better integration for this could be to read the `.exr` file with `cv2.imread()` in to the [function](https://github.com/eivindtn/Master-Thesis/blob/5ddeaffd9e485bce62d68658d5bdabeacb657fdd/blender/geometry_vision.py#L165).

## Example
### Synthetic
The synthetic experiment depend on the third part software Blender API. 
For rendering new point clouds, look into the scene in the [folder](https://github.com/eivindtn/Master-Thesis/tree/main/blender/scenes)
A test example can be runned from runing the script in this [folder](https://github.com/eivindtn/Master-Thesis/tree/main/blender):
```python
python blender_projection-v2.py
```
### Lab
The lab use a Zivid Two camera for capturing point clouds. A test example can be runned from running this script in the [folder](https://github.com/eivindtn/Master-Thesis/tree/main/lab):

```python
python lab_projection.py
```