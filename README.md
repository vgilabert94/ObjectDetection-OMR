# Object Detection of handwritten music documents using deep learning (RetinaNET)


I used Anaconda with an enviorment named retina. Enter on this enviorment: `activate retina` and place ourselves in the folder that we are going to work with `cd C:/tfg`. In this folfer im going to install RetinaNET.

### Installation RetinaNET

I use the RetinaNET model implementated in keras. For use RetinaNET-Keras, follows the steps that they provide in his repository:

1) Clone this repository. (https://github.com/fizyr/keras-retinanet)
2) Ensure numpy is installed using `pip install numpy`
3) In the repository, execute `pip install . --user`
4) Run this command to compile Cython code: `python setup.py build_ext --inplace` 

### Annotations format
The CSV file with annotations should contain one annotation per line.
Images with multiple bounding boxes should use one row per bounding box.
Note that indexing for pixel values starts at 0.
The expected format of each line is:
```
path/to/image.jpg,x1,y1,x2,y2,class_name
```
Some images may not contain any labeled objects.
To add these images to the dataset as negative examples,
add an annotation where `x1`, `y1`, `x2`, `y2` and `class_name` are all empty:
```
path/to/image.jpg,,,,,
```
My annotations files:
train_labels.csv
test_labels.csv

### Class mapping format
The class name to ID mapping file should contain one mapping per line.
Each line should use the following format:
```
class_name,id
```
For example:
```
staff,0
lyrics,1
empty_staff,2
```
My class file:
classes.csv


### Anchor optimization

The object that want to detect In some cases, the default anchor configuration is not suitable for detecting objects in your dataset, for example, if your objects are smaller than the 32x32px (size of the smallest anchors). In this case, it might be suitable to modify the anchor configuration, this can be done automatically by following the steps in the [anchor-optimization](https://github.com/martinzlocha/anchor-optimization/) repository. To use the generated configuration check [here](https://github.com/fizyr/keras-retinanet-test-data/blob/master/config/config.ini) for an example config file and then pass it to `train.py` using the `--config` parameter.


