# Object Detection of handwritten music documents using deep learning (RetinaNET)

Im going to train Keras-RetinaNET model to detect objets in handwritten music documents. 

I was using Google Colab connected to My Google Drive [Google Drive](https://drive.google.com/drive/folders/1oBDuIsCN0bmCj0MlGA7CfsqMHOKyHq8p) to store data. You can follow the installation, traning, convert model and evaluation in: "RetinaNET Keras.ipynb".
The annotation procces I used MATLAB Image Labeler and converting to CSV files with python script.

### Installation RetinaNET

I use the RetinaNET model implementated in keras. For install correctly RetinaNET-Keras, follows the steps that they provide in his repository:

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

My class file:
classes.csv


### Anchor optimization

My annotated objects are very long (aspect ratio). we can't use the predefined anchor boxes. I need to optimize.
This can be done automatically by following the steps in the [anchor-optimization](https://github.com/martinzlocha/anchor-optimization/) repository. 

When I have the optimization anchors boxes values, I created a config.ini file with this values.
Path: config/config.ini


### Results

Some examples using retinanet-evaluate with IoU=0.5 and IoU=0.7: 

#### IoU=0.5
```
267 instances of class staff with average precision: 0.9915
264 instances of class lyrics with average precision: 0.8005
29 instances of class empty_staff with average precision: 0.8276
mAP using the weighted average of precisions among classes: 0.8930
mAP: 0.8732
```

#### IoU=0.7
```
267 instances of class staff with average precision: 0.9090
264 instances of class lyrics with average precision: 0.5846
29 instances of class empty_staff with average precision: 0.8276
mAP using the weighted average of precisions among classes: 0.7519
mAP: 0.7737
```
#### Images output

Green= GT ; Red = staff ; orange = lyrics ; blue = empty_staff

<p align="center">
  <img src="https://github.com/vgilabert94/ObjectDetection-OMR/blob/master/results/11.png" width="600" height="400" alt="Example result of RetinaNet on MS COCO"/>
  <img src="https://github.com/vgilabert94/ObjectDetection-OMR/blob/master/results/40.png" width="600" height="400" alt="Example result of RetinaNet on MS COCO"/>
  <img src="https://github.com/vgilabert94/ObjectDetection-OMR/blob/master/results/44.png" width="600" height="400" alt="Example result of RetinaNet on MS COCO"/>
</p>
