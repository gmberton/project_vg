# Project of Visual Geo-localization

This repository provides a ready-to-use visual geo-localization (VG) pipeline, which you can use to train a model on a given dataset.
Specifically, it implements a ResNet-18 followed by an average pooling, which can be trained on VG datasets such as Pitts30k, using negative mining and triplet loss as explained in the [NetVLAD paper](https://arxiv.org/abs/1511.07247).
You will have to replace the average pooling with a GeM layer and a NetVLAD layer.


# Datasets
We provide the datasets of [Pitts30k](https://drive.google.com/file/d/1QpF5nO1SivJ5QOx1kkhoCeMqFvvrksey/view?usp=sharing) and [St Lucia](https://drive.google.com/file/d/1nEmjnEePTQNdB0JdKFE8ISJMcfbZMPPZ/view?usp=sharing])

About the datasets formatting, the adopted convention is that the names of the files with the images are:

@ UTM_easting @ UTM_northing @ UTM_zone_number @ UTM_zone_letter @ latitude @ longitude @ pano_id @ tile_num @ heading @ pitch @ roll @ height @ timestamp @ note @ extension

Note that some of these values can be empty (e.g. the timestamp might be unknown), and the only required values are UTM coordinates (obtained from latitude and longitude).


# Getting started
To get started first download the repository

```git clone https://github.com/gmberton/project_vg```

then download Pitts30k [(link)](https://drive.google.com/file/d/1QpF5nO1SivJ5QOx1kkhoCeMqFvvrksey/view?usp=sharing), and extract the zip file.
Then install the required packages

 ```pip install -r requirements.txt```

and finally run

```python3 train.py --datasets_folder path/to/folder/containing/pitts30k```

This will train, validate, and test the model on Pitts30k.
If the previous steps were executed correctly, training will take only a few of hours, and end with a recall@1 (R@1) of roughly 60%. Results should heavily improve using GeM or NetVLAD layer.

To visualize all the parameters that it is possible to set, run 

```python3 train.py -h```

