# MTCNN CPP

MTCNN
Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks.

This project is forked from: https://github.com/wowo200/MTCNN . Which itself is a C++ version of: https://github.com/kpzhang93/MTCNN_face_detection_alignment

The models used here are trained and copied from the original project (https://github.com/kpzhang93/MTCNN_face_detection_alignment).


This fork fixes some bugs over the original C++ version. Moreover, the source code layout is changed so that it can be used as 3rd party from other libraries.

### Prerequisites

- OpenCV (tested with 3.4)
- Caffe (built with CUDA -- tested )
- Boost (tested with 1.66)

### Install
 
 ```
 $ cd projects
 $ git clone https://github.com/vvirag/MTCNN.git
 $ cd MTCNN
 $ cmake-gui
 ```
 After selecting source and build directories, click ```Configure```.
 If Caffe not found, define the path manually by setting up ```Caffe_INCLUDE_DIRS``` and ```Caffe_LIBS```.
 Afterwards click ```Generate```.
 
```
$ cd build
$ make
```

### Run
```
$ cd projects/MTCNN
$ ./build/example/main.bin ../path_to_image_folder_containing_jpgs_or_jpegs/
```

