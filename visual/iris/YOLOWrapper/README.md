# YOLOWrapper
this project implement a C++ Wrapper for YOLOv3, which will be used for iris detection. The location information will be used for biometric identification in latter projects.

## How to test the detector on objects of common categories

### download weights trained on MS COCO
download with the following commands
```Shell
wget -P models https://pjreddie.com/media/files/yolov3.weights
wget -P models https://pjreddie.com/media/files/yolov3-tiny.weights
```

### build everything
build with the following commands
```Shell
make -C darknet
make
```

### test detector on street view images
test YOLOv3 with the following commands
```Shell
make run
```

