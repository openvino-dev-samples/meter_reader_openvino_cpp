# meter_reader_openvino_cpp

This repository shows how to create an industrial meter reader with OpenVINO cpp runtime.
## 1 Install requirements
Please follow the Guides to install [OpenVINO](https://docs.openvino.ai/latest/openvino_docs_install_guides_installing_openvino_linux.html#install-openvino) and [OpenCV](https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html)

## 2 Download and extract pre-trained model from [PPYOLOv2](https://bj.bcebos.com/paddlex/examples2/meter_reader/meter_det_model.tar.gz) for detection and [DeeplabV3+](https://bj.bcebos.com/paddlex/examples2/meter_reader/meter_seg_model.tar.gz) for segmentation.

## 3 Convert the pdmodel to OpenVINO IR
```shell
  $ mo --input_model meter_det_model/model.pdmodel
  $ mo --input_model meter_seg_model/model.pdmodel
 ```

## 4 Compile the source code
```shell
  $ cd ~/meter_reader_openvino_cpp
  modify the `CMakeLists.txt` according to your local environment
  $ mkdir build && cd build
  $ cmake ..
  $ make
 ```

## 5 Run inference
 ```shell
  $ meter_reader meter_det_model/model.xml meter_seg_model/model.xml data/horses.jpg data/text.jpg
 ```
## 6 Results
the image with inference results will be saved locally.
[results](https://user-images.githubusercontent.com/91237924/189855947-75b368b9-4680-4cc9-a013-f5f84bb170a2.jpg)
