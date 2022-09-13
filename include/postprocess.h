#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <vector>
#include "include/meter_config.h"
using namespace std;
using namespace cv;

bool Erode(const int32_t &kernel_size,
           const vector<Mat> &seg_results,
           vector<vector<uint8_t>> *seg_label_maps);

bool CircleToRectangle(
    const vector<uint8_t> &seg_label_map,
    vector<uint8_t> *rectangle_meter);

bool RectangleToLine(const vector<uint8_t> &rectangle_meter,
                     vector<int> *line_scale,
                     vector<int> *line_pointer);

bool MeanBinarization(const vector<int> &data,
                      vector<int> *binaried_data);

bool LocateScale(const vector<int> &scale,
                 vector<float> *scale_location);

bool LocatePointer(const vector<int> &pointer,
                   float *pointer_location);

bool GetRelativeLocation(
    const vector<float> &scale_location,
    const float &pointer_location,
    MeterResult *result);

bool CalculateReading(const MeterResult &result,
                      float *reading);

bool PrintMeterReading(const vector<float> &readings);

bool Visualize(Mat &img,
               vector<Rect> &detected_objects,
               const vector<float> &readings);

bool GetMeterReading(
    const vector<vector<uint8_t>> &seg_label_maps,
    vector<float> *readings);