
#include <iostream>
#include <vector>
#include <utility>
#include <limits>
#include <cmath>
#include <string>
#include "include/postprocess.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/core.hpp>

#include "include/postprocess.h"

bool Erode(const int32_t &kernel_size,
           const vector<Mat> &seg_results,
           vector<vector<uint8_t>> *seg_label_maps) {
  Mat kernel(kernel_size, kernel_size, CV_8U, Scalar(1));
  for (auto mask : seg_results) {
    erode(mask, mask, kernel);
    imwrite("2.jpg", mask);
    vector<uint8_t> map;
    if (mask.isContinuous()) {
        map.assign(mask.data, mask.data + mask.total() * mask.channels());
    } else {
      for (int r = 0; r < mask.rows; r++) {
        map.insert(map.end(),
                   mask.ptr<int64_t>(r),
                   mask.ptr<int64_t>(r) + mask.cols * mask.channels());
      }
    }
    seg_label_maps->push_back(map);
  }
  return true;
}


bool CircleToRectangle(
  const vector<uint8_t> &seg_label_map,
  vector<uint8_t> *rectangle_meter) {
  float theta;
  int rho;
  int image_x;
  int image_y;

  // The minimum scale value is at the bottom left, the maximum scale value
  // is at the bottom right, so the vertical down axis is the starting axis and
  // rotates around the meter ceneter counterclockwise.
  *rectangle_meter =
    vector<uint8_t> (RECTANGLE_WIDTH * RECTANGLE_HEIGHT, 0);
  for (int row = 0; row < RECTANGLE_HEIGHT; row++) {
    for (int col = 0; col < RECTANGLE_WIDTH; col++) {
      theta = PI * 2 / RECTANGLE_WIDTH * (col + 1);
      rho = CIRCLE_RADIUS - row - 1;
      int y = static_cast<int>(CIRCLE_CENTER[0] + rho * cos(theta) + 0.5);
      int x = static_cast<int>(CIRCLE_CENTER[1] - rho * sin(theta) + 0.5);
      (*rectangle_meter)[row * RECTANGLE_WIDTH + col] =
        seg_label_map[y * METER_SHAPE[1] + x];
    }
  }

  return true;
}

bool RectangleToLine(const vector<uint8_t> &rectangle_meter,
                     vector<int> *line_scale,
                     vector<int> *line_pointer) {
  // Accumulte the number of positions whose label is 1 along the height axis.
  // Accumulte the number of positions whose label is 2 along the height axis.
  (*line_scale) = vector<int> (RECTANGLE_WIDTH, 0);
  (*line_pointer) = vector<int> (RECTANGLE_WIDTH, 0);
  for (int col = 0; col < RECTANGLE_WIDTH; col++) {
    for (int row = 0; row < RECTANGLE_HEIGHT; row++) {
        if (rectangle_meter[row * RECTANGLE_WIDTH + col] ==
          static_cast<uint8_t>(SEG_CNAME2CLSID["pointer"])) {
            (*line_pointer)[col]++;
        } else if (rectangle_meter[row * RECTANGLE_WIDTH + col] ==
          static_cast<uint8_t>(SEG_CNAME2CLSID["scale"])) {
            (*line_scale)[col]++;
        }
    }
  }
  return true;
}

bool MeanBinarization(const vector<int> &data,
                      vector<int> *binaried_data) {
  int sum = 0;
  float mean = 0;
  for (auto i = 0; i < data.size(); i++) {
    sum = sum + data[i];
  }
  mean = static_cast<float>(sum) / static_cast<float>(data.size());

  for (auto i = 0; i < data.size(); i++) {
    if (static_cast<float>(data[i]) >= mean) {
      binaried_data->push_back(1);
    } else {
      binaried_data->push_back(0);
    }
  }
  return  true;
}

bool LocateScale(const vector<int> &scale,
                 vector<float> *scale_location) {
  float one_scale_location = 0;
  bool find_start = false;
  int one_scale_start = 0;
  int one_scale_end = 0;

  for (int i = 0; i < RECTANGLE_WIDTH; i++) {
    // scale location
    if (scale[i] > 0 && scale[i + 1] > 0) {
      if (!find_start) {
        one_scale_start = i;
        find_start = true;
      }
    }
    if (find_start) {
      if (scale[i] == 0 && scale[i + 1] == 0) {
          one_scale_end = i - 1;
          one_scale_location = (one_scale_start + one_scale_end) / 2.;
          scale_location->push_back(one_scale_location);
          one_scale_start = 0;
          one_scale_end = 0;
          find_start = false;
      }
    }
  }
  return true;
}

bool LocatePointer(const vector<int> &pointer,
                   float *pointer_location) {
  bool find_start = false;
  int one_pointer_start = 0;
  int one_pointer_end = 0;

  for (int i = 0; i < RECTANGLE_WIDTH; i++) {
    // pointer location
    if (pointer[i] > 0 && pointer[i + 1] > 0) {
      if (!find_start) {
        one_pointer_start = i;
        find_start = true;
      }
    }
    if (find_start) {
      if ((pointer[i] == 0) && (pointer[i+1] == 0)) {
        one_pointer_end = i - 1;
        *pointer_location = (one_pointer_start + one_pointer_end) / 2.;
        one_pointer_start = 0;
        one_pointer_end = 0;
        find_start = false;
        break;
      }
    }
  }
  return true;
}

bool GetRelativeLocation(
  const vector<float> &scale_location,
  const float &pointer_location,
  MeterResult *result) {
  int num_scales = static_cast<int>(scale_location.size());
  result->num_scales_ = num_scales;
  result->pointed_scale_ = -1;
  if (num_scales > 0) {
    for (auto i = 0; i < num_scales - 1; i++) {
      if (scale_location[i] <= pointer_location &&
            pointer_location < scale_location[i + 1]) {
        result->pointed_scale_ = i + 1 +
          (pointer_location-scale_location[i]) /
          (scale_location[i+1]-scale_location[i] + 1e-05);
      }
    }
  }
  return true;
}

bool CalculateReading(const MeterResult &result,
                      float *reading) {
  // Provide a digital readout according to point location relative
  // to the scales
  if (result.num_scales_ > TYPE_THRESHOLD) {
    *reading = result.pointed_scale_ * METER_CONFIG[0].scale_interval_value_;
  } else {
    *reading = result.pointed_scale_ * METER_CONFIG[1].scale_interval_value_;
  }
  return true;
}

bool PrintMeterReading(const vector<float> &readings) {
  for (auto i = 0; i < readings.size(); ++i) {
    cout << "Meter " << i + 1 << ": " << readings[i] << endl;
  }
  return true;
}

bool Visualize(Mat &img,
               vector<Rect> &detected_objects,
               const vector<float> &readings)
{
  for (auto i = 0; i < detected_objects.size(); i++)
  {
    rectangle(img, Point(detected_objects[i].tl()), Point(detected_objects[i].br()), Scalar(0, 255, 0), 2);
    string label = format("%.3f", readings[i]);
    putText(img, label, Point(detected_objects[i].tl()), FONT_HERSHEY_SIMPLEX, 1.5, Scalar(0, 255, 0), 2);
  }
  imwrite("results.jpg", img);
  return true;
}

bool GetMeterReading(
  const vector<vector<uint8_t>> &seg_label_maps,
  vector<float> *readings) {
  for (auto i = 0; i < seg_label_maps.size(); i++) {
    vector<uint8_t> rectangle_meter;
    CircleToRectangle(seg_label_maps[i], &rectangle_meter);

    vector<int> line_scale;
    vector<int> line_pointer;
    RectangleToLine(rectangle_meter, &line_scale, &line_pointer);

    vector<int> binaried_scale;
    MeanBinarization(line_scale, &binaried_scale);
    vector<int> binaried_pointer;
    MeanBinarization(line_pointer, &binaried_pointer);

    vector<float> scale_location;
    LocateScale(binaried_scale, &scale_location);

    float pointer_location;
    LocatePointer(binaried_pointer, &pointer_location);

    MeterResult result;
    GetRelativeLocation(
      scale_location, pointer_location, &result);

    float reading;
    CalculateReading(result, &reading);
    readings->push_back(reading);
  }
  return true;
}