#include "include/detector.h"
#include "include/segmenter.h"
#include "include/postprocess.h"

int main(int argc, char *argv[])
{
    const string detect_model_path{argv[1]};
    const string segment_model_path{argv[2]};
    const char *image_path{argv[3]};

    vector<Rect> detected_objects;
    vector<Mat> segment_results;
    vector<Mat> masks;
    vector<float> readings;

    Mat src_img = imread(image_path);

    Detector detector;
    detector.init(detect_model_path, 0.9);
    Segmenter segmenter;
    segmenter.init(segment_model_path);

    detector.run(src_img, detected_objects);
    for (auto detected_object : detected_objects)
    {
        segment_results.push_back(src_img(detected_object));
    }
    segmenter.run(segment_results, masks);

    vector<vector<uint8_t>> seg_label_maps;
    Erode(4, masks, &seg_label_maps);
    // The postprocess are done to get the reading or each meter
    GetMeterReading(seg_label_maps, &readings);
    Visualize(src_img, detected_objects, readings);
    PrintMeterReading(readings);
}