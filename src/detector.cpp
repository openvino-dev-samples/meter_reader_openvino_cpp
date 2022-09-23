#include "include/detector.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <openvino/openvino.hpp>

Detector::Detector() {}

Detector::~Detector() {}

bool Detector::init(string model_path, double threshold)
{
    _model_path = model_path;
    _threshold = threshold;
    ov::Core core;
    shared_ptr<ov::Model> model = core.read_model(_model_path);
    map<string, ov::PartialShape> name_to_shape;
    name_to_shape["image"] = ov::PartialShape{1, 3, 608, 608};
    name_to_shape["im_shape"] = ov::PartialShape{1, 2};
    name_to_shape["scale_factor"] = ov::PartialShape{1, 2};
    model->reshape(name_to_shape);
    ov::CompiledModel detect_model = core.compile_model(model, "CPU");
    detect_infer_request = detect_model.create_infer_request();
    return true;
}

bool Detector::run(Mat &src_img, vector<Rect> &detected_objects)
{
    int total_num = 22743;
    float mean[3] = {0.485, 0.456, 0.406};
    float std[3] = {0.229, 0.224, 0.225};
    float height = src_img.rows;
    float width = src_img.cols;
    float scale_x = width / 608 * 2;
    float scale_y = height / 608;

    Mat img;
    resize(src_img, img, Size(608, 608));
    ov::Tensor input_tensor0 = detect_infer_request.get_tensor("im_shape");
    ov::Tensor input_tensor1 = detect_infer_request.get_tensor("image");
    ov::Tensor input_tensor2 = detect_infer_request.get_tensor("scale_factor");
    // nhwc -> nchw
    auto data1 = input_tensor1.data<float>();
    for (int h = 0; h < 608; h++)
    {
        for (int w = 0; w < 608; w++)
        {
            for (int c = 0; c < 3; c++)
            {
                int out_index = c * 608 * 608 + h * 608 + w;
                data1[out_index] = float(((float(img.at<Vec3b>(h, w)[c]) / 255.0f) - mean[c]) / std[c]);
            }
        }
    }

    auto data0 = input_tensor0.data<float>();
    data0[0] = 608;
    data0[1] = 608;

    auto data2 = input_tensor2.data<float>();
    data2[0] = 1;
    data2[1] = 2;

    //start inference
    detect_infer_request.infer();

    //extract the output data
    auto output = detect_infer_request.get_output_tensor(0);
    const float *result = output.data<const float>();
    for (int num = 0; num < total_num; num++)
    {
        auto box_prob = result[num * 6 + 1];
        if (box_prob > _threshold && box_prob <= 1)
        {
            float x0 = result[num * 6 + 2] * scale_x;
            float y0 = result[num * 6 + 3] * scale_y;
            float x1 = result[num * 6 + 4] * scale_x;
            float y1 = result[num * 6 + 5] * scale_y;
            Rect rect = Rect(round(x0), round(y0), round(x1 - x0), round(y1 - y0));
            detected_objects.push_back(rect);
        }
    }

    return true;
}