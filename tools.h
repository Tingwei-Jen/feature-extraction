#pragma once
#ifndef TOOL_H
#define TOOL_H

#include <opencv2/opencv.hpp>

class Tools
{
public:
    static cv::Mat Quaternion2RotM(const float& x, const float& y, const float& z, const float& w);
    static cv::Point3f RotM2FixedAngle(const cv::Mat& rotM);
    static cv::Mat FixedAngle2RotM(const cv::Point3f& fixedAngle);
};
#endif //TOOL_H