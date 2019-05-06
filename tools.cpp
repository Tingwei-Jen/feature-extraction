#include "tools.h"

using namespace cv;

cv::Mat Tools::Quaternion2RotM(const float& x, const float& y, const float& z, const float& w)
{
    float xx = x*x;
    float yy = y*y;
    float zz = z*z;

    float wx = w*x;
    float wy = w*y;
    float wz = w*z;

    float xy = x*y;
    float xz = x*z;
    float yz = y*z;

    Mat R = ( Mat_<float> ( 3,3 ) <<
                ( 1 - 2*yy - 2*zz ), ( 2*xy - 2*wz ), ( 2*xz + 2*wy ),
                ( 2*xy + 2*wz ), ( 1 - 2*xx - 2*zz ), ( 2*yz - 2*wx ),
                ( 2*xz - 2*wy ), ( 2*yz + 2*wx ), ( 1 - 2*xx - 2*yy ));

    return R;
}

cv::Point3f Tools::RotM2FixedAngle(const cv::Mat& rotM)
{
    float yaw, pitch, roll;
    yaw = atan2(rotM.at<float>(1,0), rotM.at<float>(0,0));
    pitch = atan2(-rotM.at<float>(2,0), sqrt(rotM.at<float>(2,1)*rotM.at<float>(2,1)+rotM.at<float>(2,2)*rotM.at<float>(2,2)));
    roll = atan2(rotM.at<float>(2,1), rotM.at<float>(2,2));
    return Point3f(roll, pitch, yaw);
}

cv::Mat Tools::FixedAngle2RotM(const cv::Point3f& fixedAngle)
{
    Mat Rx = Mat(3,3, CV_32F);
    Mat Ry = Mat(3,3, CV_32F);
    Mat Rz = Mat(3,3, CV_32F);

    Rx.at<float>(0,0) = 1.0;    Rx.at<float>(0,1) = 0.0;    Rx.at<float>(0,2) = 0.0; 
    Rx.at<float>(1,0) = 0.0;    Rx.at<float>(1,1) = cos(fixedAngle.x);  Rx.at<float>(1,2) = -sin(fixedAngle.x); 
    Rx.at<float>(2,0) = 0.0;    Rx.at<float>(2,1) = sin(fixedAngle.x);  Rx.at<float>(2,2) = cos(fixedAngle.x);

    Ry.at<float>(0,0) = cos(fixedAngle.y);  Ry.at<float>(0,1) = 0.0;    Ry.at<float>(0,2) = sin(fixedAngle.y);
    Ry.at<float>(1,0) = 0.0;                 Ry.at<float>(1,1) = 1.0;    Ry.at<float>(1,2) = 0.0;
    Ry.at<float>(2,0) = -sin(fixedAngle.y); Ry.at<float>(2,1) = 0.0;    Ry.at<float>(2,2) = cos(fixedAngle.y);

    Rz.at<float>(0,0) = cos(fixedAngle.z);  Rz.at<float>(0,1) = -sin(fixedAngle.z); Rz.at<float>(0,2) = 0.0;
    Rz.at<float>(1,0) = sin(fixedAngle.z);  Rz.at<float>(1,1) = cos(fixedAngle.z);  Rz.at<float>(1,2) = 0.0;
    Rz.at<float>(2,0) = 0.0;                 Rz.at<float>(2,1) = 0.0;                 Rz.at<float>(2,2) = 1.0;

    return Rx*Ry*Rz;
}