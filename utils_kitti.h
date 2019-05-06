#pragma once
#ifndef UTILS_KITTI_H
#define UTILS_KITTI_H
#include <opencv2/opencv.hpp>

class UtilsKitti
{
public:
    UtilsKitti();
    void ReadData(const std::string& img_pattern, 
                  const std::string& csv_path_vel,
                  const std::string& csv_path_tf,
                  std::vector<cv::Mat>& imgs, 
                  std::vector<double>& imgtimes, 
                  std::vector<cv::Point3f>& linear_v, 
                  std::vector<cv::Point3f>& angular_v,
                  std::vector<cv::Mat>& Tcws);
    cv::Mat GetK0926(){ return this->mK0926; }
    cv::Mat GetK0929(){ return this->mK0929; }
    
private:
    //read imgs
    void ReadImgs(const std::string& pattern, std::vector<cv::Mat>& imgs, std::vector<double>& imgtimes); 
    //read velocity and change to camera frame
    void ReadOXTsVel(const std::string& csv_path, std::vector<cv::Point3f>& linear_v, std::vector<cv::Point3f>& angular_v); 
    //ground truth and change to camera frame
    void ReadTF(const std::string& csv_path, std::vector<cv::Mat>& Tcws);

private:
    cv::Mat mT_cam_imu;
    cv::Mat mR_cam_imu;
    cv::Mat mK0926;
    cv::Mat mK0929;
};
#endif //UTILS_KITTI_H