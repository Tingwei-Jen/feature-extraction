#pragma once
#ifndef UTILS_H
#define UTILS_H
#include <opencv2/opencv.hpp>

class Utils
{
public:
    Utils();
    void ReadData();
    cv::Mat GetK(){ return this->mK; }

private:
    void ReadImgAndTime(const std::string& pattern, const int& idx, std::vector<std::string>& imgtimes, std::vector<cv::Mat>& imgs);
    void ReadCameraPose(const std::string& csv_path, const std::vector<std::string>& imgtimes, std::vector<cv::Mat>& Twcs);
    void ReadOdometry(const std::string& csv_path, const std::vector<std::string>& imgtimes, 
                            std::vector<cv::Mat>& Twcs, std::vector<cv::Point3f>& LinearVels);
    void ReadIMUGyro(const std::string& csv_path, const std::vector<std::string>& imgtimes, std::vector<cv::Point3f>& AngularVels);
    void Pose2PosAndAngle(const std::vector<cv::Mat>& Twcs, std::vector<cv::Point3f>& Pos, std::vector<cv::Point3f>& Ang);
    
    void ReadCameraPose(const std::string& csv_path, std::vector<cv::Mat>& Twcs);
    void ReadOdometry(const std::string& csv_path, std::vector<cv::Mat>& Twcs, std::vector<cv::Point3f>& LinearVels);

private:
    cv::Mat mK;

};
#endif //UTILS_H