#pragma once
#ifndef MATCHER_H
#define MATCHER_H
#include "frame.h"
#include <opencv2/opencv.hpp>

class Matcher
{
public:
    Matcher();
    int SearchForInitialization(Frame& f1, Frame& f2, std::vector<int>& vMatchesQIdx, std::vector<int>& vMatchesTIdx);
    int SearchForProjection(Frame& last, Frame& current, std::vector<int>& vMatchesQIdx, std::vector<int>& vMatchesTIdx);
    int SearchForCreateNewMatches(Frame& last, Frame& current, std::vector<int>& vMatchesQIdx, std::vector<int>& vMatchesTIdx);

private:    
    // Px velocity due to camera angular velocity and under camera frame
    cv::Point3f MotionFieldRotation(const cv::Point3f& camAngularVel, const cv::Point3f& pCam);
    // Px velocity due to camera linear velocity and under camera frame
    cv::Point3f MotionFieldTranslation(const cv::Point3f& camLinearVel, const float& depth, const cv::Point3f& pCam);

    // Get epipolar line from two camera pose
	cv::Point3f GetEpipolarLine(Frame& f1, Frame& f2, const cv::Point2f& px1);
    // Get epipolar line from prediction of frame1 velocity
    cv::Point3f GetEpipolarLinePred(Frame& f1, Frame& f2, const cv::Point2f& px1);
   
    // Compute fundamental matrix of frame1 and frame2
    cv::Mat ComputeF21(const cv::Mat& R21, const cv::Mat& t21, const cv::Mat& K);
	// Epipole in frame2 by ground truth
    cv::Point2f GetEpipole(Frame& f1, Frame& f2);
	// FOE in frame1 by velocity
    cv::Point2f GetFOE(Frame& f1);

    // Triangulation method to recover depth
    bool Triangulation(Frame& f1, Frame& f2, const cv::Point2f& px1, const cv::Point2f& px2, cv::Point3f& x3Dp);

private:
    float mDistFOETH;
    float mDepthMax;
    float mDepthMin;
    float mWindowSizeMaxX;
    float mWindowSizeMinX;
    float mWindowSizeMaxY;
    float mWindowSizeMinY;   
    float mDepthMaxTH;                              //ignore the feature point whose depth over this threshold 
    float mEpipolarLineTH;
    float mDescriptorDistTH;
    float mDescriptorMinDistTH;
    float mNNratio;
};
#endif //MATCHER_H