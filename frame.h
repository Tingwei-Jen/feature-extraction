#pragma once
#ifndef FRAME_H
#define FRAME_H
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

#define FRAME_GRID_ROWS 24
#define FRAME_GRID_COLS 32

class Frame
{
public:
    Frame();

    //Copy
    Frame(const Frame& frame);

    // Constructor for Monocular cameras.
	Frame(const cv::Mat& img, const cv::Mat &K, const float& timeStamp, const cv::Ptr<cv::xfeatures2d::SURF>& Detector);
	
    void SetPose(const cv::Mat& Tcw);
    void SetLinearVelocity(const cv::Point3f& linearVel);
    void SetAngularVelocity(const cv::Point3f& angularVel);
    void AssignFeaturesToGrid();

	cv::Mat GetPose(){ return mTcw.clone(); }
    cv::Mat GetCameraCenter(){ return mOw.clone(); }
    cv::Mat GetRotation(){ return mRcw.clone(); }
    cv::Mat GetTranslation(){ return mtcw.clone(); }
    cv::Mat GetRotationInverse(){ return mRwc.clone(); }
    std::vector<int> GetFeaturesInArea(const float& x, const float& y, const float& lx, const float& ly);
    float GetAverageDepthInArea(const float& x, const float& y);



    cv::Point2f Cam2Px(const cv::Point3f& pCam);
    cv::Point3f Px2Cam(const cv::Point2f& px);
    cv::Point3f World2Cam(const cv::Point3f& pWorld);
    cv::Point3f Cam2World(const cv::Point3f& pCam);

public:
    static int frameCounter;
    static float fx;
    static float fy;
    static float cx;
    static float cy;
    static int width;
    static int height;
    static float mGridElementWidthInv;                              ///< 坐标乘以gridElementWidthInv和gridElementHeightInv就可以确定在哪个格子
    static float mGridElementHeightInv;

    static bool mbInitialComputations;
    
    int mId;
    cv::Mat mImg;
    cv::Mat mK;
    float mTimestamp;
    
    cv::Point3f mLinearVel;                                         ///< camera linear velocity in camera frame
    cv::Point3f mAngularVel;                                        ///< camera angular velocity in camera frame
    cv::Point2f mFocusOfExpansion;                                  ///< FOE of the frame based on camera linear velocity

    int N;                                                          ///< number of keypoints
    std::vector<cv::Point2f> mvKps;                                 ///< all key points in this frame
    cv::Mat mDescriptors;                                           ///< descriptors of key points in this frame
    std::vector<float> mvKpsDepth;                                  ///< depth of each key point in this frame   
    
    std::vector<int> mGrid[FRAME_GRID_COLS][FRAME_GRID_ROWS];       ///< 每个格子分配的特征点数，将图像分成格子，保证提取的特征点比较均匀

private:
    cv::Ptr<cv::xfeatures2d::SURF> mDetector;
	cv::Mat mTcw;                                          ///< 相机姿态 世界坐标系到相机坐标坐标系的变换矩阵
    cv::Mat mRcw;                                          ///< Rotation from world to camera
    cv::Mat mRwc;                                          ///< Rotation from camera to world
    cv::Mat mtcw;                                          ///< Translation from world to camera   
    cv::Mat mOw;                                           ///< mtwc,Translation from camera to world
};  
#endif //FRAME_H
