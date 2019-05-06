#include "converter.h"
#include "frame.h"
#include "matcher.h"
#include "utils.h"
#include "utils_kitti.h"
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;


int main()
{
    UtilsKitti utilskitti;
 
    string pattern = "/home/tom/VIO_dataset/kitti_dataset/0059/imgs/*.png";
    string csv_path_vel = "/home/tom/VIO_dataset/kitti_dataset/0059/kitti_2011_09_26_drive_0059_synced-kitti-oxts-gps-vel.csv";
    string csv_path_tf = "/home/tom/VIO_dataset/kitti_dataset/0059/kitti_2011_09_26_drive_0059_synced-tf.csv"; 
  
    vector<Mat> imgs;
    vector<double> imgtimes;
    vector<Point3f> linear_v;
    vector<Point3f> angular_v;
    vector<Mat> Tcws;
    utilskitti.ReadData(pattern, csv_path_vel, csv_path_tf, imgs, imgtimes, linear_v, angular_v, Tcws);

    Mat K = utilskitti.GetK0926();
    int n_frames = imgs.size();


    cv::Ptr<cv::xfeatures2d::SURF> Detector = cv::xfeatures2d::SURF::create(400);

    vector<Frame> frames;
    frames.reserve(n_frames);
    for(int i=0; i<10; i++)
    {
        Frame frame(imgs[i], K, (float)imgtimes[i], Detector);
        frame.SetPose(Tcws[i]);
        frame.SetLinearVelocity(linear_v[i]);
        frame.SetAngularVelocity(angular_v[i]);
        frames.push_back(frame);
    }

    // int idx = 150;

    // Frame frame1(imgs[idx], K, (float)imgtimes[idx], 800);
    // frame1.SetPose(Tcws[idx]);
    // frame1.SetLinearVelocity(linear_v[idx]);
    // frame1.SetAngularVelocity(angular_v[idx]);
    // frames.push_back(frame1);

    // Frame frame2(imgs[idx+1], K, (float)imgtimes[idx+1], 800);
    // frame2.SetPose(Tcws[idx+1]);
    // frame2.SetLinearVelocity(linear_v[idx+1]);
    // frame2.SetAngularVelocity(angular_v[idx+1]);
    // frames.push_back(frame2);

    // Frame frame3(imgs[idx+2], K, (float)imgtimes[idx+2], 800);
    // frame3.SetPose(Tcws[idx+2]);
    // frame3.SetLinearVelocity(linear_v[idx+2]);
    // frame3.SetAngularVelocity(angular_v[idx+2]);
    // frames.push_back(frame3);



    // Size videoSize = Size(frames[0].width, frames[0].height);
    // VideoWriter writer;
    // writer.open("Video0004.avi", CV_FOURCC('M', 'J', 'P', 'G'), 30, videoSize);
    // namedWindow("show image",0);


    bool init = false;
    Matcher matcher;
    for(int i=0; i<frames.size()-1; i++)
    {
        Frame f1 = frames[i];
        Frame f2 = frames[i+1];
        vector<int> vMatchesQIdx, vMatchesTIdx;

        if(!init)
        {
            int nMatches = matcher.SearchForInitialization(f1, f2, vMatchesQIdx, vMatchesTIdx);
            cout<<"i: "<<i<<"  nMatches: "<<nMatches<<endl;
            init = true;
            frames[i] = Frame(f1);     //because of key point's depth
            frames[i+1] = Frame(f2);
        }
        else 
        {
            int nMatches = matcher.SearchForProjection(f1, f2, vMatchesQIdx, vMatchesTIdx);
            cout<<"i: "<<i<<"  nMatches: "<<nMatches<<endl;
            frames[i+1] = Frame(f2);

            vector<int> vMatchesQIdx_new, vMatchesTIdx_new;
            int nMatches_new = matcher.SearchForCreateNewMatches(f1, f2, vMatchesQIdx_new, vMatchesTIdx_new);
            cout<<"i: "<<i<<"  nMatches_new: "<<nMatches_new<<endl;
        }



        Mat color1, color2;
        cv::cvtColor(f1.mImg, color1, cv::COLOR_GRAY2BGR);   
        cv::cvtColor(f2.mImg, color2, cv::COLOR_GRAY2BGR);    
        for(int i=0; i<vMatchesQIdx.size(); i++)
        {
            cv::circle(color1, f1.mvKps[vMatchesQIdx[i]], 2, cv::Scalar(0,255,0), -1);
            cv::circle(color2, f1.mvKps[vMatchesQIdx[i]], 2, cv::Scalar(0,255,0), -1);
            cv::circle(color2, f2.mvKps[vMatchesTIdx[i]], 2, cv::Scalar(0,0,255), -1);
        }

        imshow("color1", color1);
        imshow("color2", color2);        
        //writer.write(color2);
        //imwrite("withoutVariation.jpg", color2);
        waitKey(0);    
    }

    return 0;
}


// string pattern = "/home/tom/VIO_dataset/kitti_dataset/0059/imgs/*.png";
// string csv_path_vel = "/home/tom/VIO_dataset/kitti_dataset/0059/kitti_2011_09_26_drive_0059_synced-kitti-oxts-gps-vel.csv";
// string csv_path_tf = "/home/tom/VIO_dataset/kitti_dataset/0059/kitti_2011_09_26_drive_0059_synced-tf.csv";  

// string pattern = "/home/tom/VIO_dataset/kitti_dataset/0017/imgs/*.png";
// string csv_path_vel = "/home/tom/VIO_dataset/kitti_dataset/0017/kitti_2011_09_26_drive_0017_synced-kitti-oxts-gps-vel.csv";
// string csv_path_tf = "/home/tom/VIO_dataset/kitti_dataset/0017/kitti_2011_09_26_drive_0017_synced-tf.csv";  

// string pattern = "/home/tom/VIO_dataset/kitti_dataset/0018/imgs/*.png";
// string csv_path_vel = "/home/tom/VIO_dataset/kitti_dataset/0018/kitti_2011_09_26_drive_0018_synced-kitti-oxts-gps-vel.csv";
// string csv_path_tf = "/home/tom/VIO_dataset/kitti_dataset/0018/kitti_2011_09_26_drive_0018_synced-tf.csv"; 

// string pattern = "/home/tom/VIO_dataset/kitti_dataset/0005/imgs/*.png";
// string csv_path_vel = "/home/tom/VIO_dataset/kitti_dataset/0005/kitti_2011_09_26_drive_0005_synced-kitti-oxts-gps-vel.csv";
// string csv_path_tf = "/home/tom/VIO_dataset/kitti_dataset/0005/kitti_2011_09_26_drive_0005_synced-tf.csv"; 