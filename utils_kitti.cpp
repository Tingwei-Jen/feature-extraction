#include "utils_kitti.h"
#include "tools.h"
#include <iostream>

using namespace cv;
using namespace std;

UtilsKitti::UtilsKitti()
{
    cout<<"Construct UtilsKitti"<<endl;
    mT_cam_imu = cv::Mat(4,4, CV_32F);
    mT_cam_imu.at<float>(0,0) = 0.00831785890370637; mT_cam_imu.at<float>(0,1) = -0.999864653701645; mT_cam_imu.at<float>(0,2) = 0.0141906856665874; mT_cam_imu.at<float>(0,3) = -0.329215773856599;
    mT_cam_imu.at<float>(1,0) = 0.0127776989230672;  mT_cam_imu.at<float>(1,1) = -0.014083738773391; mT_cam_imu.at<float>(1,2) = -0.999819239796591; mT_cam_imu.at<float>(1,3) = 0.711581353599277;
    mT_cam_imu.at<float>(2,0) = 0.999883767619045;   mT_cam_imu.at<float>(2,1) = 0.00849767893876991;mT_cam_imu.at<float>(2,2) = 0.0126588227868662; mT_cam_imu.at<float>(2,3) = -1.08978265189793;
    mT_cam_imu.at<float>(3,0) = 0;                   mT_cam_imu.at<float>(3,1) = 0;                  mT_cam_imu.at<float>(3,2) = 0;                  mT_cam_imu.at<float>(3,3) = 1;                

    this->mR_cam_imu = this->mT_cam_imu.rowRange(0,3).colRange(0,3);

    this->mK0926 = cv::Mat(3,3, CV_32F);
    mK0926.at<float>(0,0) = 7.215377e+02; mK0926.at<float>(0,1) = 0.0;          mK0926.at<float>(0,2) = 6.095593e+02;
    mK0926.at<float>(1,0) = 0.0;          mK0926.at<float>(1,1) = 7.215377e+02; mK0926.at<float>(1,2) = 1.728540e+02;
    mK0926.at<float>(2,0) = 0.0;          mK0926.at<float>(2,1) = 0.0;          mK0926.at<float>(2,2) = 1.0;

    this->mK0929 = cv::Mat(3,3, CV_32F);
    mK0929.at<float>(0,0) = 7.183351e+02; mK0929.at<float>(0,1) = 0.0;          mK0929.at<float>(0,2) = 6.003891e+02;
    mK0929.at<float>(1,0) = 0.0;          mK0929.at<float>(1,1) = 7.183351e+02; mK0929.at<float>(1,2) = 1.815122e+02;
    mK0929.at<float>(2,0) = 0.0;          mK0929.at<float>(2,1) = 0.0;          mK0929.at<float>(2,2) = 1.0;    
}

void UtilsKitti::ReadData(const std::string& img_pattern, 
                          const std::string& csv_path_vel,
                          const std::string& csv_path_tf,
                          std::vector<cv::Mat>& imgs, 
                          std::vector<double>& imgtimes, 
                          std::vector<cv::Point3f>& linear_v, 
                          std::vector<cv::Point3f>& angular_v,
                          std::vector<cv::Mat>& Tcws)
{

    ReadImgs(img_pattern, imgs, imgtimes);
    ReadOXTsVel(csv_path_vel, linear_v, angular_v);
    ReadTF(csv_path_tf, Tcws);
}

void UtilsKitti::ReadImgs(const std::string& pattern, std::vector<cv::Mat>& imgs, std::vector<double>& imgtimes)
{
    imgs.clear();
    imgtimes.clear();
    vector<String> fn;
    glob(pattern, fn, false);
    size_t count = fn.size();
    vector<string> imgtimes_s;

    for (size_t i = 0; i < count; i++)
    {
        string img_name = fn[i].substr(46,100);
        imgtimes_s.push_back(img_name.substr(0,12));
        imgs.push_back(imread(fn[i], 0));
    }

    for(int i=0; i<imgtimes_s.size(); i++)
    {
        imgtimes.push_back(stod(imgtimes_s[i])-stod(imgtimes_s[0]));
    }
}

void UtilsKitti::ReadOXTsVel(const string& csv_path, vector<Point3f>& linear_v, vector<Point3f>& angular_v)
{
    linear_v.clear();
    angular_v.clear();

    ifstream file(csv_path);
    std::string line;
    int row = 0;
    while(getline(file, line))
    {
        if(row==0)
        {
            row++;
            continue;       
        }

        stringstream ss(line);
        string str;
        
        int col = 0;
        double vx, vy, vz;
        double wx, wy, wz;
        while (getline(ss, str, ','))
        {
            std::stringstream convertor(str);
            double value;
            convertor >> value;

            if(col==5)
                vx = value;
            else if(col==6)
                vy = value;
            else if(col==7)
                vz = value;
            else if (col==8)
                wx = value;
            else if (col==9)
                wy = value;
            else if (col==10)
                wz = value;
            col++;
        }

        //Change to camera frame
        Mat linear_v_imu = cv::Mat(3,1, CV_32F);
        linear_v_imu.at<float>(0,0) = vx;
        linear_v_imu.at<float>(1,0) = vy;
        linear_v_imu.at<float>(2,0) = vz;

        Mat angular_v_imu = cv::Mat(3,1, CV_32F);
        angular_v_imu.at<float>(0,0) = wx;
        angular_v_imu.at<float>(1,0) = wy;
        angular_v_imu.at<float>(2,0) = wz;

        Mat linear_v_cam = this->mR_cam_imu * linear_v_imu;
        Mat angular_v_cam = this->mR_cam_imu * angular_v_imu;

        linear_v.push_back(Point3f(linear_v_cam.at<float>(0,0), linear_v_cam.at<float>(1,0), linear_v_cam.at<float>(2,0)));
        angular_v.push_back(Point3f(angular_v_cam.at<float>(0,0), angular_v_cam.at<float>(1,0), angular_v_cam.at<float>(2,0)));

        row++;        
    }
} 

void UtilsKitti::ReadTF(const string& csv_path, vector<Mat>& Tcws)
{
    Tcws.clear();

    ifstream file(csv_path);
    std::string line;
    int row = 0;
    double x,y,z, qx, qy, qz, qw;
    while(getline(file, line))
    {
        if(row==0)
        {
            row++;
            continue;       
        }

        string value = line.substr(7,19);
        
        if(row%17 == 10)
            x = stod(value);
        else if(row%17 == 11)
            y = stod(value);
        else if(row%17 == 12)
            z = stod(value);
        else if(row%17 == 14)
            qx = stod(value);
        else if(row%17 == 15)
            qy = stod(value);
        else if(row%17 == 16)
            qz = stod(value);
        else if(row%17 == 0)
        {
            qw = stod(value);

            Mat Rwi = Tools::Quaternion2RotM(qx, qy, qz, qw);
            Mat Twi = (Mat_<float> (4,4) <<
                Rwi.at<float>(0,0), Rwi.at<float>(0,1), Rwi.at<float>(0,2), x,
                Rwi.at<float>(1,0), Rwi.at<float>(1,1), Rwi.at<float>(1,2), y,
                Rwi.at<float>(2,0), Rwi.at<float>(2,1), Rwi.at<float>(2,2), z,
                                0,                0,                0,     1
                );

            Mat Tiw = Twi.inv();
            Tcws.push_back(this->mT_cam_imu * Tiw);
        }
        row++;        
    }
}