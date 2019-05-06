#include "utils.h"
#include "tools.h"
using namespace cv;
using namespace std;

Utils::Utils()
{
    cout<<"Construct Utils"<<endl;

    this->mK = cv::Mat(3,3, CV_32F);
    mK.at<float>(0,0) = 712.584655; mK.at<float>(0,1) = 0.0;        mK.at<float>(0,2) = 613.712890;
    mK.at<float>(1,0) = 0.0;        mK.at<float>(1,1) = 713.578491; mK.at<float>(1,2) = 386.504699;
    mK.at<float>(2,0) = 0.0;        mK.at<float>(2,1) = 0.0;        mK.at<float>(2,2) = 1.0;
}

void Utils::ReadImgAndTime(const string& pattern, const int& idx, vector<string>& imgtimes, vector<Mat>& imgs)
{
	imgtimes.clear();
    imgs.clear();
    vector<String> fn;
    glob(pattern, fn, false);
    size_t count = fn.size();

    for (size_t i = 0; i < count; i++)
    {
        string img_name = fn[i].substr(idx,100);
        imgtimes.push_back(img_name.substr(0,13));
        imgs.push_back(imread(fn[i]));
    }
}

void Utils::ReadCameraPose(const string& csv_path, const vector<string>& imgtimes, vector<Mat>& Twcs)
{
    string imgtimestart = imgtimes[0];
    string imgtimeend = imgtimes[imgtimes.size()-1];

    Twcs.clear();
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
        double timesec, timensec;
        double posx, posy, posz;
        double qx, qy, qz, qw;
        while (getline(ss, str, ','))
        {
            std::stringstream convertor(str);
            double value;
            convertor >> value;

            if(col==2)
                timesec = value;
            else if(col==3)
                timensec = value;
            else if(col==6)
                posx = value;
            else if (col==7)
                posy = value;
            else if (col==8)
                posz = value;
            else if (col==9)
                qx = value;
            else if (col==10)
                qy = value;
            else if (col==11)
                qz = value;
            else if (col==12)
                qw = value;

            col++;
        }
        
        timensec = timensec/1000000000;
        timesec = timesec + timensec;
        string name =  std::to_string(timesec).substr(0,13);
        double named = stod(name);

        if(named>=stod(imgtimestart) && named<=stod(imgtimeend))
        {
            Mat R = Tools::Quaternion2RotM(qx, qy, qz, qw);
            Mat T = (Mat_<float> (4,4) <<
                R.at<float>(0,0), R.at<float>(0,1), R.at<float>(0,2), posx,
                R.at<float>(1,0), R.at<float>(1,1), R.at<float>(1,2), posy,
                R.at<float>(2,0), R.at<float>(2,1), R.at<float>(2,2), posz,
                               0,                0,                0,     1
                );
            Twcs.push_back(T);
        }
        row++;        
    }
}

void Utils::ReadCameraPose(const string& csv_path, vector<Mat>& Twcs)
{
    Twcs.clear();
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
        double timesec, timensec;
        double posx, posy, posz;
        double qx, qy, qz, qw;
        while (getline(ss, str, ','))
        {
            std::stringstream convertor(str);
            double value;
            convertor >> value;

            if(col==2)
                timesec = value;
            else if(col==3)
                timensec = value;
            else if(col==6)
                posx = value;
            else if (col==7)
                posy = value;
            else if (col==8)
                posz = value;
            else if (col==9)
                qx = value;
            else if (col==10)
                qy = value;
            else if (col==11)
                qz = value;
            else if (col==12)
                qw = value;

            col++;
        }

        Mat R = Tools::Quaternion2RotM(qx, qy, qz, qw);
        Mat T = (Mat_<float> (4,4) <<
            R.at<float>(0,0), R.at<float>(0,1), R.at<float>(0,2), posx,
            R.at<float>(1,0), R.at<float>(1,1), R.at<float>(1,2), posy,
            R.at<float>(2,0), R.at<float>(2,1), R.at<float>(2,2), posz,
                            0,                0,                0,     1
            );
        Twcs.push_back(T);
        
        row++;        
    }
}

void Utils::ReadOdometry(const string& csv_path, const vector<string>& imgtimes, vector<Mat>& Twcs, vector<Point3f>& LinearVels)
{
    string imgtimestart = imgtimes[0];
    string imgtimeend = imgtimes[imgtimes.size()-1];

    Twcs.clear();
    LinearVels.clear();
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
        double timesec, timensec;
        double posx, posy, posz;
        double qx, qy, qz, qw;
        double velx, vely, velz;
        while (getline(ss, str, ','))
        {
            std::stringstream convertor(str);
            double value;
            convertor >> value;

            if(col==2)
                timesec = value;
            else if(col==3)
                timensec = value;
            else if(col==6)
                posx = value;
            else if (col==7)
                posy = value;
            else if (col==8)
                posz = value;
            else if (col==9)
                qx = value;
            else if (col==10)
                qy = value;
            else if (col==11)
                qz = value;
            else if (col==12)
                qw = value;
            else if (col==49)
                velx = value;
            else if (col==50)
                vely = value;
            else if (col==51)
                velz = value;
            
            col++;
        }
        
        timensec = timensec/1000000000;
        timesec = timesec + timensec;
        string name =  std::to_string(timesec).substr(0,13);
        double named = stod(name);

        if(named>=stod(imgtimestart) && named<=stod(imgtimeend))
        {
            Mat R = Tools::Quaternion2RotM(qx, qy, qz, qw);
            Mat T = (Mat_<float> (4,4) <<
                R.at<float>(0,0), R.at<float>(0,1), R.at<float>(0,2), posx,
                R.at<float>(1,0), R.at<float>(1,1), R.at<float>(1,2), posy,
                R.at<float>(2,0), R.at<float>(2,1), R.at<float>(2,2), posz,
                               0,                0,                0,     1
                );
            Twcs.push_back(T);

            LinearVels.push_back(Point3f(velx, vely, velz));
        }
        row++;        
    }
}

void Utils::ReadOdometry(const string& csv_path, vector<Mat>& Twcs, vector<Point3f>& LinearVels)
{
    Twcs.clear();
    LinearVels.clear();
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
        double timesec, timensec;
        double posx, posy, posz;
        double qx, qy, qz, qw;
        double velx, vely, velz;
        while (getline(ss, str, ','))
        {
            std::stringstream convertor(str);
            double value;
            convertor >> value;

            if(col==2)
                timesec = value;
            else if(col==3)
                timensec = value;
            else if(col==6)
                posx = value;
            else if (col==7)
                posy = value;
            else if (col==8)
                posz = value;
            else if (col==9)
                qx = value;
            else if (col==10)
                qy = value;
            else if (col==11)
                qz = value;
            else if (col==12)
                qw = value;
            else if (col==49)
                velx = value;
            else if (col==50)
                vely = value;
            else if (col==51)
                velz = value;
            
            col++;
        }

        Mat R = Tools::Quaternion2RotM(qx, qy, qz, qw);
        Mat T = (Mat_<float> (4,4) <<
            R.at<float>(0,0), R.at<float>(0,1), R.at<float>(0,2), posx,
            R.at<float>(1,0), R.at<float>(1,1), R.at<float>(1,2), posy,
            R.at<float>(2,0), R.at<float>(2,1), R.at<float>(2,2), posz,
                            0,                0,                0,     1
            );
        Twcs.push_back(T);

        LinearVels.push_back(Point3f(velx, vely, velz));
    
        row++;        
    }
}

void Utils::ReadIMUGyro(const std::string& csv_path, const std::vector<std::string>& imgtimes, std::vector<cv::Point3f>& AngularVels)
{
    string imgtimestart = imgtimes[0];
    string imgtimeend = imgtimes[imgtimes.size()-1];

    AngularVels.clear();
    vector<Point3f> temp;
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
        double timesec, timensec;
        double wx, wy, wz;
        while (getline(ss, str, ','))
        {
            std::stringstream convertor(str);
            double value;
            convertor >> value;

            if(col==2)
                timesec = value;
            else if(col==3)
                timensec = value;
            else if(col==18)
                wx = value;
            else if (col==19)
                wy = value;
            else if (col==20)
                wz = value;

            col++;
        }
        
        timensec = timensec/1000000000;
        timesec = timesec + timensec;
        string name =  std::to_string(timesec).substr(0,14);
        double named = stod(name);

        if(named>=stod(imgtimestart) && named<=stod(imgtimeend))
        {
            for(int i=0; i<imgtimes.size(); i++)
            {
                if(abs(named-stod(imgtimes[i]))<=0.0025)
                {
                    AngularVels.push_back(Point3f(wx, wy, wz));
                }
            }
        }
        row++;        
    }
}

void Utils::Pose2PosAndAngle(const vector<Mat>& Twcs, vector<Point3f>& Pos, vector<Point3f>& Ang)
{
    Pos.clear();
    Ang.clear();

    for(int i=0; i<Twcs.size(); i++)
    {
        Pos.push_back(Point3f(Twcs[i].at<float>(0,3), Twcs[i].at<float>(1,3), Twcs[i].at<float>(2,3)));
        Ang.push_back(Tools::RotM2FixedAngle(Twcs[i].rowRange(0,3).colRange(0,3)));
    }
}