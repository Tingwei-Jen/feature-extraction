#include "matcher.h"
#include "converter.h"
#include "tools.h"
#include <iostream>
#include <Eigen/Dense>
using namespace cv;
using namespace std;

Matcher::Matcher()
{
    cout<<"Construct Matcher"<<endl;
    this->mDistFOETH = 100.0;
    this->mDepthMax = 100.0;
    this->mDepthMin = 10.0;
    this->mWindowSizeMaxX = 50.0;
    this->mWindowSizeMinX = 15.0;
    this->mWindowSizeMaxY = 50.0;
    this->mWindowSizeMinY = 15.0;    
    this->mDepthMaxTH = 150.0;
    this->mEpipolarLineTH = 5.0;
    this->mDescriptorDistTH = 0.3;
    this->mDescriptorMinDistTH = 0.05;
    this->mNNratio = 0.9;
}

int Matcher::SearchForInitialization(Frame& f1, Frame& f2, vector<int>& vMatchesQIdx, vector<int>& vMatchesTIdx)
{
    vMatchesQIdx.clear();
    vMatchesTIdx.clear();

    float delTime = f2.mTimestamp - f1.mTimestamp;

    vector<float> vDistTemp;
    vector<int> vQueryIdxTemp;
    vector<int> vTrainIdxTemp;

    // minimum dist between descriptors
    float minDistAll = 10000;

    for(int i=0; i<f1.N; i++)
    {
        // 0. Filter out area nearby FOE 
        if(norm(f1.mvKps[i]-GetFOE(f1)) < this->mDistFOETH)
            continue;

        // 1. Get features in predicted area in frame2
        Point3f p1Cam = f1.Px2Cam(f1.mvKps[i]);

        Point3f p1CamVelMax = MotionFieldRotation(f1.mAngularVel, p1Cam) 
                                + MotionFieldTranslation(f1.mLinearVel, this->mDepthMin, p1Cam);
        Point3f p1CamVelMin = MotionFieldRotation(f1.mAngularVel, p1Cam) 
                                + MotionFieldTranslation(f1.mLinearVel, this->mDepthMax, p1Cam);

        Point3f p1CamPredMax = p1Cam + p1CamVelMax*delTime;
        Point3f p1CamPredMin = p1Cam + p1CamVelMin*delTime;
        
        Point2f px1PredMax = f1.Cam2Px(p1CamPredMax);
        Point2f px1PredMin = f1.Cam2Px(p1CamPredMin);

        // predict pixel location int next frame according to camera velocity
        Point2f px1PredCenter = (px1PredMax+px1PredMin)/2;

        // window size
        float windowSizeX = 1.5*abs(px1PredMax.x-px1PredMin.x);
        float windowSizeY = 1.5*abs(px1PredMax.y-px1PredMin.y);
        
        if(windowSizeX > this->mWindowSizeMaxX)
            windowSizeX = this->mWindowSizeMaxX;
        else if (windowSizeX < this->mWindowSizeMinX)
            windowSizeX = this->mWindowSizeMinX;

        if(windowSizeY > this->mWindowSizeMaxY)
            windowSizeY = this->mWindowSizeMaxY;
        else if (windowSizeY < this->mWindowSizeMinY)
            windowSizeY = this->mWindowSizeMinY;

        vector<int> vIdx = f2.GetFeaturesInArea(px1PredCenter.x, px1PredCenter.y, windowSizeX, windowSizeY);

        if(vIdx.empty())
            continue;       
        
        // descriptors of f1 
        Mat d1 = f1.mDescriptors.row(i);
        
        // first two lowest distance
        float bestDist = 10000;     
        float bestDist2 = 10000;
        int bestIdx = -1;

        // 2. Get best matched feature
        for(int j=0; j<vIdx.size(); j++)
        {
            Mat d2 = f2.mDescriptors.row(vIdx[j]);
            
            float dist = norm(d1, d2, NORM_L2);
            if(dist >= 10000)
                continue;

            // first two minimum
            if(dist<bestDist)
            {
                bestDist2 = bestDist;
                bestDist = dist;
                bestIdx = vIdx[j];
            }
            else if(dist<bestDist2)
            {
                bestDist2 = dist;
            }

            // All
            if(dist<minDistAll)
                minDistAll = dist;
        }
        
        if(bestDist>this->mDescriptorDistTH)
            continue;

        // Need obvious difference between bestDist and bestDist2, so bestDist would be the best.
        if(bestDist>bestDist2*this->mNNratio)
            continue;    

        // 3. Check Pixel variation accordiing to depth max (significant motion)
        Point3f p1CamVelTH = MotionFieldRotation(f1.mAngularVel, p1Cam)  
                                + MotionFieldTranslation(f1.mLinearVel, this->mDepthMaxTH, p1Cam);
        
        Point3f p1CamPredTH = p1Cam + p1CamVelTH*delTime;
        Point2f px1PredTH = f1.Cam2Px(p1CamPredTH);

        if(norm(f2.mvKps[bestIdx]-f1.mvKps[i]) < norm(px1PredTH-f1.mvKps[i]))
            continue;

        // 4.Check Epipolar constraint
        cv::Point3f v1 = f1.mLinearVel;
        cv::Point3f w1 = f1.mAngularVel;
        
        cv::Point3f displacement = v1*delTime;
        cv::Point3f deltaTheta = w1*delTime;

        Mat R12 = Tools::FixedAngle2RotM(deltaTheta);
        Mat R21 = R12.t();

        Mat t12 = Converter::toCvMat(displacement);    
        Mat t21 = -R21*t12;

        Mat F21 = ComputeF21(R21, t21, f1.mK);
        Mat px1_ = ( Mat_<float> ( 3,1 ) <<  f1.mvKps[i].x, f1.mvKps[i].y, 1 );
        Mat line = F21 * px1_;

        float a = line.at<float>(0,0); 
        float b = line.at<float>(1,0); 
        float c = line.at<float>(2,0);

        // 计算px2特征点到极线的距离：
        // 极线l：ax + by + c = 0
        // (u,v)到l的距离为： |au+bv+c| / sqrt(a^2+b^2)
        float num = a*f2.mvKps[bestIdx].x + b*f2.mvKps[bestIdx].y + c;
        float den = a*a+b*b;

        if(den==0)
            return false;

        float dsqr = num*num/den;

        if(sqrt(dsqr)>this->mEpipolarLineTH)
            continue;

        // 4.5. Record temp matches 
        vDistTemp.push_back(bestDist);
        vQueryIdxTemp.push_back(i);
        vTrainIdxTemp.push_back(bestIdx);


        //Plot
        Mat color1, color2;
        cv::cvtColor(f1.mImg, color1, cv::COLOR_GRAY2BGR);    
        cv::cvtColor(f2.mImg, color2, cv::COLOR_GRAY2BGR);
       
        cv::circle(color1, px1PredMax, 2, cv::Scalar(255,0,255), -1);
        cv::circle(color1, px1PredMin, 2, cv::Scalar(255,0,255), -1);        
        cv::circle(color1, GetFOE(f1), 2, cv::Scalar(0,255,255), -1); 
        cv::circle(color1, GetFOE(f1), this->mDistFOETH, cv::Scalar(0,255,255), 1);    

        cv::circle(color2, px1PredMax, 2, cv::Scalar(255,0,255), -1);
        cv::circle(color2, px1PredMin, 2, cv::Scalar(255,0,255), -1);
        
        Point p1, p2;
        p1 = Point((px1PredCenter.x-windowSizeX/2), (px1PredCenter.y-windowSizeY/2));
        p2 = Point((px1PredCenter.x+windowSizeX/2), (px1PredCenter.y+windowSizeY/2));
        cv::rectangle(color2, p1, p2, cv::Scalar(255,255,0), 1);
        for(int j=0; j<vIdx.size(); j++)
            cv::circle(color2, f2.mvKps[vIdx[j]], 2, cv::Scalar(255,255,0), -1);

        cv::line(color2, Point(0,-c/b), Point(f2.width, -(c+a*f2.width)/b), Scalar(0,0,255));
        //Point3f epipolarLine2 = GetEpipolarLine(f1,f2, f1.mvKps[i]);
        // float a2 = epipolarLine2.x; 
        // float b2 = epipolarLine2.y; 
        // float c2 = epipolarLine2.z;
        //cv::line(color2, Point(0,-c2/b2), Point(f2.width, -(c2+a2*f2.width)/b2), Scalar(255,0,255));

        cv::circle(color1, f1.mvKps[i], 2, cv::Scalar(0,255,0), -1);    
        cv::circle(color2, f1.mvKps[i], 2, cv::Scalar(0,255,0), -1);    
        cv::circle(color2, f2.mvKps[bestIdx], 2, cv::Scalar(0,0,255), -1);
        // imshow("color1", color1);
        // imshow("color2", color2);  
        // //imwrite("color1.jpg", color1);
        // waitKey(0); 
    }

    // 5. Find good matches
    for(int i=0; i<vDistTemp.size(); i++)
    {
        if(vDistTemp[i]<=max(2*minDistAll, this->mDescriptorMinDistTH))
        {
            vMatchesQIdx.push_back(vQueryIdxTemp[i]);
            vMatchesTIdx.push_back(vTrainIdxTemp[i]);
        }
    }

    // 6. Update Depth of frame2
    for(int i=0; i<vMatchesQIdx.size(); i++)
    {
        Point2f px1 = f1.mvKps[vMatchesQIdx[i]];
        Point2f px2 = f2.mvKps[vMatchesTIdx[i]];
    
        Point3f x3Dp;
        
        if(!Triangulation(f1, f2, px1, px2, x3Dp))
            continue;

        f1.mvKpsDepth[vMatchesQIdx[i]] = f1.World2Cam(x3Dp).z;
        f2.mvKpsDepth[vMatchesTIdx[i]] = f2.World2Cam(x3Dp).z;
    }

    return vMatchesQIdx.size();
}

int Matcher::SearchForProjection(Frame& last, Frame& current, vector<int>& vMatchesQIdx, vector<int>& vMatchesTIdx)
{
    vMatchesQIdx.clear();
    vMatchesTIdx.clear();

    float delTime = current.mTimestamp - last.mTimestamp;

    cv::Point3f vl = last.mLinearVel;
    cv::Point3f wl = last.mAngularVel;

    cv::Point3f displacement = vl*delTime;
    cv::Point3f deltaTheta = wl*delTime;

    // 0. Get Rcl, tcl
    Mat Rlc = Tools::FixedAngle2RotM(deltaTheta);
    Mat Rcl = Rlc.t();
    
    Mat tlc = Converter::toCvMat(displacement);    
    Mat tcl = -Rcl*tlc;

    vector<float> vDistTemp;
    vector<int> vQueryIdxTemp;
    vector<int> vTrainIdxTemp;

    // minimum dist between descriptors
    float minDistAll = 10000;

    for(int i=0; i<last.N; i++)
    {
        if(last.mvKpsDepth[i] > -1.0)    //only match the feature with depth
        {
            // 0. Filter out area nearby FOE 
            if(norm(last.mvKps[i]-GetFOE(last)) < this->mDistFOETH)
                continue;

            // 1.1. recover depth of px in last frame
            Point2f pxl = last.mvKps[i];
            Point3f plCam = last.Px2Cam(pxl);
            plCam = plCam*last.mvKpsDepth[i];

            // 1.2. project to current frame
            Point3f pcCam = Converter::toCvPoint3f(Rcl*Converter::toCvMat(plCam) + tcl);                
            Point2f pxc = current.Cam2Px(pcCam);

            if(pxc.x > current.width || pxc.x < 0)
                continue;
            if(pxc.y > current.height || pxc.y < 0)
                continue;

            // 1.3. find features nearby pxc
            vector<int> vIdx = current.GetFeaturesInArea(pxc.x, pxc.y, this->mWindowSizeMinX, this->mWindowSizeMinY);

            if(vIdx.empty())
                continue;      

            Mat d1 = last.mDescriptors.row(i);

            // lowest distance
            float bestDist = 10000;     
            int bestIdx = -1;

            // 2. Get best matched feature
            for(int j=0; j<vIdx.size(); j++)
            {
                // skip the features that with depth
                if(current.mvKpsDepth[vIdx[j]] != -1.0)
                    continue;

                Mat d2 = current.mDescriptors.row(vIdx[j]);
                
                // get dist between descriptors
                float dist = norm(d1, d2, NORM_L2);
                if(dist >= 10000)
                    continue;

                if(dist<bestDist)
                {
                    bestDist = dist;
                    bestIdx = vIdx[j];
                }
            }

            // distance threshold
            if(bestDist>this->mDescriptorDistTH)
                continue;

            // 3. Check Epipolar constraint
            Mat Fcl = ComputeF21(Rcl, tcl, last.mK);
            Mat pxl_ = ( Mat_<float> ( 3,1 ) <<  pxl.x, pxl.y, 1 );
            Mat line = Fcl * pxl_;

            float a = line.at<float>(0,0); 
            float b = line.at<float>(1,0); 
            float c = line.at<float>(2,0);

            float num = a*current.mvKps[bestIdx].x + b*current.mvKps[bestIdx].y + c;
            float den = a*a+b*b;

            if(den==0)
                continue;

            float dsqr = num*num/den;

            if(sqrt(dsqr)>this->mEpipolarLineTH)
                continue;;

            // 4. Record
            current.mvKpsDepth[bestIdx] = last.mvKpsDepth[i];
            vMatchesQIdx.push_back(i);
            vMatchesTIdx.push_back(bestIdx);


            // //Plot
            Mat color1, color2;
            cv::cvtColor(last.mImg, color1, cv::COLOR_GRAY2BGR);    
            cv::cvtColor(current.mImg, color2, cv::COLOR_GRAY2BGR);
            cv::circle(color1, pxl, 3, cv::Scalar(0,255,0), -1);  
            cv::circle(color1, GetFOE(last), 2, cv::Scalar(0,255,255), -1); 
            cv::circle(color1, GetFOE(last), this->mDistFOETH, cv::Scalar(0,255,255), 1);                  
            
            cv::circle(color2, pxl, 3, cv::Scalar(0,255,0), -1);    
            cv::circle(color2, pxc, 3, cv::Scalar(0,0,255), -1);
            cv::circle(color2, current.mvKps[bestIdx], 5, cv::Scalar(255,0,255), -1);
            Point p1, p2;
            p1 = Point((pxc.x-this->mWindowSizeMinX/2), (pxc.y-this->mWindowSizeMinY/2));
            p2 = Point((pxc.x+this->mWindowSizeMinX/2), (pxc.y+this->mWindowSizeMinY/2));
            cv::rectangle(color2, p1, p2, cv::Scalar(255,255,0), 1);
            for(int i=0; i<vIdx.size(); i++)
                cv::circle(color2, current.mvKps[vIdx[i]], 2, cv::Scalar(0,255,0), -1);
            cv::line(color2, Point(0,-c/b), Point(current.width, -(c+a*current.width)/b), Scalar(0,0,255));
            
            // imshow("color1", color1);
            // imshow("color2", color2);  
            // waitKey(0); 
     
        }
    }

    return vMatchesQIdx.size();
}

int Matcher::SearchForCreateNewMatches(Frame& last, Frame& current, vector<int>& vMatchesQIdx, vector<int>& vMatchesTIdx)
{

    vMatchesQIdx.clear();
    vMatchesTIdx.clear();

    float delTime = current.mTimestamp - last.mTimestamp;

    vector<float> vDistTemp;
    vector<int> vQueryIdxTemp;
    vector<int> vTrainIdxTemp;

    // minimum dist between descriptors
    float minDistAll = 10000;


    for(int i=0; i<last.N; i++)
    {
        if(last.mvKpsDepth[i] == -1.0)    //only match the feature without depth (new)
        {

            float avgZ = last.GetAverageDepthInArea(last.mvKps[i].x, last.mvKps[i].y);

            cout<<avgZ<<endl;








            //plot
            Mat color1, color2;
            cv::cvtColor(last.mImg, color1, cv::COLOR_GRAY2BGR);    
            cv::cvtColor(current.mImg, color2, cv::COLOR_GRAY2BGR);

            cv::circle(color1, last.mvKps[i], 3, cv::Scalar(0,255,0), -1);  

            imshow("color1", color1);
            imshow("color2", color2);  
            waitKey(0); 

        }

    }


}



/*
private
*/ 

/**
 * vx = x*y*Wx-(1+x*x)*Wy+y*Wz
 * vy = (1+y*y)*wx-x*y*Wy-x*Wz
 */
Point3f Matcher::MotionFieldRotation(const Point3f& camAngularVel, const Point3f& pCam)
{
    cv::Point3f pxCamRotationFlow = Point3f(pCam.x*pCam.y*camAngularVel.x - (1+pCam.x*pCam.x)*camAngularVel.y + pCam.y*camAngularVel.z,
                                           (1+pCam.y*pCam.y)*camAngularVel.x - pCam.x*pCam.y*camAngularVel.y - pCam.x*camAngularVel.z,
                                            0);
    return pxCamRotationFlow;
}           
/**
 * vx = (x*Vz-Vx)/depth
 * vy = (y*Vz-Vy)/depth
 */
Point3f Matcher::MotionFieldTranslation(const Point3f& camLinearVel, const float& depth, const Point3f& pCam)
{
    cv::Point3f pxCamTranslationFlow = Point3f(( pCam.x*camLinearVel.z - camLinearVel.x )/depth, 
                                               ( pCam.y*camLinearVel.z - camLinearVel.y )/depth, 
                                                0);
    return pxCamTranslationFlow;
}

/**
 * float a = line.x; float b = line.y; float c = line.z;
 * cv::line(plot_cur, Point(0,-c/b), Point(width, -(c+a*width)/b), Scalar(0,0,255));
 */
cv::Point3f Matcher::GetEpipolarLine(Frame& f1, Frame& f2, const cv::Point2f& px1)
{
    Mat Rcw1 = f1.GetRotation();    // c1Rw
    Mat Rcw2 = f2.GetRotation();    // c2Rw
    Mat R21 = Rcw2*Rcw1.t();        // c2Rc1 = c2Rw * wRc1

    Mat tcw1 = f1.GetTranslation(); // c1tw
    Mat twc1 = -Rcw1.t() * tcw1;    // wtc1(w)

    Mat tcw2 = f2.GetTranslation();  //c2tw
    Mat t21 = tcw2 + Rcw2*twc1;      // c2tc1 = c2tw + wtc1(c2), wtc1(c2) = c2Rw*wtc1

    Mat F21 = ComputeF21(R21, t21, f1.mK);
    Mat px1_ = ( Mat_<float> ( 3,1 ) <<  px1.x, px1.y, 1 );
    Mat line = F21 * px1_;
    return Point3f(line.at<float>(0,0), line.at<float>(1,0), line.at<float>(2,0));
}

cv::Point3f Matcher::GetEpipolarLinePred(Frame& f1, Frame& f2, const cv::Point2f& px1)
{
    cv::Point3f v1 = f1.mLinearVel;
    cv::Point3f w1 = f1.mAngularVel;

    float delTime = f2.mTimestamp - f1.mTimestamp;
    cv::Point3f displacement = v1*delTime;
    cv::Point3f deltaTheta = w1*delTime;

    Mat R12 = Tools::FixedAngle2RotM(deltaTheta);
    Mat R21 = R12.t();

    Mat t12 = Converter::toCvMat(displacement);    
    Mat t21 = -R21*t12;

    Mat F21 = ComputeF21(R21, t21, f1.mK);
    Mat px1_ = ( Mat_<float> ( 3,1 ) <<  px1.x, px1.y, 1 );
    Mat line = F21 * px1_;
    return Point3f(line.at<float>(0,0), line.at<float>(1,0), line.at<float>(2,0));
}

/**
 * @brief        从R21, t21 & K 求fundamental matrix
 * @param  R21   Rotation matrix between frame1 and frame2
 * @param  t21   Translation matrix between frame1 and frame2
 * @param  K     camera parameter
 * @return       Fundamental matrix
 */
cv::Mat Matcher::ComputeF21(const Mat& R21, const Mat& t21, const Mat& K)
{
    Mat t21x = Mat(3,3, CV_32F);
    t21x.at<float>(0,0) = 0;                  t21x.at<float>(0,1) = -t21.at<float> (2);  t21x.at<float>(0,2) = t21.at<float> (1);
    t21x.at<float>(1,0) = t21.at<float> (2);  t21x.at<float>(1,1) = 0;                   t21x.at<float>(1,2) = -t21.at<float> (0);
    t21x.at<float>(2,0) = -t21.at<float> (1); t21x.at<float>(2,1) = t21.at<float> (0);   t21x.at<float>(2,2) = 0;

    Mat E = t21x * R21;
    Mat F = K.inv().t() * E * K.inv();

    //clean up F
    Eigen::MatrixXf f(3,3);
    f(0,0) = F.at<float>(0,0); f(0,1) = F.at<float>(0,1); f(0,2) = F.at<float>(0,2);
    f(1,0) = F.at<float>(1,0); f(1,1) = F.at<float>(1,1); f(1,2) = F.at<float>(1,2);
    f(2,0) = F.at<float>(2,0); f(2,1) = F.at<float>(2,1); f(2,2) = F.at<float>(2,2);

    Eigen::JacobiSVD<Eigen::MatrixXf> svd(f, Eigen::ComputeFullV | Eigen::ComputeFullU); // ComputeThinU | ComputeThinV
    Eigen::MatrixXf singular_values = svd.singularValues();
    Eigen::MatrixXf left_singular_vectors = svd.matrixU();
    Eigen::MatrixXf right_singular_vectors = svd.matrixV();
    
    Eigen::MatrixXf d(3,3);
    d(0,0) = singular_values(0); d(0,1) = 0;                  d(0,2) = 0;
    d(1,0) = 0;                  d(1,1) = singular_values(1); d(1,2) = 0;
    d(2,0) = 0;                  d(2,1) = 0;                  d(2,2) = 0;

    f = left_singular_vectors*d*right_singular_vectors.transpose();
    F = ( Mat_<float> ( 3,3 ) << f(0,0), f(0,1), f(0,2), f(1,0), f(1,1), f(1,2), f(2,0), f(2,1), f(2,2));

    return F;
}

/**
 * @brief          Find epipole in frame2
 * @param f1       frame1
 * @param f2       frame2
 * @return         pixel location of epipole
 */
cv::Point2f Matcher::GetEpipole(Frame& f1, Frame& f2)
{
    Mat Rcw2 = f2.GetRotation();
    Mat tcw2 = f2.GetTranslation();

    Mat Ow1 = f1.GetCameraCenter();
    Mat Oc2w1 = Rcw2 * Ow1 + tcw2;

    return f2.Cam2Px(Converter::toCvPoint3f(Oc2w1));
}

/**
 * @brief          Find FOE in frame1
 * @param f1       frame1
 * @param f2       frame2
 * @return         pixel location of FOE
 */
cv::Point2f Matcher::GetFOE(Frame& f1)
{
    cv::Point3f v1 = f1.mLinearVel;

    if(v1.z==0)
        return Point2f(-1.0, -1.0);
    else 
        return f1.Cam2Px(Point3f(v1.x/v1.z, v1.y/v1.z, 1.0));
}


// |xp2  - p0 |     |0|
// |yp2  - p1 | X = |0| ===> AX = 0
// |x'p2'- p0'|     |0|
// |y'p2'- p1'|     |0|
bool Matcher::Triangulation(Frame& f1, Frame& f2, const cv::Point2f& px1, const cv::Point2f& px2, cv::Point3f& x3Dp)
{
    Mat Tcw1 = f1.GetPose();
    Mat Tcw2 = f2.GetPose();

    Point3f p1Cam = f1.Px2Cam(px1);
    Point3f p2Cam = f2.Px2Cam(px2);

    Mat A(4,4, CV_32F);
    A.row(0) = p1Cam.x*Tcw1.row(2)-Tcw1.row(0);
    A.row(1) = p1Cam.y*Tcw1.row(2)-Tcw1.row(1);
    A.row(2) = p2Cam.x*Tcw2.row(2)-Tcw2.row(0);
    A.row(3) = p2Cam.y*Tcw2.row(2)-Tcw2.row(1);

    Mat u,w,vt;
    cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);
    Mat x3D = vt.row(3).t();

    if(x3D.at<float>(3) == 0)
        return false;
    x3D = x3D.rowRange(0,3)/x3D.at<float>(3);  

    // check triangulation in front of camera.
    Mat Rcw1, tcw1;
    Tcw1.rowRange(0,3).colRange(0,3).copyTo(Rcw1);
    Tcw1.rowRange(0,3).col(3).copyTo(tcw1);
    
    Mat Rcw2, tcw2;
    Tcw2.colRange(0,3).copyTo(Rcw2);
    Tcw2.col(3).copyTo(tcw2);

    Mat x3Dt = x3D.t();

    float z1 = Rcw1.row(2).dot(x3Dt)+tcw1.at<float>(2);
    if(z1<=0)
        return false;
        
    float z2 = Rcw2.row(2).dot(x3Dt)+tcw2.at<float>(2);
    if(z2<=0)
        return false;

    x3Dp = Point3f(x3D.at<float>(0), x3D.at<float>(1), x3D.at<float>(2));

    return true;
}