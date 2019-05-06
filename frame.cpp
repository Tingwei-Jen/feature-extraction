#include "frame.h"
#include "converter.h"

int Frame::frameCounter = 0;
float Frame::fx, Frame::fy, Frame::cx, Frame::cy;
int Frame::width, Frame::height;
float Frame::mGridElementWidthInv, Frame::mGridElementHeightInv;
bool Frame::mbInitialComputations=true;

Frame::Frame()
{
}
    
//Copy
Frame::Frame(const Frame& frame):mImg(frame.mImg), mId(frame.mId), mK(frame.mK), mTimestamp(frame.mTimestamp), 
mLinearVel(frame.mLinearVel), mAngularVel(frame.mAngularVel), mFocusOfExpansion(frame.mFocusOfExpansion),
N(frame.N), mvKps(frame.mvKps), mvKpsDepth(frame.mvKpsDepth), mDescriptors(frame.mDescriptors), mDetector(frame.mDetector)
{
    for(int i=0;i<FRAME_GRID_COLS;i++)
        for(int j=0; j<FRAME_GRID_ROWS; j++)
            this->mGrid[i][j]=frame.mGrid[i][j];
        
    if(!frame.mTcw.empty())
        SetPose(frame.mTcw);
}

// Initialization
Frame::Frame(const cv::Mat& img, const cv::Mat &K, const float& timeStamp, const cv::Ptr<cv::xfeatures2d::SURF>& Detector)
:mImg(img.clone()), mK(K.clone()), mTimestamp(timeStamp), mLinearVel(cv::Point3f(0,0,0)), mAngularVel(cv::Point3f(0,0,0)),
mFocusOfExpansion(cv::Point2f(0,0)), mDetector(Detector)
{
    
    // Frame ID
	this->mId = frameCounter++;

    // Extract features
    std::vector<cv::KeyPoint> vKeyPoints;
    this->mDetector->detectAndCompute( img, cv::Mat(), vKeyPoints, this->mDescriptors );
    cv::KeyPoint::convert(vKeyPoints, this->mvKps);
    
    if(this->mvKps.empty())
        return;

    this->N = this->mvKps.size();

    // Init depth
    this->mvKpsDepth.reserve(this->N);
    for(int i=0; i<this->N; i++)
    {
        this->mvKpsDepth.push_back(-1.0);
    }

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        width = img.cols;
        height = img.rows;
        mGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(width);
        mGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(height);
        
        mbInitialComputations = false;
    }

    AssignFeaturesToGrid();
}

void Frame::SetPose(const cv::Mat& Tcw)
{
	this->mTcw = Tcw.clone();   
    this->mRcw = this->mTcw.rowRange(0,3).colRange(0,3);
    this->mRwc = this->mRcw.t();
    this->mtcw = this->mTcw.rowRange(0,3).col(3);
    this->mOw = -this->mRcw.t()*this->mtcw;
}

void Frame::SetLinearVelocity(const cv::Point3f& linearVel)
{
    this->mLinearVel = linearVel;

    if(linearVel.z == 0)
        this->mFocusOfExpansion =  cv::Point2f(width/2, height/2);
    else
        this->mFocusOfExpansion = Cam2Px(cv::Point3f(linearVel.x/linearVel.z, linearVel.y/linearVel.z, 1));
}

void Frame::SetAngularVelocity(const cv::Point3f& angularVel)
{
    this->mAngularVel = angularVel;
}

void Frame::AssignFeaturesToGrid()
{
    int nReserve = 0.5f*this->N/(FRAME_GRID_COLS*FRAME_GRID_ROWS);
    for(unsigned int i=0; i<FRAME_GRID_COLS;i++)
        for (unsigned int j=0; j<FRAME_GRID_ROWS;j++)
            mGrid[i][j].reserve(nReserve);

    // 在mGrid中记录了各特征点
    for(int i=0; i<this->N; i++)
    {
        int gridPosX = round((this->mvKps[i].x)*mGridElementWidthInv);
        int gridPosY = round((this->mvKps[i].y)*mGridElementHeightInv);

        if(gridPosX<0 || gridPosX>=FRAME_GRID_COLS || gridPosY<0 || gridPosY>=FRAME_GRID_ROWS)
            continue;

        mGrid[gridPosX][gridPosY].push_back(i);
    }
}


/**
 * @brief 找到在 以x,y为中心,边长为r的方形内的特征点
 * @param x        图像坐标u
 * @param y        图像坐标v
 * @param l        边长
 * @return         满足条件的特征点的序号
 */
std::vector<int> Frame::GetFeaturesInArea(const float& x, const float& y, const float& lx, const float& ly)
{
    std::vector<int> vIndices;
    vIndices.reserve(this->N);

    float rx = lx/2;
    float ry = ly/2;

    //boundry
    int minCellX = std::max(0,(int)floor((x-rx)*mGridElementWidthInv));
    if(minCellX>=FRAME_GRID_COLS)
        return vIndices;

    int maxCellX = std::min((int)FRAME_GRID_COLS-1,(int)ceil((x+rx)*mGridElementWidthInv));
    if(maxCellX<0)
        return vIndices;

    int minCellY = std::max(0,(int)floor((y-ry)*mGridElementHeightInv));
    if(minCellY>=FRAME_GRID_ROWS)
        return vIndices;

    int maxCellY = std::min((int)FRAME_GRID_ROWS-1,(int)ceil((y+ry)*mGridElementHeightInv));
    if(maxCellY<0)
        return vIndices;

    //find key points in the boundry
    for(int i=minCellX; i<=maxCellX; i++)
    {
        for(int j=minCellY; j<=maxCellY; j++)
        {
            std::vector<int> vCell = mGrid[i][j];
            if(vCell.empty())
                continue;

            for(int k=0; k<vCell.size(); k++)
            {
                cv::Point2f pt = this->mvKps[vCell[k]];
                    
                float distx = pt.x-x;
                float disty = pt.y-y;

                if(fabs(distx)<rx && fabs(disty)<ry)
                    vIndices.push_back(vCell[k]);
            }
        }
    }

    return vIndices;
}

float Frame::GetAverageDepthInArea(const float& x, const float& y)
{
    //boundry
    int cellX = (int)x*mGridElementWidthInv;
    int cellY = (int)y*mGridElementHeightInv;

    if(cellX>=FRAME_GRID_COLS || cellX<0)
        return -1.0;

    if(cellY>=FRAME_GRID_ROWS || cellY<0)
        return -1.0;

    std::vector<int> vCell = mGrid[cellX][cellY];
    if(vCell.empty())
        return -1.0;

    float avgDepth = 0.0;
    int n = 0;

    for(int k=0; k<vCell.size(); k++)
    {
        if(this->mvKpsDepth[vCell[k]] != -1.0 )
        {
            avgDepth += this->mvKpsDepth[vCell[k]];
            n++;
        }
    }

    if(n==0)
        return -1.0;
    else
        return avgDepth/n;
}






cv::Point2f Frame::Cam2Px(const cv::Point3f& pCam)
{
    cv::Point2f px = cv::Point2f(pCam.x*fx/pCam.z + cx, pCam.y*fy/pCam.z + cy);
    return px;
}

cv::Point3f Frame::Px2Cam(const cv::Point2f& px)
{
    cv::Point3f pCam = cv::Point3f((px.x-cx)/fx, (px.y-cy)/fy, 1);
    return pCam;
}

cv::Point3f Frame::World2Cam(const cv::Point3f& pWorld)
{
    cv::Mat pCam = this->mRcw*Converter::toCvMat(pWorld) + this->mtcw;
    return Converter::toCvPoint3f(pCam);
}

cv::Point3f Frame::Cam2World(const cv::Point3f& pCam)
{
    cv::Mat pWorld = this->mRwc*Converter::toCvMat(pCam) + this->mOw;
    return Converter::toCvPoint3f(pWorld);
}