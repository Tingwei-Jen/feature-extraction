#ifndef CONVERTER_H
#define CONVERTER_H
#include <opencv2/core/core.hpp>
#include <Eigen/Dense>

/**
 * @brief 提供了一些常见的转换
 * 
 * orb中以cv::Mat为基本存储结构，到g2o和Eigen需要一个转换
 * 这些转换都很简单，整个文件可以单独从orbslam里抽出来而不影响其他功能
 */
class Converter
{
public:
    /**
     * @brief 一个描述子矩阵到一串单行的描述子向量
     */
    static std::vector<cv::Mat> toDescriptorVector(const cv::Mat &Descriptors);

    /**
     * @name toCvMat
     */
    ///@{
    static cv::Mat toCvMat(const Eigen::Matrix<double,4,4> &m);
    static cv::Mat toCvMat(const Eigen::Matrix3d &m);
    static cv::Mat toCvMat(const Eigen::Matrix<double,3,1> &m);
    static cv::Mat toCvMat(const cv::Point3d &cvPoint);
    static cv::Mat toCvMat(const cv::Point3f &cvPoint);

    ///@}

    /**
     * @name toEigen
     */
    ///@{
    static Eigen::Matrix<double,3,1> toVector3d(const cv::Mat &cvVector);
    static Eigen::Matrix<double,3,1> toVector3d(const cv::Point3f &cvPoint);
    static Eigen::Matrix<double,3,3> toMatrix3d(const cv::Mat &cvMat3);
    static std::vector<float> toQuaternion(const cv::Mat &M);
    ///@}


    /**
     * @name toCvPoint
     */
    ///@{
    static cv::Point3d toCvPoint3d(const cv::Mat &cvVector);
    static cv::Point3f toCvPoint3f(const cv::Mat &cvVector);
    ///@}   
};
#endif // CONVERTER_H
