#include <opencv/cv.h>
#include <opencv2/highgui/highgui.hpp>
#include <boost/math/distributions/binomial.hpp>

#include <hrl_head_registration/skin_color_util.h>

using namespace std;
using namespace boost::math;

int main(int argc, char **argv)
{
    cv::Mat img_in = cv::imread("in.jpg", 1);
    cv::Mat img_label(img_in.size(), CV_32F);
    cv::Mat img_label2(img_in.size(), CV_32F, cv::Scalar(0));
    cv::Mat img_label3(img_in.size(), CV_32F, cv::Scalar(0));
    double max_skin_prob = 0, skin_like, skin_prob, nskin_prob;
    vector<double> skin_likes;
    for(int i=0;i<img_in.rows;i++) {
        for(int j=0;j<img_in.cols;j++) {
            skin_prob = gaussian_mix_skin(img_in.at<cv::Vec3b>(i,j)[2], 
                                          img_in.at<cv::Vec3b>(i,j)[1], 
                                          img_in.at<cv::Vec3b>(i,j)[0]);
            nskin_prob = gaussian_mix_nskin(img_in.at<cv::Vec3b>(i,j)[2], 
                                            img_in.at<cv::Vec3b>(i,j)[1], 
                                            img_in.at<cv::Vec3b>(i,j)[0]);
            skin_like = skin_prob / nskin_prob;
            if(skin_like > max_skin_prob)
                max_skin_prob = skin_like;
            img_label.at<float>(i,j) = skin_like;
            if(skin_like > atof(argv[1]))
                img_label2.at<float>(i,j) = 255;
            skin_likes.push_back(skin_like);
        }
    }
    sort(skin_likes.begin(), skin_likes.end());
    double skin_like_thresh = skin_likes[(int) (atof(argv[2]) * skin_likes.size())];
    for(int i=0;i<img_in.rows;i++) 
        for(int j=0;j<img_in.cols;j++) 
            if(img_label.at<float>(i,j) > skin_like_thresh)
                img_label3.at<float>(i,j) = 255;

    img_label /= max_skin_prob / 255;
    cv::imwrite("out.jpg", img_label);
    cv::imwrite("out2.jpg", img_label2);
    cv::imwrite("out3.jpg", img_label3);
}
