#include <hrl_head_registration/head_registration.h>

#define USE_COLOR_MODEL

#define KINECT2 
#ifdef KINECT1
#define KINECT_WIDTH 640
#else
#define KINECT_WIDTH 960
#endif 

void extractSkinPC(const PCRGB::Ptr& pc_in, PCRGB::Ptr& pc_out, double thresh) 
{
    uint8_t r, g, b; 
    double skin_like, h, s, l;
    for(uint32_t i=0;i<pc_in->size();i++) {
        if(PT_IS_NOT_NAN(pc_in, i)) {
            extractRGB(pc_in->points[i].rgb, r, g, b);
            skin_like = skin_likelihood(r, g, b);
            if(skin_like > thresh) 
                COPY_PT_INTO_CLOUD(pc_in, pc_out, i);
        }
    }
}

int32_t findClosestPoint(const PCRGB::Ptr& pc, uint32_t u, uint32_t v)
{
    if(PT_IS_NOT_NAN(pc, v*KINECT_WIDTH + u))
        return v*KINECT_WIDTH + u;
    for(uint32_t i=1;i<5;i++) {
        for(uint32_t j=1;j<5;j++) {
            if(PT_IS_NOT_NAN(pc, v*KINECT_WIDTH + u + i))
                return v*KINECT_WIDTH + u + i;
            if(PT_IS_NOT_NAN(pc, v*KINECT_WIDTH + u - i))
                return v*KINECT_WIDTH + u + i;
            if(PT_IS_NOT_NAN(pc, v*KINECT_WIDTH + u + j*KINECT_WIDTH))
                return v*KINECT_WIDTH + u + i;
            if(PT_IS_NOT_NAN(pc, v*KINECT_WIDTH + u - j*KINECT_WIDTH))
                return v*KINECT_WIDTH + u + i;
        }
    }
    return -1;
}

void generateColorModel(const PCRGB::Ptr& pc_in, Eigen::Vector3d& mean, Eigen::Matrix3d& cov)
{
    Eigen::MatrixXd X(3, pc_in->size());
    for(uint32_t i=0;i<pc_in->size();i++)
        extractHSL(pc_in->points[i].rgb, X(0,i), X(1,i), X(2,i));
    mean = X.rowwise().sum() / pc_in->size();
    cov = X * X.transpose() / pc_in->size() - mean * mean.transpose();
}

bool extractPCFromColorModel(const PCRGB::Ptr& pc_in, PCRGB::Ptr& pc_out,
                             Eigen::Vector3d& mean, Eigen::Matrix3d& cov, double std_devs)
{
    double hue_weight;
    ros::param::param<double>("~hue_weight", hue_weight, 1.0);
    printf("%f\n", hue_weight);

    Eigen::Vector3d x, x_m;
    Eigen::Matrix3d cov_inv = cov.inverse();
    for(uint32_t i=0;i<pc_in->size();i++) {
        extractHSL(pc_in->points[i].rgb, x(0), x(1), x(2));
        x_m = x - mean;
        x_m(0) *= hue_weight;
        double dist = std::sqrt(x_m.transpose() * cov_inv * x_m);
        if(dist <= std_devs) 
            pc_out->points.push_back(pc_in->points[i]);
    }
}

void sphereTrim(const PCRGB::Ptr& pc_in, PCRGB::Ptr& pc_out, uint32_t ind, double radius)
{
    KDTree::Ptr kd_tree(new pcl::KdTreeFLANN<PRGB> ());
    kd_tree->setInputCloud(pc_in);
    vector<int> inds;
    vector<float> dists;
    kd_tree->radiusSearch(*pc_in, ind, radius, inds, dists);
    for(uint32_t j=0;j<inds.size();j++)
        COPY_PT_INTO_CLOUD(pc_in, pc_out, inds[j]);
}

namespace pcl {

class ColorPointRepresentation : public PointRepresentation<PRGB> {
public:
    ColorPointRepresentation(float h_mult = 0.0005, float s_mult = 0.0005, float l_mult = 0.0005) {
        nr_dimensions_ = 6;
        alpha_.resize(6);
        alpha_[0] = 1; alpha_[1] = 1; alpha_[2] = 1;
        alpha_[3] = h_mult; alpha_[4] = s_mult; alpha_[5] = l_mult;
    }
    virtual void copyToFloatArray(const PRGB& p, float* out) const {
        double h, s, l;
        extractHSL(p.rgb, h, s, l);
        out[0] = p.x; out[1] = p.y; out[2] = p.z;
        out[3] = h; out[4] = s; out[5] = l;
    }
    //bool isValid<PRGB>(const PRGB& p) const {
    //    if(p.x == p.x && p.y == p.y && p.z == p.z)
    //        return true;
    //    else
    //        return false;
    //}
};
}

void computeICPRegistration(const PCRGB::Ptr& target_pc, const PCRGB::Ptr& source_pc,
                            Eigen::Affine3d& tf_mat, int max_iters, double color_weight) 
                            {
    double icp_trans_eps, icp_max_corresp;
    ros::param::param<double>("~icp_max_corresp", icp_max_corresp, 0.1);
    ros::param::param<double>("~icp_trans_eps", icp_trans_eps, 1e-6);
    pcl::IterativeClosestPoint<PRGB, PRGB> icp;
    PCRGB::Ptr aligned_pc(new PCRGB());
    boost::shared_ptr<pcl::PointRepresentation<PRGB> const> 
        pt_rep(new pcl::ColorPointRepresentation(color_weight, color_weight, color_weight));
    icp.setPointRepresentation(pt_rep);
    icp.setInputTarget(target_pc);
    icp.setInputSource(source_pc);
    icp.setTransformationEpsilon(icp_trans_eps);
    icp.setMaxCorrespondenceDistance(icp_max_corresp);
    icp.setMaximumIterations(max_iters);
    icp.align(*aligned_pc);
    tf_mat = icp.getFinalTransformation().cast<double>();
    /*
    icp.setTransformationEpsilon(1e-4);
    icp.setMaxCorrespondenceDistance(0.5);
    Eigen::Matrix4f cur_tf = Eigen::Matrix4f::Identity(), last_tf;
    PCRGB::Ptr last_aligned_pc;

    for(int i=0;i<max_iters;i++) {
        last_aligned_pc = aligned_pc;
        icp.setInputCloud(last_aligned_pc);
        
        icp.setMaximumIterations(2);
        icp.align(*aligned_pc);
        cur_tf = icp.getFinalTransformation() * cur_tf;
        if (fabs ((icp.getLastIncrementalTransformation () - last_tf).sum ()) < 
                   icp.getTransformationEpsilon ())
            icp.setMaxCorrespondenceDistance (icp.getMaxCorrespondenceDistance () - 0.001);
        last_tf = icp.getLastIncrementalTransformation();
    }
    tf_mat = cur_tf.cast<double>();
    */

#if 0
    vector<PCRGB::Ptr> pcs;
    vector<string> pc_topics;
    aligned_pc->header.frame_id = "/openni_rgb_optical_frame";
    source_pc->header.frame_id = "/openni_rgb_optical_frame";
    target_pc->header.frame_id = "/openni_rgb_optical_frame";
    pcs.push_back(aligned_pc);
    pcs.push_back(source_pc);
    pcs.push_back(target_pc);
    pc_topics.push_back("/aligned_pc");
    pc_topics.push_back("/source_pc");
    pc_topics.push_back("/target_pc");
    pubLoop(pcs, pc_topics, 5.0, 5);
#endif
}

void extractFace(const PCRGB::Ptr& input_pc, PCRGB::Ptr& out_pc, int u_click, int v_click)
{
    double trim_radius, skin_thresh;
    ros::param::param<double>("~trim_radius", trim_radius, 0.12);
    ros::param::param<double>("~skin_thresh", skin_thresh, 0.8);

    PCRGB::Ptr trimmed_pc(new PCRGB());
    int32_t closest_ind = findClosestPoint(input_pc, u_click, v_click);
    if(closest_ind < 0)
        return;
    sphereTrim(input_pc, trimmed_pc, closest_ind, trim_radius);
    extractSkinPC(trimmed_pc, out_pc, skin_thresh);
}

void extractFaceColorModel(const PCRGB::Ptr& input_pc, PCRGB::Ptr& out_pc, int u_click, int v_click)
{
    double trim_radius, model_radius, color_std_thresh;
    ros::param::param<double>("~trim_radius", trim_radius, 0.12);
    ros::param::param<double>("~model_radius", model_radius, 0.02);
    ros::param::param<double>("~color_std_thresh", color_std_thresh, 1.5);

    int32_t closest_ind = findClosestPoint(input_pc, u_click, v_click);
    if(closest_ind < 0)
        return;

    PCRGB::Ptr model_pc(new PCRGB());
    sphereTrim(input_pc, model_pc, closest_ind, model_radius);
    Eigen::Vector3d mean_color;
    Eigen::Matrix3d cov_color;
    generateColorModel(model_pc, mean_color, cov_color);

    PCRGB::Ptr trimmed_pc(new PCRGB());
    sphereTrim(input_pc, trimmed_pc, closest_ind, trim_radius);
    extractPCFromColorModel(trimmed_pc, out_pc, mean_color, cov_color, color_std_thresh);
}

bool findFaceRegistration(const PCRGB::Ptr& template_pc, const PCRGB::Ptr& input_pc,
                          int u_click, int v_click, Eigen::Affine3d& tf_mat)
{
    double color_weight;
    int max_iters;
    ros::param::param<double>("~color_weight", color_weight, 0.0);
    ros::param::param<int>("~max_iters", max_iters, 200);
    PCRGB::Ptr skin_pc(new PCRGB());

#ifdef USE_COLOR_MODEL
    extractFaceColorModel(input_pc, skin_pc, u_click, v_click);
#else
    extractFace(input_pc, skin_pc, u_click, v_click);
#endif
    if(skin_pc->size() == 0)
        return false;
    computeICPRegistration(template_pc, skin_pc, tf_mat, max_iters, color_weight);

    return true;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "head_registration");
    ros::NodeHandle nh;

#if 0
    PCRGB::Ptr input_pc, template_pc;
    readPCBag(argv[1], template_pc);
    readPCBag(argv[2], input_pc);
    int u, v;
    FILE* file = fopen(argv[3], "r");
    fscanf(file, "%d,%d\n", &u, &v);
    fclose(file);

    Eigen::Affine3d tf_mat;
    findFaceRegistration(template_pc, input_pc, u, v, tf_mat);

    PCRGB::Ptr tf_pc(new PCRGB());
    transformPC(*input_pc, *tf_pc, tf_mat);
    tf_pc->header.frame_id = "/base_link";
    pubLoop(tf_pc, "test", 5);
    return 0;
#endif

#if 1
    PCRGB::Ptr input_pc(new PCRGB()), face_extract_pc(new PCRGB());
    readPCBag(argv[1], input_pc);
    int u, v;
    FILE* file = fopen(argv[2], "r");
    fscanf(file, "%d,%d\n", &u, &v);
    fclose(file);
    Eigen::Affine3d tf_mat;
    extractFaceColorModel(input_pc, face_extract_pc, u, v);
    face_extract_pc->header.frame_id = "/base_link";
    pubLoop(face_extract_pc, "/test", 5);
#endif

}
