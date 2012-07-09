
#include <hrl_head_registration/head_registration.h>

void expandPC(const PCRGB::Ptr& pc_label, const PCRGB::Ptr& pc_base, 
              PCRGB::Ptr& pc_expand, double radius)
{
    KDTree::Ptr kd_tree(new pcl::KdTreeFLANN<PRGB> ());
    kd_tree->setInputCloud(pc_base);
    vector<int> inds;
    vector<float> dists;
    for(uint32_t i=0;i<pc_label->size();i++) {
        inds.clear(); dists.clear();
        kd_tree->radiusSearch(*pc_label, i, radius, inds, dists);
        for(uint32_t j=0;j<inds.size();j++)
            COPY_PT_INTO_CLOUD(pc_label, pc_expand, inds[j]);
    }
}

/*
void extractHeadCluster(const PCRGB::Ptr& pc_label, PCRGB::Ptr& pc_out, PRGB& seed)
{
    double pc_clust_dist;
    int pc_clust_min_size;
    ros::param::param<double>("~pc_clust_dist", pc_clust_dist, 0.02);
    ros::param::param<int>("~pc_clust_min_size", pc_clust_min_size, 20);

    // Extracts Euclidean clusters
    pcl::PointIndices::Ptr inds_ptr (new pcl::PointIndices());
    for(uint32_t i=0;i<pc_label->size();i++) 
        inds_ptr->indices.push_back(i);
    pcl::EuclideanClusterExtraction<PRGB> pc_clust;
    pcl::KdTree<PRGB>::Ptr clust_tree (new pcl::KdTreeFLANN<PRGB> ());
    pc_clust.setClusterTolerance(pc_clust_dist);
    pc_clust.setMinClusterSize(pc_clust_min_size);
    pc_clust.setIndices(inds_ptr);
    pc_clust.setInputCloud(pc_label);
    pc_clust.setSearchMethod(clust_tree);
    std::vector<pcl::PointIndices> pc_clust_list;
    pc_clust.extract(pc_clust_list);

    // find the cluster closest our seed
    uint32_t closest_clust = 0;
    for(uint32_t i=0;i<pc_clust_list.size();i++) {
        PCRGB::Ptr pc_clust(new PCRGB());
        for(uint32_t j=0;j<pc_clust_list[i].indices.size();j++)
            COPY_PT_INTO_CLOUD(pc_label, pc_clust, j);
        KDTree::Ptr kd_tree(new pcl::KdTreeFLANN<PRGB> ());
        kd_tree->setInputCloud(pc_clust);
        // TODO NN TO COMPUTE DIST
    }

    // extract cluster into output
    for(uint32_t i=0;i<pc_clust_list[closest_clust].indices.size();i++) 
        COPY_PT_INTO_CLOUD(pc_label, pc_out, pc_clust_list[closest_clust].indices[i]);
}
*/

int main(int argc, char **argv)
{
    ros::init(argc, argv, "head_training");
    ros::NodeHandle nh;
    PCRGB::Ptr input_pc;
    readPCBag(argv[1], input_pc);
    PCRGB::Ptr skin_pc(new PCRGB());
    PCRGB::Ptr expanded_pc(new PCRGB());
    PCRGB::Ptr trimmed_pc(new PCRGB());

    uint32_t closest_ind = findClosestPoint(input_pc, atoi(argv[3]), atoi(argv[4]));
    sphereTrim(input_pc, trimmed_pc, closest_ind, atof(argv[5]));
    extractSkinPC(trimmed_pc, skin_pc, atof(argv[2]));

    skin_pc->header.frame_id = "/base_link";
    savePCBag("skin_pc.bag", skin_pc);
    pubLoop(skin_pc, "test", 5);
    return 0;
}
