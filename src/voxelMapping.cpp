#include <omp.h>
#include <mutex>
#include <cmath>
#include <thread>
#include <fstream>
#include <csignal>
#include <unistd.h>
// #include <Python.h>
#include <so3_math.h>
#include <ros/ros.h>
#include <Eigen/Core>
#include "IMU_Processing.h"
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/voxel_grid.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/Vector3.h>
#include <livox_ros_driver/CustomMsg.h>
#include "preprocess.h"
#include "voxel_map_util.hpp"
#include "pv_lio/save_map.h"
// #include "imuProcessing.h"
using namespace std;
using namespace Eigen;

#define INIT_TIME           (0.1)
#define LASER_POINT_COV     (0.001)
#define MAXN                (720000)
#define PUBFRAME_PERIOD     (20)

/*** Time Log Variables ***/
// double T1[MAXN], s_plot[MAXN], s_plot2[MAXN], s_plot3[MAXN], s_plot4[MAXN], s_plot5[MAXN], s_plot6[MAXN], s_plot7[MAXN], s_plot8[MAXN], s_plot9[MAXN], s_plot10[MAXN], s_plot11[MAXN];
double match_time = 0, solve_time = 0, solve_const_H_time = 0;
bool   time_sync_en = false, extrinsic_est_en = true, path_en = true;
double lidar_time_offset = 0.0;
/**************************/

float DET_RANGE = 300.0f;
const float MOV_THRESHOLD = 1.5f;
double R_inv , acc_norm;
std::mutex mtx_buffer;
// condition_variable sig_buffer;

string root_dir = ROOT_DIR;
string  lid_topic, imu_topic;

double res_mean_last = 0.05, total_residual = 0.0;
double last_timestamp_lidar = 0, last_timestamp_imu = -1.0;
double gyr_cov = 0.1, acc_cov = 0.1, b_gyr_cov = 0.0001, b_acc_cov = 0.0001;
double filter_size_surf_min = 0;
double decayed_time = 0;
double total_distance = 0, lidar_end_time = 0, first_lidar_time = 0.0;
int    effct_feat_num = 0, time_log_counter = 0, scan_count = 0, publish_count = 0;
int    iterCount = 0, feats_down_size = 0, NUM_MAX_ITERATIONS = 0, laserCloudValidNum = 0, pcd_index = 0;
bool   lidar_pushed, flg_first_scan = true, flg_exit = false, flg_EKF_inited;
bool   scan_pub_en = false, dense_pub_en = false, scan_body_pub_en = false;

bool pcd_save_en = false;
int pcd_save_interval = -1;

PointCloudXYZI::Ptr pcl_wait_save(new PointCloudXYZI());
std::string pcd_file_prefix;

vector<double>       extrinT(3, 0.0);
vector<double>       extrinR(9, 0.0);
vector<double>       gravity = {0, 0, -9.81};

deque<double>                     time_buffer;
deque<PointCloudXYZI::Ptr>        lidar_buffer;
deque<sensor_msgs::Imu::ConstPtr> imu_buffer;

PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI());
std::vector<M3D> var_down_body;

pcl::VoxelGrid<PointType> downSizeFilterSurf;

V3D Lidar_T_wrt_IMU = V3D(0, 0, 0);
M3D Lidar_R_wrt_IMU = M3D::Identity();

// params for voxel mapping algorithm
double min_eigen_value = 0.003;
int max_layer = 0;

int max_cov_points_size = 50;
int max_points_size = 50;
double sigma_num = 2.0;
double max_voxel_size = 1.0;
std::vector<int> layer_size;

double ranging_cov = 0.0;
double angle_cov = 0.0;
std::vector<double> layer_point_size;

bool publish_voxel_map = false;
int publish_max_voxel_layer = 0;

std::unordered_map<VOXEL_LOC, std::shared_ptr<OctoTree>> voxel_map;

/*** EKF inputs and output ***/
MeasureGroup Measures;
esekfom::esekf<state_ikfom, 12, input_ikfom> kf;
state_ikfom state_point;

nav_msgs::Path path;
nav_msgs::Odometry odomAftMapped;
geometry_msgs::Quaternion geoQuat;
geometry_msgs::PoseStamped msg_body_pose;

shared_ptr<Preprocess> p_pre(new Preprocess());
shared_ptr<ImuProcess> p_imu(new ImuProcess());

// void SigHandle(int sig)
// {
//     flg_exit = true;
//     ROS_WARN("catch sig %d", sig);
//     // sig_buffer.notify_all();
// }

// const bool var_contrast(pointWithCov &x, pointWithCov &y) {
//     return (x.cov.diagonal().norm() < y.cov.diagonal().norm());
// };
PointCloudXYZI::Ptr pcl_wait_save_(new PointCloudXYZI());

bool saveMapService(pv_lio::save_map::Request& req, pv_lio::save_map::Response& res)
{
    //   string saveMapDirectory;
    
      cout << "****************************************************" << endl;
      cout << "Saving map to pcd files ..." << endl;
    //   if(req.destination.empty()) saveMapDirectory = std::getenv("HOME") + savePCDDirectory;
    //   else saveMapDirectory = std::getenv("HOME") + req.destination;
    //   cout << "Save destination: " << saveMapDirectory << endl;
    std::string p = req.destination;
    float leaf_size = req.resolution;

    pcl::VoxelGrid<PointType> vg;
    vg.setLeafSize(leaf_size, leaf_size, leaf_size);
    vg.setInputCloud(pcl_wait_save);
    vg.filter(*pcl_wait_save_);
    int ret = pcl::io::savePCDFileBinary(p + "/scans.pcd", *pcl_wait_save_);
    res.success = ret == 0;

    if (res.success) {
        std::cout << "done" << std::endl;
    } else {
        std::cout << "failed" << std::endl;
    }

    // if (pcl_wait_save->size() > 0 && pcd_save_en)
    // {
    //     string file_name = string("scans.pcd");
    //     string all_points_dir(string(string("/point-lio/src/Point-LIO/") + "PCD/") + file_name);
    //     pcl::PCDWriter pcd_writer;
    //     pcd_writer.writeBinary("/point-lio/src/Point-LIO/PCD/scans.pcd", *pcl_wait_save);
    // }
    return res.success;
}

const bool var_contrast_const(pointWithCov &x, pointWithCov &y) {
    return (x.cov_diagonal_norm < y.cov_diagonal_norm );
};


void RGBpointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    // po->intensity = pi->intensity;
    po->intensity = pi->normal_y;
}

void RGBpointBodyLidarToIMU(PointType const * const pi, PointType * const po)
{
    V3D p_body_lidar(pi->x, pi->y, pi->z);
    V3D p_body_imu(state_point.offset_R_L_I*p_body_lidar + state_point.offset_T_L_I);

    po->x = p_body_imu(0);
    po->y = p_body_imu(1);
    po->z = p_body_imu(2);
    po->intensity = pi->intensity;

    po->curvature = pi->curvature;
    po->normal_x = pi->normal_x;
}

void standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg)
{
    auto time_offset = lidar_time_offset;
//    std::printf("lidar offset:%f\n", lidar_time_offset);
    std::unique_lock<std::mutex> ul(mtx_buffer, std::defer_lock);
    scan_count ++;
    double preprocess_start_time = omp_get_wtime();
    if (msg->header.stamp.toSec() + time_offset < last_timestamp_lidar)
    {
        ROS_ERROR("lidar loop back, clear buffer");
        ul.lock();
        lidar_buffer.clear();
        ul.unlock();
    }

    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    ul.lock();
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(msg->header.stamp.toSec() + time_offset);
    last_timestamp_lidar = msg->header.stamp.toSec() + time_offset;
    ul.unlock();
}


void livox_pcl_cbk(const livox_ros_driver::CustomMsg::ConstPtr &msg)
{
    static double timediff_lidar_wrt_imu = 0.0;
    static bool   timediff_set_flg = false;
    std::unique_lock<std::mutex> ul(mtx_buffer, std::defer_lock);
    double preprocess_start_time = omp_get_wtime();
    scan_count ++;
    if (msg->header.stamp.toSec() < last_timestamp_lidar)
    {
        ROS_ERROR("lidar loop back, clear buffer");
        ul.lock();
        lidar_buffer.clear();
        ul.unlock();
    }
    ul.lock();
    last_timestamp_lidar = msg->header.stamp.toSec();

    if (!time_sync_en && abs(last_timestamp_imu - last_timestamp_lidar) > 10.0 && !imu_buffer.empty() && !lidar_buffer.empty() )
    {
        printf("IMU and LiDAR not Synced, IMU time: %lf, lidar header time: %lf \n",last_timestamp_imu, last_timestamp_lidar);
    }

    if (time_sync_en && !timediff_set_flg && abs(last_timestamp_lidar - last_timestamp_imu) > 1 && !imu_buffer.empty())
    {
        timediff_set_flg = true;
        timediff_lidar_wrt_imu = last_timestamp_lidar + 0.1 - last_timestamp_imu;
        printf("Self sync IMU and LiDAR, time diff is %.10lf \n", timediff_lidar_wrt_imu);
    }
    ul.unlock();

    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    ul.lock();
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(last_timestamp_lidar);
    ul.unlock();
}

void imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in)
{
    // publish_count ++;
    // cout<<"IMU got at: "<<msg_in->header.stamp.toSec()<<endl;
    sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));

    // if (abs(timediff_lidar_wrt_imu) > 0.1 && time_sync_en)
    // {
    //     msg->header.stamp = \
    //     ros::Time().fromSec(timediff_lidar_wrt_imu + msg_in->header.stamp.toSec());
    // }

    double timestamp = msg->header.stamp.toSec();

    if (timestamp < last_timestamp_imu)
    {
//        ROS_WARN("imu loop back, clear buffer");
//        imu_buffer.clear();
        ROS_WARN("imu loop back, ignoring!!!");
        ROS_WARN("current T: %f, last T: %f", timestamp, last_timestamp_imu);
        return;
    }
    // 剔除异常数据
    if (std::abs(msg->angular_velocity.x) > 10
        || std::abs(msg->angular_velocity.y) > 10
        || std::abs(msg->angular_velocity.z) > 10) {
        ROS_WARN("Large IMU measurement!!! Drop Data!!! %.3f  %.3f  %.3f",
                 msg->angular_velocity.x,
                 msg->angular_velocity.y,
                 msg->angular_velocity.z
        );
        return;
    }

//    // 如果是第一帧 拿过来做重力对齐
//    // TODO 用多帧平均的重力
//    if (is_first_imu) {
//        double acc_vec[3] = {msg_in->linear_acceleration.x, msg_in->linear_acceleration.y, msg_in->linear_acceleration.z};
//
//        R__world__o__initial = SO3(g2R(Eigen::Vector3d(acc_vec)));
//
//        is_first_imu = false;
//    }

    std::lock_guard<std::mutex> lg(mtx_buffer);
    last_timestamp_imu = timestamp;
    imu_buffer.push_back(msg);
    // sig_buffer.notify_all();
}

double lidar_mean_scantime = 0.0;
int    scan_num = 0;
bool sync_packages(MeasureGroup &meas)
{
    std::lock_guard<std::mutex> lg(mtx_buffer);
    if (lidar_buffer.empty() || imu_buffer.empty()) {
        return false;
    }

    /*** push a lidar scan ***/
    if(!lidar_pushed)
    {
        meas.lidar = lidar_buffer.front();
        meas.lidar_beg_time = time_buffer.front();
        if (meas.lidar->points.size() <= 1) // time too little
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
            ROS_WARN("Too few input point cloud!\n");
        }
        else if (meas.lidar->points.back().curvature / double(1000) < 0.5 * lidar_mean_scantime)
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
        }
        else
        {
            scan_num ++;
            lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000);
            lidar_mean_scantime += (meas.lidar->points.back().curvature / double(1000) - lidar_mean_scantime) / scan_num;
        }
        meas.lidar_end_time = lidar_end_time;
        lidar_pushed = true;
    }

    if (last_timestamp_imu < lidar_end_time)
    {
        return false;
    }

    /*** push imu data, and pop from imu buffer ***/
    double imu_time = imu_buffer.front()->header.stamp.toSec();
    meas.imu.clear();
    while ((!imu_buffer.empty()) && (imu_time < lidar_end_time))
    {
        imu_time = imu_buffer.front()->header.stamp.toSec();
        if(imu_time > lidar_end_time) break;
        meas.imu.push_back(imu_buffer.front());
        imu_buffer.pop_front();
    }

    lidar_buffer.pop_front();
    time_buffer.pop_front();
    lidar_pushed = false;
    return true;
}

void publish_frame_world(const ros::Publisher & pubLaserCloudFull)
{
    PointCloudXYZI::Ptr laserCloudWorld;

    if(scan_pub_en || pcd_save_en)
    {
        PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body);
        int size = laserCloudFullRes->points.size();
        laserCloudWorld = PointCloudXYZI::Ptr(new PointCloudXYZI(size, 1));
#ifdef MP_EN
        #pragma omp parallel for
#endif
        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&laserCloudFullRes->points[i], &laserCloudWorld->points[i]);
        }

        if(scan_pub_en)
        { 
            sensor_msgs::PointCloud2::Ptr laserCloudmsg(new sensor_msgs::PointCloud2());
            pcl::toROSMsg(*laserCloudWorld, *laserCloudmsg);
            laserCloudmsg->header.stamp = ros::Time().fromSec(lidar_end_time);
            laserCloudmsg->header.frame_id = "camera_init";
            pubLaserCloudFull.publish(laserCloudmsg);
            // publish_count -= PUBFRAME_PERIOD;
        }

        /**************** save map ****************/
        /* 1. make sure you have enough memories
        /* 2. noted that pcd save will influence the real-time performences **/
        if (pcd_save_en)
        {        
            *pcl_wait_save += *laserCloudWorld;

            static int scan_wait_num = 0;
            scan_wait_num ++;
            if (pcl_wait_save->size() > 0 && pcd_save_interval > 0  && scan_wait_num >= pcd_save_interval)
            {
                std::string all_points_dir(pcd_file_prefix + "-" + std::to_string(pcd_index++) + ".pcd");
                pcl::PCDWriter pcd_writer;
                cout << "current scan saved to /PCD/" << all_points_dir << endl;
                pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
                pcl_wait_save->clear();
                scan_wait_num = 0;
            }
        }
    }

}

void publish_frame_body(const ros::Publisher & pubLaserCloudFull_body)
{
//    int size = feats_undistort->points.size();
    PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body);
    int size = laserCloudFullRes->points.size();
    PointCloudXYZI::Ptr laserCloudIMUBody(new PointCloudXYZI(size, 1));
#ifdef MP_EN
    #pragma omp parallel for
#endif
    for (int i = 0; i < size; i++)
    {
        RGBpointBodyLidarToIMU(&laserCloudFullRes->points[i],
                               &laserCloudIMUBody->points[i]);
    }

    sensor_msgs::PointCloud2::Ptr laserCloudmsg(new sensor_msgs::PointCloud2());
    pcl::toROSMsg(*laserCloudIMUBody, *laserCloudmsg);
    laserCloudmsg->header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudmsg->header.frame_id = "body";
    pubLaserCloudFull_body.publish(laserCloudmsg);
    // publish_count -= PUBFRAME_PERIOD;
}

// void publish_map(const ros::Publisher & pubLaserCloudMap)
// {
//     sensor_msgs::PointCloud2 laserCloudMap;
//     pcl::toROSMsg(*featsFromMap, laserCloudMap);
//     laserCloudMap.header.stamp = ros::Time().fromSec(lidar_end_time);
//     laserCloudMap.header.frame_id = "camera_init";
//     pubLaserCloudMap.publish(laserCloudMap);
// }

template<typename T>
void set_posestamp_1(T & out)
{
    out.pose.position.x = state_point.pos(0);
    out.pose.position.y = state_point.pos(1);
    out.pose.position.z = state_point.pos(2);
    out.pose.orientation.x = geoQuat.x;
    out.pose.orientation.y = geoQuat.y;
    out.pose.orientation.z = geoQuat.z;
    out.pose.orientation.w = geoQuat.w;
}

template<typename T>
void set_posestamp_2(T & out)
{
    out.pose.pose.position.x = state_point.pos(0);
    out.pose.pose.position.y = state_point.pos(1);
    out.pose.pose.position.z = state_point.pos(2);
    out.pose.pose.orientation.x = geoQuat.x;
    out.pose.pose.orientation.y = geoQuat.y;
    out.pose.pose.orientation.z = geoQuat.z;
    out.pose.pose.orientation.w = geoQuat.w;
    out.twist.twist.linear.x = R_inv;
}

void publish_odometry(const ros::Publisher & pubOdomAftMapped)
{
    odomAftMapped.header.frame_id = "camera_init";
    odomAftMapped.child_frame_id = "body";
    odomAftMapped.header.stamp = ros::Time().fromSec(lidar_end_time);// ros::Time().fromSec(lidar_end_time);
    set_posestamp_2(odomAftMapped);
    pubOdomAftMapped.publish(odomAftMapped);
    auto P = kf.get_P();
    for (int i = 0; i < 6; i ++)
    {
        int k = i < 3 ? i + 3 : i - 3;
        odomAftMapped.pose.covariance[i*6 + 0] = P(k, 3);
        odomAftMapped.pose.covariance[i*6 + 1] = P(k, 4);
        odomAftMapped.pose.covariance[i*6 + 2] = P(k, 5);
        odomAftMapped.pose.covariance[i*6 + 3] = P(k, 0);
        odomAftMapped.pose.covariance[i*6 + 4] = P(k, 1);
        odomAftMapped.pose.covariance[i*6 + 5] = P(k, 2);
    }

    static tf::TransformBroadcaster br;
    tf::Transform                   transform;
    tf::Quaternion                  q;
    transform.setOrigin(tf::Vector3(odomAftMapped.pose.pose.position.x, \
                                    odomAftMapped.pose.pose.position.y, \
                                    odomAftMapped.pose.pose.position.z));
    q.setW(odomAftMapped.pose.pose.orientation.w);
    q.setX(odomAftMapped.pose.pose.orientation.x);
    q.setY(odomAftMapped.pose.pose.orientation.y);
    q.setZ(odomAftMapped.pose.pose.orientation.z);
    transform.setRotation( q );
    br.sendTransform( tf::StampedTransform( transform, odomAftMapped.header.stamp, "camera_init", "body" ) );

    // static tf::TransformBroadcaster br_world;
    // transform.setOrigin(tf::Vector3(0, 0, 0));
    // q.setValue(p_imu->Initial_R_wrt_G.x(), p_imu->Initial_R_wrt_G.y(), p_imu->Initial_R_wrt_G.z(), p_imu->Initial_R_wrt_G.w());
    // transform.setRotation(q);
    // br_world.sendTransform(tf::StampedTransform(transform, odomAftMapped.header.stamp, "world", "camera_init"));
}

void publish_path(const ros::Publisher pubPath)
{
    set_posestamp_1(msg_body_pose);
    msg_body_pose.header.stamp = ros::Time().fromSec(lidar_end_time);
    msg_body_pose.header.frame_id = "camera_init";

    /*** if path is too large, the rvis will crash ***/
    static int jjj = 0;
    jjj++;
    if (jjj % 1 == 0)
    {
        path.header.stamp = msg_body_pose.header.stamp;
        path.poses.push_back(msg_body_pose);
        pubPath.publish(path);
    }
}

void transformLidar(const state_ikfom &state_point, const PointCloudXYZI::Ptr &input_cloud, PointCloudXYZI::Ptr &trans_cloud)
{
    trans_cloud->clear();
    trans_cloud->points.resize(input_cloud->size());
#ifdef MP_EN
    #pragma omp parallel for
#endif
    for (size_t i = 0; i < input_cloud->size(); i++) {
        pcl::PointXYZINormal &p_c = input_cloud->points[i];
        Eigen::Vector3d p_lidar(p_c.x, p_c.y, p_c.z);
        // HACK we need to specify p_body as a V3D type!!!
        V3D p_body = state_point.rot * (state_point.offset_R_L_I * p_lidar + state_point.offset_T_L_I) + state_point.pos;
        PointType pi;
        pi.x = p_body(0);
        pi.y = p_body(1);
        pi.z = p_body(2);
        pi.intensity = p_c.intensity;
        trans_cloud->points[i] = std::move(pi);
    }
}

//M3D transformLiDARCovToWorld(Eigen::Vector3d &p_lidar, const esekfom::esekf<state_ikfom, 12, input_ikfom>& kf, const Eigen::Matrix3d& COV_lidar)
//{
//    double match_start = omp_get_wtime();
//    // FIXME 这里首先假定LiDAR系和body是重叠的 没有外参
//    M3D point_crossmat;
//    point_crossmat << SKEW_SYM_MATRX(p_lidar);
//    // 注意这里Rt的cov顺序
//    M3D rot_var = kf.get_P().block<3, 3>(3, 3);
//    M3D t_var = kf.get_P().block<3, 3>(0, 0);
//    auto state = kf.get_x();
//
//    // Eq. (3)
//    M3D COV_world =
//            state.rot * COV_lidar * state.rot.conjugate()
//            + state.rot * (-point_crossmat) * rot_var * (-point_crossmat).transpose()  * state.rot.conjugate()
//            + t_var;
//    return COV_world;
//    // Voxel map 真实实现
////    M3D cov_world = R_body * COV_lidar * R_body.conjugate() +
////          (-point_crossmat) * rot_var * (-point_crossmat).transpose() + t_var;
//
//}

M3D transformLiDARCovToWorld(const Eigen::Vector3d &p_lidar, const esekfom::esekf<state_ikfom, 12, input_ikfom>& kf, const Eigen::Matrix3d& COV_lidar)
{
    M3D point_crossmat;
    point_crossmat << SKEW_SYM_MATRX(p_lidar);
    auto state = kf.get_x();

    // lidar到body的方差传播
    // 注意外参的var是先rot 后pos
    M3D il_rot_var = kf.get_P().block<3, 3>(6, 6);
    M3D il_t_var = kf.get_P().block<3, 3>(9, 9);

    M3D COV_body =
            state.offset_R_L_I * COV_lidar * state.offset_R_L_I.conjugate()
            + state.offset_R_L_I * (-point_crossmat) * il_rot_var * (-point_crossmat).transpose() * state.offset_R_L_I.conjugate()
            + il_t_var;

    // body的坐标
    V3D p_body = state.offset_R_L_I * p_lidar + state.offset_T_L_I;

    // body到world的方差传播
    // 注意pose的var是先pos 后rot
    point_crossmat << SKEW_SYM_MATRX(p_body);
    M3D rot_var = kf.get_P().block<3, 3>(3, 3);
    M3D t_var = kf.get_P().block<3, 3>(0, 0);

    // Eq. (3)
    M3D COV_world =
        state.rot * COV_body * state.rot.conjugate()
        + state.rot * (-point_crossmat) * rot_var * (-point_crossmat).transpose()  * state.rot.conjugate()
        + t_var;

    return COV_world;
    // Voxel map 真实实现
//    M3D cov_world = R_body * COV_lidar * R_body.conjugate() +
//          (-point_crossmat) * rot_var * (-point_crossmat).transpose() + t_var;

}

void observation_model_share(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data)
{
//    laserCloudOri->clear();
//    corr_normvect->clear();
    // feats_with_correspondence->clear();
    total_residual = 0.0;

    // =================================================================================================================
    // 用当前迭代轮最新的位姿估计值 将点云转换到world地图系
    PointCloudXYZI::Ptr world_lidar(new PointCloudXYZI);
    // FIXME stupid mistake 这里应该用迭代的最新线性化点
    // FIXME stupid mistake 这里应该用迭代的最新线性化点
//    transformLidar(state_point, feats_down_body, world_lidar);
    transformLidar(s, feats_down_body, world_lidar);
    vector<pointWithCov> pv_list(feats_down_body->size());
#ifdef MP_EN
    #pragma omp parallel for
#endif
    for (size_t i = 0; i < feats_down_body->size(); i++) {
        // 保存body系和world系坐标
        pointWithCov pv;
        pv.point << feats_down_body->points[i].x, feats_down_body->points[i].y, feats_down_body->points[i].z;
        pv.point_world << world_lidar->points[i].x, world_lidar->points[i].y, world_lidar->points[i].z;
        // 计算lidar点的cov
        // 注意这个在每次迭代时是存在重复计算的 因为lidar系的点云covariance是不变的
        // M3D cov_lidar = calcBodyCov(pv.point, ranging_cov, angle_cov);
        M3D &cov_lidar = var_down_body[i];
        // 将body系的var转换到world系
        M3D cov_world = transformLiDARCovToWorld(pv.point, kf, cov_lidar);
        pv.cov = cov_world;
        pv.cov_lidar = cov_lidar;
        pv_list[i] = pv;
    }

    // ===============================================================================================================
    // 查找最近点 并构建residual
    double match_start = omp_get_wtime();
    std::vector<ptpl> ptpl_list;
    std::vector<V3D> non_match_list;
    BuildResidualListOMP(voxel_map, max_voxel_size, 3.0, max_layer, pv_list,
                         ptpl_list, non_match_list);
    double match_end = omp_get_wtime();
    // std::printf("Match Time: %f\n", match_end - match_start);

    /*** Computation of Measuremnt Jacobian matrix H and measurents vector ***/
    // 根据匹配结果 设置H和R的维度
    // h_x是观测值对状态量的导数 TODO 为什么不加上状态量对状态量误差的导数？？？？像quaternion那本书？
    effct_feat_num = ptpl_list.size();
    if (effct_feat_num < 1){
        ekfom_data.valid = false;
        ROS_WARN("No Effective Points! \n");
        return;
    }
    ekfom_data.h_x = MatrixXd::Zero(effct_feat_num, 12); //23 因为点面距离只和位姿 外参有关 对其他状态量的导数都是0
    ekfom_data.h.resize(effct_feat_num);
    ekfom_data.R.resize(effct_feat_num, 1); // 把R作为向量 用的时候转换成diag

#ifdef MP_EN
    #pragma omp parallel for
#endif
    for (int i = 0; i < effct_feat_num; i++)
    {
//        const PointType &laser_p  = laserCloudOri->points[i];
        V3D point_this_be(ptpl_list[i].point);
        M3D point_be_crossmat;
        point_be_crossmat << SKEW_SYM_MATRX(point_this_be);
        V3D point_this = s.offset_R_L_I * point_this_be + s.offset_T_L_I;
        M3D point_crossmat;
        point_crossmat<<SKEW_SYM_MATRX(point_this);

        /*** get the normal vector of closest surface/corner ***/
//        const PointType &norm_p = corr_normvect->points[i];
//        V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);
        V3D norm_vec(ptpl_list[i].normal);

        /*** calculate the Measuremnt Jacobian matrix H ***/
        V3D C(s.rot.conjugate() *norm_vec);
        V3D A(point_crossmat * C);
        if (extrinsic_est_en)
        {
            V3D B(point_be_crossmat * s.offset_R_L_I.conjugate() * C); //s.rot.conjugate()*norm_vec);
            // ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
            ekfom_data.h_x.block<1, 12>(i,0) << norm_vec.x(), norm_vec.y(), norm_vec.z(), VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
        }
        else
        {
            // ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
            ekfom_data.h_x.block<1, 12>(i,0) << norm_vec.x(), norm_vec.y(), norm_vec.z(), VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
        }

        /*** Measuremnt: distance to the closest surface/corner ***/
//        ekfom_data.h(i) = -norm_p.intensity;
        float pd2 = norm_vec.x() * ptpl_list[i].point_world.x()
                + norm_vec.y() * ptpl_list[i].point_world.y()
                + norm_vec.z() * ptpl_list[i].point_world.z()
                + ptpl_list[i].d;
        ekfom_data.h(i) = -pd2;

        /*** Covariance ***/
//        // norm_p中存了匹配的平面法向 还有点面距离
//        V3D point_world = s.rot * (s.offset_R_L_I * ptpl_list[i].point + s.offset_T_L_I) + s.pos;
//        // /*** get the normal vector of closest surface/corner ***/
//        Eigen::Matrix<double, 1, 6> J_nq;
//        J_nq.block<1, 3>(0, 0) = point_world - ptpl_list[i].center;
//        J_nq.block<1, 3>(0, 3) = -ptpl_list[i].normal;
//        double sigma_l = J_nq * ptpl_list[i].plane_cov * J_nq.transpose();
//
//        M3D cov_lidar = calcBodyCov(ptpl_list[i].point, ranging_cov, angle_cov);
//        M3D R_cov_Rt = s.rot * cov_lidar * s.rot.conjugate();
//        // HACK 1. 因为是标量 所以求逆直接用1除
//        // HACK 2. 不同分量的方差用加法来合成 因为公式(12)中的Sigma是对角阵，逐元素运算之后就是对角线上的项目相加
//        double R_inv = 1.0 / (sigma_l + norm_vec.transpose() * R_cov_Rt * norm_vec);

        // norm_p中存了匹配的平面法向 还有点面距离
        // V3D point_world = s.rot * (s.offset_R_L_I * ptpl_list[i].point + s.offset_T_L_I) + s.pos;
        V3D point_world = ptpl_list[i].point_world;
        // /*** get the normal vector of closest surface/corner ***/
        Eigen::Matrix<double, 1, 6> J_nq;
        J_nq.block<1, 3>(0, 0) = point_world - ptpl_list[i].center;
        J_nq.block<1, 3>(0, 3) = -ptpl_list[i].normal;
        double sigma_l = J_nq * ptpl_list[i].plane_cov * J_nq.transpose();

        // M3D cov_lidar = calcBodyCov(ptpl_list[i].point, ranging_cov, angle_cov);
        M3D cov_lidar = ptpl_list[i].cov_lidar;
        M3D R_cov_Rt = s.rot * s.offset_R_L_I * cov_lidar * s.offset_R_L_I.conjugate() * s.rot.conjugate();
        // HACK 1. 因为是标量 所以求逆直接用1除
        // HACK 2. 不同分量的方差用加法来合成 因为公式(12)中的Sigma是对角阵，逐元素运算之后就是对角线上的项目相加
        R_inv = 1.0 / (sigma_l + norm_vec.transpose() * R_cov_Rt * norm_vec);

        // 计算测量方差R并赋值 目前暂时使用固定值
        // ekfom_data.R(i) = 1.0 / LASER_POINT_COV;
        ekfom_data.R(i) = R_inv;
    }

    // std::printf("Effective Points: %d\n", effct_feat_num);
    res_mean_last = total_residual / effct_feat_num;
    // std::printf("res_mean: %f\n", res_mean_last);
    // std::printf("ef_num: %d\n", effct_feat_num);
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "laserMapping");
    ros::NodeHandle nh;

#ifdef MP_EN
    Eigen::setNbThreads(MP_PROC_NUM);
    Eigen::initParallel();
    omp_set_num_threads(MP_PROC_NUM);
#endif

    nh.param<double>("time_offset", lidar_time_offset, 0.0);

    nh.param<bool>("publish/path_en",path_en, true);
    nh.param<bool>("publish/scan_publish_en",scan_pub_en, true);
    nh.param<bool>("publish/dense_publish_en",dense_pub_en, true);
    nh.param<bool>("publish/scan_bodyframe_pub_en",scan_body_pub_en, true);
    nh.param<string>("common/lid_topic",lid_topic,"/livox/lidar");
    nh.param<string>("common/imu_topic", imu_topic,"/livox/imu");
    nh.param<bool>("common/time_sync_en", time_sync_en, false);

    // mapping algorithm params
    nh.param<float>("mapping/det_range",DET_RANGE,300.f);
    nh.param<int>("mapping/max_iteration", NUM_MAX_ITERATIONS, 4);
    nh.param<int>("mapping/max_points_size", max_points_size, 100);
    nh.param<int>("mapping/max_cov_points_size", max_cov_points_size, 100);
    nh.param<vector<double>>("mapping/layer_point_size", layer_point_size,vector<double>());
    nh.param<int>("mapping/max_layer", max_layer, 2);
    nh.param<double>("mapping/voxel_size", max_voxel_size, 1.0);
    nh.param<double>("mapping/down_sample_size", filter_size_surf_min, 0.5);
    nh.param<double>("mapping/decayed_time", decayed_time, 10.0);
    std::cout << "filter_size_surf_min:" << filter_size_surf_min << std::endl;
    nh.param<double>("mapping/plannar_threshold", min_eigen_value, 0.01);
    nh.param<bool>("mapping/extrinsic_est_en", extrinsic_est_en, true);
    nh.param<vector<double>>("mapping/extrinsic_T", extrinT, vector<double>());
    nh.param<vector<double>>("mapping/extrinsic_R", extrinR, vector<double>());
    nh.param<double>("mapping/acc_norm", acc_norm, 1.0);
    nh.param<vector<double>>("mapping/gravity", gravity, vector<double>());

    // noise model params
    nh.param<double>("noise_model/ranging_cov", ranging_cov, 0.02);
    nh.param<double>("noise_model/angle_cov", angle_cov, 0.05);
    nh.param<double>("noise_model/gyr_cov",gyr_cov,0.1);
    nh.param<double>("noise_model/acc_cov",acc_cov,0.1);
    nh.param<double>("noise_model/b_gyr_cov",b_gyr_cov,0.0001);
    nh.param<double>("noise_model/b_acc_cov",b_acc_cov,0.0001);

    // visualization params
    nh.param<bool>("publish/pub_voxel_map", publish_voxel_map, false);
    nh.param<int>("publish/publish_max_voxel_layer", publish_max_voxel_layer, 0);

    nh.param<bool>("pcd_save/pcd_save_en", pcd_save_en, false);
    nh.param<int>("pcd_save/interval", pcd_save_interval, -1);

    nh.param<float>("preprocess/blind", p_pre->blind, 0.3f);
    nh.param<int>("preprocess/lidar_type", p_pre->lidar_type, AVIA);
    nh.param<int>("preprocess/scan_line", p_pre->N_SCANS, 16);
    nh.param<int>("preprocess/scan_channels", p_pre->M_SCANS, 1024);
    nh.param<int>("preprocess/scan_rate", p_pre->SCAN_RATE, 10);
    nh.param<int>("preprocess/point_filter_num", p_pre->point_filter_num, 1);
    nh.param<float>("preprocess/intensity_threshold", p_pre->intensity_threshold, 0.0);
    cout<<"p_pre->lidar_type "<<p_pre->lidar_type<<endl;
    for (int i = 0; i < layer_point_size.size(); i++) {
        layer_size.push_back(layer_point_size[i]);
    }

    path.header.stamp    = ros::Time::now();
    path.header.frame_id = "camera_init";

    downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);

    // XXX 暂时现在lidar callback中固定转换到IMU系下
    Lidar_T_wrt_IMU<<VEC_FROM_ARRAY(extrinT);
    Lidar_R_wrt_IMU<<MAT_FROM_ARRAY(extrinR);
    p_imu->set_extrinsic(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU);
    p_imu->set_gyr_cov(V3D(gyr_cov, gyr_cov, gyr_cov));
    p_imu->set_acc_cov(V3D(acc_cov, acc_cov, acc_cov));
    p_imu->set_gyr_bias_cov(V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
    p_imu->set_acc_bias_cov(V3D(b_acc_cov, b_acc_cov, b_acc_cov));
    p_imu->gravity_ << VEC_FROM_ARRAY(gravity);
    p_imu->acc_norm = acc_norm;

    double epsi[23] = {0.001};
    fill(epsi, epsi+23, 0.001);
    kf.init_dyn_share(get_f, df_dx, df_dw, observation_model_share, NUM_MAX_ITERATIONS, epsi);

    /*** ROS subscribe initialization ***/
    ros::Subscriber sub_pcl = p_pre->lidar_type == AVIA ? \
        nh.subscribe(lid_topic, 100, livox_pcl_cbk) : \
        nh.subscribe(lid_topic, 100, standard_pcl_cbk);
    ros::Subscriber sub_imu = nh.subscribe(imu_topic, 10000, imu_cbk);
    ros::Publisher pubLaserCloudFull = nh.advertise<sensor_msgs::PointCloud2>
            ("/cloud_registered", 100);
    ros::Publisher pubLaserCloudFull_body = nh.advertise<sensor_msgs::PointCloud2>
            ("/cloud_registered_body", 100);
    // ros::Publisher pubLaserCloudEffect = nh.advertise<sensor_msgs::PointCloud2>
    //         ("/cloud_effected", 100000);
    // ros::Publisher pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>
    //         ("/Laser_map", 100000);
    ros::Publisher pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>
            ("/Odometry", 100000);
    // ros::Publisher pubExtrinsic = nh.advertise<nav_msgs::Odometry>
    //         ("/Extrinsic", 100000);
    ros::Publisher pubPath          = nh.advertise<nav_msgs::Path>
            ("/path", 100000);
    ros::Publisher voxel_map_pub =
            nh.advertise<visualization_msgs::MarkerArray>("/planes", 10000);
    ros::ServiceServer srvSaveMap  = nh.advertiseService("/save_map" ,  &saveMapService);

//------------------------------------------------------------------------------------------------------
    // for Plane Map
    bool init_map = false;

    double sum_optimize_time = 0, sum_update_time = 0;
    int scan_index = 0;

    // signal(SIGINT, SigHandle);
    // pcl_wait_save = PointCloudXYZI::Ptr(new PointCloudXYZI);
    pcd_file_prefix = root_dir + "PCD/scans-" + std::to_string(uint64_t(ros::Time::now().toSec()));
    pcd_index = 0;

    auto worker_thread = std::thread([&]() {
        ros::Rate rate(5000);
        while (ros::ok())
        {
            if (flg_exit) break;
            // ros::spinOnce();
            if(sync_packages(Measures))
            {
                if (flg_first_scan)
                {
                    first_lidar_time = Measures.lidar_beg_time;
                    p_imu->first_lidar_time = first_lidar_time;
                    flg_first_scan = false;
                    continue;
                }

                p_imu->Process(Measures, kf , feats_undistort);
                state_point = kf.get_x();
                // pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;

                if (feats_undistort == nullptr || feats_undistort->empty())
                {
                    ROS_WARN("No point, skip this scan!\n");
                    continue;
                }

                flg_EKF_inited = (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? \
                                false : true;
                // ===============================================================================================================
                // 第一帧 如果ekf初始化了 就初始化voxel地图
                if (flg_EKF_inited && !init_map) {
                    PointCloudXYZI::Ptr world_lidar(new PointCloudXYZI);
                    transformLidar(state_point, feats_undistort, world_lidar);
                    std::vector<pointWithCov> pv_list(world_lidar->size());

                    // std::cout << kf.get_P() << std::endl;
                    // 计算第一帧所有点的covariance 并用于构建初始地图
    #ifdef MP_EN
                    #pragma omp parallel for
    #endif
                    for (size_t i = 0; i < world_lidar->size(); i++) {
                        pointWithCov pv;
                        pv.point << world_lidar->points[i].x, world_lidar->points[i].y,
                                world_lidar->points[i].z;
                        V3D point_this(feats_undistort->points[i].x,
                                    feats_undistort->points[i].y,
                                    feats_undistort->points[i].z);
                        // if z=0, error will occur in calcBodyCov. To be solved
                        if (point_this[2] == 0) {
                            point_this[2] = 0.001;
                        }
                        M3D cov_lidar = calcBodyCov(point_this, ranging_cov, angle_cov);
                        // 转换到world系
                        M3D cov_world = transformLiDARCovToWorld(point_this, kf, cov_lidar);

                        pv.cov = cov_world;
                        pv_list[i] = std::move(pv);
                        // Eigen::Vector3d sigma_pv = pv.cov.diagonal();
                        // sigma_pv[0] = sqrt(sigma_pv[0]);
                        // sigma_pv[1] = sqrt(sigma_pv[1]);
                        // sigma_pv[2] = sqrt(sigma_pv[2]);
                    }

                    buildVoxelMap(pv_list, max_voxel_size, max_layer, layer_size,
                                max_points_size, max_points_size, min_eigen_value,
                                voxel_map);
                    std::cout << "build voxel map" << std::endl;

                    if (publish_voxel_map) {
                        pubVoxelMap(voxel_map, publish_max_voxel_layer, voxel_map_pub);
                        publish_frame_world(pubLaserCloudFull);
                        publish_frame_body(pubLaserCloudFull_body);
                    }
                    init_map = true;
                    continue;
                }
                
                double t_update_start = omp_get_wtime();
                /*** downsample the feature points in a scan ***/
                downSizeFilterSurf.setInputCloud(feats_undistort);
                downSizeFilterSurf.filter(*feats_down_body);
                std::sort(feats_down_body->points.begin(), feats_down_body->points.end(), 
                    [](PointType &x, PointType &y) {return (x.curvature < y.curvature);});

                feats_down_size = feats_down_body->points.size();
                // 由于点云的body var是一直不变的 因此提前计算 在迭代时可以复用
                var_down_body.clear();
                var_down_body.resize(feats_down_size);
    #ifdef MP_EN
                #pragma omp parallel for
    #endif
                for (auto i=0; i<feats_down_size; ++i ) {
                    V3D point_this(feats_down_body->points[i].x, feats_down_body->points[i].y, feats_down_body->points[i].z);
                    var_down_body[i] = calcBodyCov(point_this, ranging_cov, angle_cov);
                }
                double t_update_end = omp_get_wtime();
                // std::printf("pointcloud processing: %.5fs \n", t_update_end - t_update_start);


                /*** ICP and iterated Kalman filter update ***/
                if (feats_down_size < 5)
                {
                    ROS_WARN("No point, skip this scan!\n");
                    continue;
                }
                // ===============================================================================================================
                // 开始迭代滤波
                /*** iterated state estimation ***/
                t_update_start = omp_get_wtime();
                // double solve_H_time = 0;
                kf.update_iterated_dyn_share_diagonal();
    //            kf.update_iterated_dyn_share_modified(LASER_POINT_COV, solve_H_time);
                t_update_end = omp_get_wtime();
                // std::printf("update_iterated_dyn_share_diagonal: %.5fs \n", t_update_end - t_update_start);
                sum_optimize_time += t_update_end - t_update_start;

                state_point = kf.get_x();
                // euler_cur = SO3ToEuler(state_point.rot);
                // pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
                geoQuat.x = state_point.rot.coeffs()[0];
                geoQuat.y = state_point.rot.coeffs()[1];
                geoQuat.z = state_point.rot.coeffs()[2];
                geoQuat.w = state_point.rot.coeffs()[3];
    //
                // std::printf("BA: %.4f %.4f %.4f   BG: %.4f %.4f %.4f   g: %.4f %.4f %.4f\n",
                //             kf.get_x().ba.x(),kf.get_x().ba.y(),kf.get_x().ba.z(),
                //             kf.get_x().bg.x(),kf.get_x().bg.y(),kf.get_x().bg.z(),
                //             kf.get_x().grav.get_vect().x(), kf.get_x().grav.get_vect().y(), kf.get_x().grav.get_vect().z()
                // );

                // ===============================================================================================================
                // 更新地图
                /*** add the points to the voxel map ***/
                // 用最新的状态估计将点及点的covariance转换到world系
                std::vector<pointWithCov> pv_list(feats_down_body->size());
                PointCloudXYZI::Ptr world_lidar(new PointCloudXYZI);
                transformLidar(state_point, feats_down_body, world_lidar);
    #ifdef MP_EN
                #pragma omp parallel for
    #endif
                for (size_t i = 0; i < feats_down_body->size(); i++) {
                    // 保存body系和world系坐标
                    pointWithCov pv;
                    pv.point << feats_down_body->points[i].x, feats_down_body->points[i].y, feats_down_body->points[i].z;
                    // 计算lidar点的cov
                    // FIXME 这里错误的使用世界系的点来calcBodyCov时 反倒在某些seq（比如hilti2022的03 15）上效果更好 需要考虑是不是init_plane时使用更大的cov更好
                    // 注意这个在每次迭代时是存在重复计算的 因为lidar系的点云covariance是不变的
                    // M3D cov_lidar = calcBodyCov(pv.point, ranging_cov, angle_cov);
                    M3D &cov_lidar = var_down_body[i];
                    // 将body系的var转换到world系
                    M3D cov_world = transformLiDARCovToWorld(pv.point, kf, cov_lidar);

                    // 最终updateVoxelMap需要用的是world系的point
                    pv.cov = cov_world;
                    pv.cov_diagonal_norm = pv.cov.diagonal().norm();
                    pv.point << world_lidar->points[i].x, world_lidar->points[i].y, world_lidar->points[i].z;
                    pv_list[i] = std::move(pv);
                }

                t_update_start = omp_get_wtime();
                std::sort(pv_list.begin(), pv_list.end(), var_contrast_const);
                updateVoxelMapOMP(pv_list, max_voxel_size, max_layer, layer_size,
                            max_points_size, max_points_size, min_eigen_value,
                            voxel_map);
                t_update_end = omp_get_wtime();
                // std::printf("updateVoxelMapOMP: %.5fs \n", t_update_end - t_update_start);
                sum_update_time += t_update_end - t_update_start;

                scan_index++;
                // std::printf("Mean  Topt: %.5fs   Tu: %.5fs\n", sum_optimize_time / scan_index, sum_update_time / scan_index);
                // ===============================================================================================================
                // 可视化相关的shit
                /******* Publish odometry *******/
                publish_odometry(pubOdomAftMapped);
    //
    //            /*** add the feature points to map kdtree ***/
    //            map_incremental();
    //
                /******* Publish points *******/
                if (path_en)                         publish_path(pubPath);
                if (scan_pub_en)      publish_frame_world(pubLaserCloudFull);
                if (scan_pub_en && scan_body_pub_en) publish_frame_body(pubLaserCloudFull_body);
                if (publish_voxel_map) {
                    pubVoxelMap(voxel_map, publish_max_voxel_layer, voxel_map_pub);
                }
                // publish_effect_world(pubLaserCloudEffect);
                // publish_map(pubLaserCloudMap);
                printf("voxel_map size = %lu\n", voxel_map.size());
                auto now_sec = ros::Time::now().toSec();
                auto it = voxel_map.begin();
                while(it!=voxel_map.end())
                {
                    auto x = it->second->voxel_center_(0) - state_point.pos(0);
                    auto y = it->second->voxel_center_(1) - state_point.pos(1);
                    auto z = it->second->voxel_center_(2) - state_point.pos(2);
                    if (now_sec - it->second->tm > decayed_time && x*x+y*y+z*z > DET_RANGE*DET_RANGE)
                    {
                        for (uint i = 0; i < 8; i++)
                            it->second->leaves_[i].reset();
                        it->second.reset();
                        it = voxel_map.erase(it);
                    }
                    else
                        ++it;
                }
                printf("purged voxel_map size = %lu\n", voxel_map.size());
            }
            else
                rate.sleep();
        }
        //--------------------------save map-----------------------------------
        /* 1. make sure you have enough memories
        /* 2. noted that pcd save will influence the real-time performences **/
        if (pcl_wait_save->size() > 0 && pcd_save_en)
        {
            std::string all_points_dir(pcd_file_prefix + "-" + std::to_string(pcd_index++) + ".pcd");
            pcl::PCDWriter pcd_writer;
            pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
        }
    });

    ros::spin();

    flg_exit = true;
    if (worker_thread.joinable())
        worker_thread.join();

    return 0;
}
