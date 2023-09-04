#include "preprocess.h"

#define RETURN0     0x00
#define RETURN0AND1 0x10

Preprocess::Preprocess()
  : lidar_type(AVIA), point_filter_num(1)
{
  N_SCANS   = 6;
  SCAN_RATE = 10;

  given_offset_time = false;
}

Preprocess::~Preprocess() {}

void Preprocess::process(const livox_ros_driver::CustomMsg::ConstPtr &msg, PointCloudXYZI::Ptr &pcl_out)
{  
  avia_handler(msg, pcl_out);
}

void Preprocess::process(const sensor_msgs::PointCloud2::ConstPtr &msg, PointCloudXYZI::Ptr &pcl_out)
{
  switch (lidar_type)
  {
  case OUST64:
    oust64_handler(msg, pcl_out);
    break;

  case VELO16:
    velodyne_handler(msg, pcl_out);
    break;
  
  default:
    printf("Error LiDAR Type");
    break;
  }
}

void Preprocess::avia_handler(const livox_ros_driver::CustomMsg::ConstPtr &msg, PointCloudXYZI::Ptr &pcl_out)
{
  pcl_out->clear();
  // pl_surf_filtered.clear();
  // pl_corn.clear();
  pl_full.clear();
  double t1 = omp_get_wtime();
  int plsize = msg->point_num;
  // cout<<"plsie: "<<plsize<<endl;

  // pl_corn.reserve(plsize);
  pcl_out->reserve(plsize);
  // pl_surf_filtered.reserve(plsize);
  pl_full.resize(plsize);

  // for(int i=0; i<N_SCANS; i++)
  // {
  //   pl_buff[i].clear();
  //   pl_buff[i].reserve(plsize);
  // }
  uint valid_num = 0;
  
  {
    for(uint i=1; i<plsize; i++)
    {
      if((msg->points[i].line < N_SCANS) && ((msg->points[i].tag & 0x30) == 0x10 || (msg->points[i].tag & 0x30) == 0x00))
      {
        valid_num ++;
        if (valid_num % point_filter_num == 0)
        {
//          if(i==1){
//              std::printf("Scan timestamp: %f, First point time: %f\n", msg->header.stamp.toSec(), msg->points[i].offset_time / float(1000000));
//          }
          pl_full[i].x = msg->points[i].x;
          pl_full[i].y = msg->points[i].y;
          pl_full[i].z = msg->points[i].z;
          pl_full[i].intensity = msg->points[i].reflectivity;
          pl_full[i].curvature = msg->points[i].offset_time / float(1000000); // use curvature as time of each laser points, curvature unit: ms

          //与上一个添加进去的点比较
          if (((abs(pl_full[i].x) > 1e-7) || (abs(pl_full[i].y) > 1e-7) || (abs(pl_full[i].z) > 1e-7))
              && (pl_full[i].x * pl_full[i].x + pl_full[i].y * pl_full[i].y + pl_full[i].z * pl_full[i].z > (blind * blind)))
          {
            pcl_out->push_back(pl_full[i]);
          }
        }
      }
    }
  }
}

void Preprocess::oust64_handler(const sensor_msgs::PointCloud2::ConstPtr &msg, PointCloudXYZI::Ptr &pcl_out)
{
  pcl_out->clear();
  pcl::PointCloud<ouster_ros::Point> pl_orig;
  pcl::fromROSMsg(*msg, pl_orig);
  int plsize = pl_orig.size();
  if (plsize != M_SCANS*N_SCANS)
  {
    printf("Error: numbers of channel and ring for Ouster lidar mismatch! channels = %d, rings = %d\n", M_SCANS, N_SCANS);
    return;
  }
  pcl_out->reserve(plsize);
  for (int i = 0; i < M_SCANS; i++)
  {
    if (i % point_filter_num != 0) continue;
    for (int j = 0; j < N_SCANS; j++)
    {
      auto &point = pl_orig.points[j * M_SCANS + i];
      if (point.intensity < intensity_threshold) continue;

      float range = (point.range & 0xfffffu) * 0.001f;
      if (range < blind) continue;

      PointType added_pt;
      added_pt.x = point.x;
      added_pt.y = point.y;
      added_pt.z = point.z;
      added_pt.intensity = point.intensity;
      added_pt.normal_x = 0;  //RGB as uint32_t
      added_pt.normal_y = sqrtf((point.reflectivity & 0xff)/255.0f); //reflectivity 0-1.0
      added_pt.normal_z = sqrtf(1.0-added_pt.normal_y*added_pt.normal_y);
      // added_pt.normal_z = pl_orig.points[i].ambient;    // NIR
      added_pt.curvature = point.t / 1e6; // curvature unit: ms

      pcl_out->points.push_back(std::move(added_pt));
    }
  }
}

void Preprocess::velodyne_handler(const sensor_msgs::PointCloud2::ConstPtr &msg, PointCloudXYZI::Ptr &pcl_out)
{
    // pl_surf.clear();
    // pl_corn.clear();
    pcl_out->clear();
    pl_full.clear();

    pcl::PointCloud<velodyne_ros::Point> pl_orig;
    pcl::fromROSMsg(*msg, pl_orig);
    int plsize = pl_orig.points.size();
    // pl_surf.reserve(plsize);
    pcl_out->reserve(plsize);

    /*** These variables only works when no point timestamps given ***/
    double omega_l = 0.361 * SCAN_RATE;       // scan angular velocity
    std::vector<bool> is_first(N_SCANS,true);
    std::vector<double> yaw_fp(N_SCANS, 0.0);      // yaw of first scan point
    std::vector<float> yaw_last(N_SCANS, 0.0);   // yaw of last scan point
    std::vector<float> time_last(N_SCANS, 0.0);  // last offset time
    /*****************************************************************/

    if (pl_orig.points[plsize - 1].time > 0)
    {
      given_offset_time = true;
    }
    else
    {
      given_offset_time = false;
      double yaw_first = atan2(pl_orig.points[0].y, pl_orig.points[0].x) * 57.29578;
      double yaw_end  = yaw_first;
      int layer_first = pl_orig.points[0].ring;
      for (uint i = plsize - 1; i > 0; i--)
      {
        if (pl_orig.points[i].ring == layer_first)
        {
          yaw_end = atan2(pl_orig.points[i].y, pl_orig.points[i].x) * 57.29578;
          break;
        }
      }
    }

  
    {
      for (int i = 0; i < plsize; i++)
      {
        // 删除第一排点 因为可能时间戳有问题
        if (std::abs(pl_orig.points[i].time) < 1.0 / SCAN_RATE / 1800){
//            std::printf("%d\n", i);
            continue;
        }
        // 还有可能当前帧的点云中出现了不属于这一帧的点 即时间戳超出当前帧采样范围太远 这样的点也要丢弃
        // 否则可能会影响pcl_end_time的计算 这个会给imu propagation的过程的时间戳产生误导
        // 这个问题可能是雷达传输过程中的丢失重传造成的 也可能是雷达内部的bug
        if (std::abs(pl_orig.points[i].time) > (1.0 / SCAN_RATE) * 1.1){
//            std::printf("PT timestamp out of range: %d    frame ts: %f  point ts: %f\n",
//                       i, msg->header.stamp.toSec(), pl_orig.points[i].time);
            continue;
        }

//        if(pl_orig.points[i].intensity < 5){
//            continue;
//        }

        PointType added_pt;
        // cout<<"!!!!!!"<<i<<" "<<plsize<<endl;

        added_pt.normal_x = pl_orig.points[i].time;
        added_pt.normal_y = 0;
        added_pt.normal_z = 0;
        added_pt.x = pl_orig.points[i].x;
        added_pt.y = pl_orig.points[i].y;
        added_pt.z = pl_orig.points[i].z;
        added_pt.intensity = pl_orig.points[i].intensity;
        added_pt.curvature = pl_orig.points[i].time * 1000.0;  // curvature unit: ms

        if (!given_offset_time)
        {
          int layer = pl_orig.points[i].ring;
          double yaw_angle = atan2(added_pt.y, added_pt.x) * 57.2957;

          if (is_first[layer])
          {
            // printf("layer: %d; is first: %d", layer, is_first[layer]);
              yaw_fp[layer]=yaw_angle;
              is_first[layer]=false;
              added_pt.curvature = 0.0;
              yaw_last[layer]=yaw_angle;
              time_last[layer]=added_pt.curvature;
              continue;
          }

          // compute offset time
          if (yaw_angle <= yaw_fp[layer])
          {
            added_pt.curvature = (yaw_fp[layer]-yaw_angle) / omega_l;
          }
          else
          {
            added_pt.curvature = (yaw_fp[layer]-yaw_angle+360.0) / omega_l;
          }

          if (added_pt.curvature < time_last[layer])  added_pt.curvature+=360.0/omega_l;

          yaw_last[layer] = yaw_angle;
          time_last[layer]=added_pt.curvature;
        }

        if (i % point_filter_num == 0)
        {
          if(added_pt.x*added_pt.x+added_pt.y*added_pt.y+added_pt.z*added_pt.z > (blind * blind))
          {
            pcl_out->points.push_back(std::move(added_pt));
          }
        }
      }
    }
}

