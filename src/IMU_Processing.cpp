#include "IMU_Processing.h"

using namespace std;

static Eigen::Vector3d R2ypr(const Eigen::Matrix3d &R)
{
    Eigen::Vector3d n = R.col(0);
    Eigen::Vector3d o = R.col(1);
    Eigen::Vector3d a = R.col(2);

    Eigen::Vector3d ypr(3);
    double y = atan2(n(1), n(0));
    double p = atan2(-n(2), n(0) * cos(y) + n(1) * sin(y));
    double r = atan2(a(0) * sin(y) - a(1) * cos(y), -o(0) * sin(y) + o(1) * cos(y));
    ypr(0) = y;
    ypr(1) = p;
    ypr(2) = r;

    return ypr / M_PI * 180.0;
}

template <typename Derived>
static Eigen::Matrix<typename Derived::Scalar, 3, 3> ypr2R(const Eigen::MatrixBase<Derived> &ypr)
{
    typedef typename Derived::Scalar Scalar_t;

    Scalar_t y = ypr(0) / 180.0 * M_PI;
    Scalar_t p = ypr(1) / 180.0 * M_PI;
    Scalar_t r = ypr(2) / 180.0 * M_PI;

    Eigen::Matrix<Scalar_t, 3, 3> Rz;
    Rz << cos(y), -sin(y), 0,
            sin(y), cos(y), 0,
            0, 0, 1;

    Eigen::Matrix<Scalar_t, 3, 3> Ry;
    Ry << cos(p), 0., sin(p),
            0., 1., 0.,
            -sin(p), 0., cos(p);

    Eigen::Matrix<Scalar_t, 3, 3> Rx;
    Rx << 1., 0., 0.,
            0., cos(r), -sin(r),
            0., sin(r), cos(r);

    return Rz * Ry * Rx;
}

static Eigen::Matrix3d g2R(const Eigen::Vector3d &g)
{
    Eigen::Matrix3d R0;
    Eigen::Vector3d ng1 = g.normalized();
    Eigen::Vector3d ng2{0, 0, 1.0};
    R0 = Eigen::Quaterniond::FromTwoVectors(ng1, ng2).toRotationMatrix();
    double yaw = R2ypr(R0).x();
    R0 = ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    // R0 = Utility::ypr2R(Eigen::Vector3d{-90, 0, 0}) * R0;
    return R0;
}

static const M3D Eye3d(M3D::Identity());
static const M3F Eye3f(M3F::Identity());
static const V3D Zero3d(0, 0, 0);
static const V3F Zero3f(0, 0, 0);

// template<typename T>
// static auto set_pose6d(const double t, const Eigen::Matrix<T, 3, 1> &a, const Eigen::Matrix<T, 3, 1> &g, \
//                 const Eigen::Matrix<T, 3, 1> &v, const Eigen::Matrix<T, 3, 1> &p, const Eigen::Matrix<T, 3, 3> &R)
// {
//     Pose6D rot_kp;
//     rot_kp.offset_time = t;
//     for (int i = 0; i < 3; i++)
//     {
//         rot_kp.acc[i] = a(i);
//         rot_kp.gyr[i] = g(i);
//         rot_kp.vel[i] = v(i);
//         rot_kp.pos[i] = p(i);
//         for (int j = 0; j < 3; j++)  rot_kp.rot[i*3+j] = R(i,j);
//     }
//     return rot_kp;
// }

ImuProcess::ImuProcess()
    : b_first_frame_(true), imu_need_init_(true), start_timestamp_(-1)
{
    init_iter_num = 1;
    Q = process_noise_cov();
    cov_acc = V3D(0.1, 0.1, 0.1);
    cov_gyr = V3D(0.1, 0.1, 0.1);
    cov_bias_gyr = V3D(0.0001, 0.0001, 0.0001);
    cov_bias_acc = V3D(0.0001, 0.0001, 0.0001);
    mean_acc = V3D(0, 0, -1.0);
    mean_gyr = V3D(0, 0, 0);
    angvel_last = Zero3d;
    Lidar_T_wrt_IMU = Zero3d;
    Lidar_R_wrt_IMU = Eye3d;
    last_imu_.reset(new sensor_msgs::Imu());
}

ImuProcess::~ImuProcess() {}

void ImuProcess::Reset()
{
    // ROS_WARN("Reset ImuProcess");
    mean_acc = V3D(0, 0, -1.0);
    mean_gyr = V3D(0, 0, 0);
    angvel_last = Zero3d;
    imu_need_init_ = true;
    start_timestamp_ = -1;
    init_iter_num = 1;
    // v_imu_.clear();
    IMUpose.clear();
    last_imu_.reset(new sensor_msgs::Imu());
    cur_pcl_un_.reset(new PointCloudXYZI());
}

void ImuProcess::set_extrinsic(const MD(4, 4) & T)
{
    Lidar_T_wrt_IMU = T.block<3, 1>(0, 3);
    Lidar_R_wrt_IMU = T.block<3, 3>(0, 0);
}

void ImuProcess::set_extrinsic(const V3D &transl)
{
    Lidar_T_wrt_IMU = transl;
    Lidar_R_wrt_IMU.setIdentity();
}

void ImuProcess::set_extrinsic(const V3D &transl, const M3D &rot)
{
    Lidar_T_wrt_IMU = transl;
    Lidar_R_wrt_IMU = rot;
}

void ImuProcess::set_gyr_cov(const V3D &scaler)
{
    cov_gyr_scale = scaler;
}

void ImuProcess::set_acc_cov(const V3D &scaler)
{
    cov_acc_scale = scaler;
}

void ImuProcess::set_gyr_bias_cov(const V3D &b_g)
{
    cov_bias_gyr = b_g;
}

void ImuProcess::set_acc_bias_cov(const V3D &b_a)
{
    cov_bias_acc = b_a;
}

void ImuProcess::IMU_init(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, int &N)
{
    /** 1. initializing the gravity, gyro bias, acc and gyro covariance
     ** 2. normalize the acceleration measurenments to unit gravity **/

    V3D cur_acc, cur_gyr;

    if (b_first_frame_)
    {
        Reset();
        N = 1;
        b_first_frame_ = false;
        const auto &imu_acc = meas.imu.front()->linear_acceleration;
        const auto &gyr_acc = meas.imu.front()->angular_velocity;
        mean_acc << imu_acc.x, imu_acc.y, imu_acc.z;
        mean_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;
        first_lidar_time = meas.lidar_beg_time;
    }

    for (const auto &imu : meas.imu)
    {
        const auto &imu_acc = imu->linear_acceleration;
        const auto &gyr_acc = imu->angular_velocity;
        cur_acc << imu_acc.x, imu_acc.y, imu_acc.z;
        cur_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;

        mean_acc += (cur_acc - mean_acc) / N;
        mean_gyr += (cur_gyr - mean_gyr) / N;

        cov_acc = cov_acc * (N - 1.0) / N + (cur_acc - mean_acc).cwiseProduct(cur_acc - mean_acc) * (N - 1.0) / (N * N);
        cov_gyr = cov_gyr * (N - 1.0) / N + (cur_gyr - mean_gyr).cwiseProduct(cur_gyr - mean_gyr) * (N - 1.0) / (N * N);

        // cout<<"acc norm: "<<cur_acc.norm()<<" "<<mean_acc.norm()<<endl;

        N++;
    }
    state_ikfom init_state = kf_state.get_x();

//       const auto& orientation = meas.imu.front()->orientation;
//   if( orientation.x != 0 || orientation.y != 0 || orientation.z != 0 || orientation.w != 0 ){
//     init_state.rot = Eigen::Quaterniond(orientation.w, orientation.x, orientation.y, orientation.z);
//   }
  // else
  // {
    // init_state.rot = Eigen::Quaterniond::FromTwoVectors(mean_acc, Eigen::Vector3d::UnitZ());
  // }

    // gravity_ << 0.0, 0.0, -9.810;
    M3D rot_init;
    init_state.grav =  -1 * mean_acc * G_m_s2 / acc_norm; 
    Set_init(init_state.grav , rot_init);
    init_state.grav = gravity_;
    init_state.rot  = rot_init;
    init_state.rot.normalize();
    // mean_acc =  -rot_init.transpose() * gravity_;

    // mean_acc = init_state.rot * mean_acc;
    // mean_gyr = init_state.rot * mean_gyr;
    // init_state.grav = S2(-mean_acc / mean_acc.norm() * G_m_s2);

    // state_inout.rot = Eye3d; // Exp(mean_acc.cross(V3D(0, 0, -1 / scale_gravity)));
    // init_state.bg = mean_gyr;
    init_state.offset_T_L_I = Lidar_T_wrt_IMU;
    init_state.offset_R_L_I = Lidar_R_wrt_IMU;
    kf_state.change_x(init_state);

    // initial world pose
    Initial_R_wrt_G = SO3(g2R(mean_acc));

    esekfom::esekf<state_ikfom, 12, input_ikfom>::cov init_P = kf_state.get_P();
    init_P.setIdentity();
    // init_P.setZero();
    init_P(6, 6) = init_P(7, 7) = init_P(8, 8) = 0.00001;
    init_P(9, 9) = init_P(10, 10) = init_P(11, 11) = 0.00001;
    init_P(15, 15) = init_P(16, 16) = init_P(17, 17) = 0.0001;
    init_P(18, 18) = init_P(19, 19) = init_P(20, 20) = 0.001;
    init_P(21, 21) = init_P(22, 22) = 0.00001;
    kf_state.change_P(init_P);
    last_imu_ = meas.imu.back();
}

void ImuProcess::Set_init(Eigen::Vector3d &tmp_gravity, Eigen::Matrix3d &rot)
{
  /** 1. initializing the gravity, gyro bias, acc and gyro covariance
   ** 2. normalize the acceleration measurenments to unit gravity **/
  // V3D tmp_gravity = - mean_acc / mean_acc.norm() * G_m_s2; // state_gravity;
  M3D hat_grav;
  hat_grav << 0.0, gravity_(2), -gravity_(1),
              -gravity_(2), 0.0, gravity_(0),
              gravity_(1), -gravity_(0), 0.0;
  double align_norm = (hat_grav * tmp_gravity).norm() / tmp_gravity.norm() / gravity_.norm();
  double align_cos = gravity_.transpose() * tmp_gravity;
  align_cos = align_cos / gravity_.norm() / tmp_gravity.norm();
  if (align_norm < 1e-6)
  {
    if (align_cos > 1e-6)
    {
      rot = Eye3d;
    }
    else
    {
      rot = -Eye3d;
    }
  }
  else
  {
    V3D align_angle = hat_grav * tmp_gravity / (hat_grav * tmp_gravity).norm() * acos(align_cos); 
    rot = Exp(align_angle(0), align_angle(1), align_angle(2));
  }
}
// static bool time_list(PointType &x, PointType &y) {return (x.curvature < y.curvature);};

void ImuProcess::UndistortPcl(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI &pcl_out)
{
    /*** add the imu of the last frame-tail to the of current frame-head ***/
    auto v_imu = meas.imu;
    v_imu.push_front(last_imu_);
    const double &imu_beg_time = v_imu.front()->header.stamp.toSec();
    const double &imu_end_time = v_imu.back()->header.stamp.toSec();
    const double &pcl_beg_time = meas.lidar_beg_time;
    const double &pcl_end_time = meas.lidar_end_time;

    /*** sort point clouds by offset time ***/
    pcl_out = std::move(*(meas.lidar));
    // No need for OUSTER !!!!
    // sort(pcl_out.points.begin(), pcl_out.points.end(), time_list); 
    // cout<<"[ IMU Process ]: Process lidar from "<<pcl_beg_time<<" to "<<pcl_end_time<<", " \
  //          <<meas.imu.size()<<" imu msgs from "<<imu_beg_time<<" to "<<imu_end_time<<endl;

    /*** Initialize IMU pose ***/
    state_ikfom imu_state = kf_state.get_x();
    IMUpose.clear();
    IMUpose.push_back(set_pose6d(0.0, acc_s_last, angvel_last, imu_state.vel, imu_state.pos, imu_state.rot.toRotationMatrix()));

    /*** forward propagation at each imu point ***/
    V3D angvel_avr, acc_avr, acc_imu, vel_imu, pos_imu;
    M3D R_imu;

    double dt = 0;

    input_ikfom in;
    for (auto it_imu = v_imu.begin(); it_imu < (v_imu.end() - 1); it_imu++)
    {
        auto &&head = *(it_imu);
        auto &&tail = *(it_imu + 1);

        if (tail->header.stamp.toSec() < last_lidar_end_time_)
            continue;

        angvel_avr << 0.5 * (head->angular_velocity.x + tail->angular_velocity.x),
            0.5 * (head->angular_velocity.y + tail->angular_velocity.y),
            0.5 * (head->angular_velocity.z + tail->angular_velocity.z);
        acc_avr << 0.5 * (head->linear_acceleration.x + tail->linear_acceleration.x),
            0.5 * (head->linear_acceleration.y + tail->linear_acceleration.y),
            0.5 * (head->linear_acceleration.z + tail->linear_acceleration.z);

        // fout_imu << setw(10) << head->header.stamp.toSec() - first_lidar_time << " " << angvel_avr.transpose() << " " << acc_avr.transpose() << endl;

        acc_avr = acc_avr * G_m_s2 / acc_norm; // - state_inout.ba;

        if (head->header.stamp.toSec() < last_lidar_end_time_)
        {
            dt = tail->header.stamp.toSec() - last_lidar_end_time_;
            // dt = tail->header.stamp.toSec() - pcl_beg_time;
        }
        else
        {
            dt = tail->header.stamp.toSec() - head->header.stamp.toSec();
        }

        in.acc = acc_avr;
        in.gyro = angvel_avr;
        Q.block<3, 3>(0, 0).diagonal() = cov_gyr;
        Q.block<3, 3>(3, 3).diagonal() = cov_acc;
        Q.block<3, 3>(6, 6).diagonal() = cov_bias_gyr;
        Q.block<3, 3>(9, 9).diagonal() = cov_bias_acc;
        // 这里应该判断是否有GPS measurements， 如果有GPS， 那么先传播到GPS，然后更新state，再重新从头开始向前传播
        // 看一下r2live的实现
        kf_state.predict(dt, Q, in);

        /* save the poses at each IMU measurements */
        imu_state = kf_state.get_x();
        angvel_last = angvel_avr - imu_state.bg;
        acc_s_last = imu_state.rot * (acc_avr - imu_state.ba);
        for (int i = 0; i < 3; i++)
        {
            acc_s_last[i] += imu_state.grav[i];
        }
        double &&offs_t = tail->header.stamp.toSec() - pcl_beg_time;
        IMUpose.push_back(set_pose6d(offs_t, acc_s_last, angvel_last, imu_state.vel, imu_state.pos, imu_state.rot.toRotationMatrix()));
    }

    /*** calculated the pos and attitude prediction at the frame-end ***/
    double note = pcl_end_time > imu_end_time ? 1.0 : -1.0;
    dt = note * (pcl_end_time - imu_end_time);
    kf_state.predict(dt, Q, in);

    imu_state = kf_state.get_x();
    last_imu_ = meas.imu.back();
    last_lidar_end_time_ = pcl_end_time;

    /*** undistort each lidar point (backward propagation) ***/
    auto it_pcl = pcl_out.points.end() - 1;
    for (auto it_kp = IMUpose.end() - 1; it_kp != IMUpose.begin(); it_kp--)
    {
        auto head = it_kp - 1;
        auto tail = it_kp;
        R_imu << MAT_FROM_ARRAY(head->rot);
        // cout<<"head imu acc: "<<acc_imu.transpose()<<endl;
        vel_imu << VEC_FROM_ARRAY(head->vel);
        pos_imu << VEC_FROM_ARRAY(head->pos);
        acc_imu << VEC_FROM_ARRAY(tail->acc);
        angvel_avr << VEC_FROM_ARRAY(tail->gyr);

        for (; it_pcl->curvature / double(1000) > head->offset_time; it_pcl--)
        {
            dt = it_pcl->curvature / double(1000) - head->offset_time;

            /* Transform to the 'end' frame, using only the rotation
             * Note: Compensation direction is INVERSE of Frame's moving direction
             * So if we want to compensate a point at timestamp-i to the frame-e
             * P_compensate = R_imu_e ^ T * (R_i * P_i + T_ei) where T_ei is represented in global frame */
            M3D R_i(R_imu * Exp(angvel_avr, dt));

            V3D P_i(it_pcl->x, it_pcl->y, it_pcl->z);
            V3D T_ei(pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt - imu_state.pos);
            V3D P_compensate = imu_state.offset_R_L_I.conjugate() * (imu_state.rot.conjugate() * (R_i * (imu_state.offset_R_L_I * P_i + imu_state.offset_T_L_I) + T_ei) - imu_state.offset_T_L_I); // not accurate!

            // save Undistorted points and their rotation
            it_pcl->x = P_compensate(0);
            it_pcl->y = P_compensate(1);
            it_pcl->z = P_compensate(2);

            if (it_pcl == pcl_out.points.begin())
                break;
        }
    }
}

void ImuProcess::Process(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI::Ptr pcl_un_)
{
    double t1, t2, t3;
    t1 = omp_get_wtime();

    if (meas.imu.empty())
    {
        return;
    };
    ROS_ASSERT(meas.lidar != nullptr);

    if (imu_need_init_)
    {
        /// The very first lidar frame
        IMU_init(meas, kf_state, init_iter_num);

        imu_need_init_ = true;

        last_imu_ = meas.imu.back();

        state_ikfom imu_state = kf_state.get_x();
        if (init_iter_num > MAX_INI_COUNT)
        {
            cov_acc *= pow(G_m_s2 / mean_acc.norm(), 2);
            imu_need_init_ = false;

            cov_acc = cov_acc_scale;
            cov_gyr = cov_gyr_scale;
            ROS_INFO("IMU Initial Done");
            // ROS_INFO("IMU Initial Done: Gravity: %.4f %.4f %.4f %.4f; state.bias_g: %.4f %.4f %.4f; acc covarience: %.8f %.8f %.8f; gry covarience: %.8f %.8f %.8f",\
      //          imu_state.grav[0], imu_state.grav[1], imu_state.grav[2], mean_acc.norm(), cov_bias_gyr[0], cov_bias_gyr[1], cov_bias_gyr[2], cov_acc[0], cov_acc[1], cov_acc[2], cov_gyr[0], cov_gyr[1], cov_gyr[2]);
            // fout_imu.open(DEBUG_FILE_DIR("imu.txt"), std::ios::out);
        }

        return ;
    }

    // PointCloudXYZI::Ptr cur_pcl_un_ = PointCloudXYZI::Ptr(new PointCloudXYZI);
    UndistortPcl(meas, kf_state, *pcl_un_);

    t2 = omp_get_wtime();
    t3 = omp_get_wtime();

    // cout<<"[ IMU Process ]: Time: "<<t3 - t1<<endl;
}

MTK::get_cov<process_noise_ikfom>::type process_noise_cov()
{
	MTK::get_cov<process_noise_ikfom>::type cov = MTK::get_cov<process_noise_ikfom>::type::Zero();
	MTK::setDiagonal<process_noise_ikfom, vect3, 0>(cov, &process_noise_ikfom::ng, 0.0001);// 0.03
	MTK::setDiagonal<process_noise_ikfom, vect3, 3>(cov, &process_noise_ikfom::na, 0.0001); // *dt 0.01 0.01 * dt * dt 0.05
	MTK::setDiagonal<process_noise_ikfom, vect3, 6>(cov, &process_noise_ikfom::nbg, 0.00001); // *dt 0.00001 0.00001 * dt *dt 0.3 //0.001 0.0001 0.01
	MTK::setDiagonal<process_noise_ikfom, vect3, 9>(cov, &process_noise_ikfom::nba, 0.00001);   //0.001 0.05 0.0001/out 0.01
	return cov;
}

Eigen::Matrix<double, 24, 1> get_f(state_ikfom &s, const input_ikfom &in)
{
	Eigen::Matrix<double, 24, 1> res = Eigen::Matrix<double, 24, 1>::Zero();
	vect3 omega;
	in.gyro.boxminus(omega, s.bg);
	vect3 a_inertial = s.rot * (in.acc-s.ba); 
	for(int i = 0; i < 3; i++ ){
		res(i) = s.vel[i];
		res(i + 3) =  omega[i]; 
		res(i + 12) = a_inertial[i] + s.grav[i]; 
	}
	return res;
}

Eigen::Matrix<double, 24, 24> df_dx(state_ikfom &s, const input_ikfom &in)
{
	Eigen::Matrix<double, 24, 24> cov = Eigen::Matrix<double, 24, 24>::Zero();
	cov.template block<3, 3>(0, 12) = Eigen::Matrix3d::Identity();
	vect3 acc_;
	in.acc.boxminus(acc_, s.ba);
	vect3 omega;
	in.gyro.boxminus(omega, s.bg);
	cov.template block<3, 3>(12, 3) = -s.rot.toRotationMatrix()*MTK::hat(acc_);
	cov.template block<3, 3>(12, 18) = -s.rot.toRotationMatrix();
	// Eigen::Matrix<state_ikfom::scalar, 2, 1> vec = Eigen::Matrix<state_ikfom::scalar, 2, 1>::Zero();
	// Eigen::Matrix<state_ikfom::scalar, 3, 2> grav_matrix;
	// s.S2_Mx(grav_matrix, vec, 21);
	// cov.template block<3, 2>(12, 21) =  grav_matrix; 
    cov.template block<3, 3>(12, 21) = Eigen::Matrix3d::Identity(); // grav_matrix; 
	cov.template block<3, 3>(3, 15) = -Eigen::Matrix3d::Identity(); 
	return cov;
}


Eigen::Matrix<double, 24, 12> df_dw(state_ikfom &s, const input_ikfom &in)
{
	Eigen::Matrix<double, 24, 12> cov = Eigen::Matrix<double, 24, 12>::Zero();
	cov.template block<3, 3>(12, 3) = -s.rot.toRotationMatrix();
	cov.template block<3, 3>(3, 0) = -Eigen::Matrix3d::Identity();
	cov.template block<3, 3>(15, 6) = Eigen::Matrix3d::Identity();
	cov.template block<3, 3>(18, 9) = Eigen::Matrix3d::Identity();
	return cov;
}

vect3 SO3ToEuler(const SO3 &orient) 
{
	Eigen::Matrix<double, 3, 1> _ang;
	Eigen::Vector4d q_data = orient.coeffs().transpose();
	//scalar w=orient.coeffs[3], x=orient.coeffs[0], y=orient.coeffs[1], z=orient.coeffs[2];
	double sqw = q_data[3]*q_data[3];
	double sqx = q_data[0]*q_data[0];
	double sqy = q_data[1]*q_data[1];
	double sqz = q_data[2]*q_data[2];
	double unit = sqx + sqy + sqz + sqw; // if normalized is one, otherwise is correction factor
	double test = q_data[3]*q_data[1] - q_data[2]*q_data[0];

	if (test > 0.49999*unit) { // singularity at north pole
	
		_ang << 2 * std::atan2(q_data[0], q_data[3]), M_PI/2, 0;
		double temp[3] = {_ang[0] * 57.3, _ang[1] * 57.3, _ang[2] * 57.3};
		vect3 euler_ang(temp, 3);
		return euler_ang;
	}
	if (test < -0.49999*unit) { // singularity at south pole
		_ang << -2 * std::atan2(q_data[0], q_data[3]), -M_PI/2, 0;
		double temp[3] = {_ang[0] * 57.3, _ang[1] * 57.3, _ang[2] * 57.3};
		vect3 euler_ang(temp, 3);
		return euler_ang;
	}
		
	_ang <<
			std::atan2(2*q_data[0]*q_data[3]+2*q_data[1]*q_data[2] , -sqx - sqy + sqz + sqw),
			std::asin (2*test/unit),
			std::atan2(2*q_data[2]*q_data[3]+2*q_data[1]*q_data[0] , sqx - sqy - sqz + sqw);
	double temp[3] = {_ang[0] * 57.3, _ang[1] * 57.3, _ang[2] * 57.3};
	vect3 euler_ang(temp, 3);
		// euler_ang[0] = roll, euler_ang[1] = pitch, euler_ang[2] = yaw
	return euler_ang;
}
