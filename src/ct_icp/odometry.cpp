#include <omp.h>
#include <chrono>
#include "odometry.hpp"
#include "Utilities/PersoTimer.h"
#include "utils.hpp"
#include <iostream>
#include <fstream>

#define _USE_MATH_DEFINES

#include <math.h>

#ifdef CT_ICP_WITH_VIZ

#include <viz3d/engine.hpp>

#endif

namespace ct_icp {

    /* -------------------------------------------------------------------------------------------------------------- */
    OdometryOptions OdometryOptions::DefaultDrivingProfile() {
        return OdometryOptions{};
    }

    /* -------------------------------------------------------------------------------------------------------------- */
    // 相关的数据集使用的配置文件
    OdometryOptions OdometryOptions::RobustDrivingProfile() {
        OdometryOptions default_options;

        default_options.voxel_size = 0.5;
        default_options.sample_voxel_size = 1.5;
        default_options.max_distance = 200.0;
        default_options.min_distance_points = 0.15;
        default_options.init_num_frames = 20;
        default_options.max_num_points_in_voxel = 20;
        default_options.min_distance_points = 0.05;
        default_options.distance_error_threshold = 5.0;
        default_options.motion_compensation = CONTINUOUS;
        default_options.initialization = INIT_CONSTANT_VELOCITY;

        default_options.debug_print = false;
        default_options.debug_viz = false;
        default_options.robust_registration = true;
        default_options.robust_full_voxel_threshold = 0.5;
        default_options.robust_empty_voxel_threshold = 0.2;
        default_options.robust_num_attempts = 10;
        default_options.robust_max_voxel_neighborhood = 4;
        default_options.robust_threshold_relative_orientation = 5;
        default_options.robust_threshold_ego_orientation = 5;

        auto &ct_icp_options = default_options.ct_icp_options;
        ct_icp_options.debug_print = false;
        ct_icp_options.init_num_frames = 40;
        ct_icp_options.max_number_neighbors = 20;
        ct_icp_options.min_number_neighbors = 20;
        ct_icp_options.num_iters_icp = 15;
        ct_icp_options.max_dist_to_plane_ct_icp = 0.5;
        ct_icp_options.threshold_orientation_norm = 0.1;
        ct_icp_options.threshold_orientation_norm = 0.01;
        ct_icp_options.point_to_plane_with_distortion = true;
        ct_icp_options.distance = CT_POINT_TO_PLANE;
        ct_icp_options.num_closest_neighbors = 1;
        ct_icp_options.beta_constant_velocity = 0.001;
        ct_icp_options.beta_location_consistency = 0.001;
        ct_icp_options.beta_small_velocity = 0.00;
        ct_icp_options.loss_function = CAUCHY;
        ct_icp_options.solver = CERES;
        ct_icp_options.ls_max_num_iters = 20;
        ct_icp_options.ls_num_threads = 8;
        ct_icp_options.ls_sigma = 0.2;
        ct_icp_options.ls_tolerant_min_threshold = 0.05;
        return default_options;
    }

    /* -------------------------------------------------------------------------------------------------------------- */
    OdometryOptions OdometryOptions::DefaultRobustOutdoorLowInertia() {
        OdometryOptions default_options;
        default_options.voxel_size = 0.3;
        default_options.sample_voxel_size = 1.5;
        default_options.min_distance_points = 0.1;
        default_options.max_distance = 200.0;
        default_options.init_num_frames = 20;
        default_options.max_num_points_in_voxel = 20;
        default_options.distance_error_threshold = 5.0;
        default_options.motion_compensation = CONTINUOUS;
        default_options.initialization = INIT_NONE;
        default_options.debug_viz = false;
        default_options.debug_print = false;

        default_options.robust_registration = true;
        default_options.robust_full_voxel_threshold = 0.5;
        default_options.robust_empty_voxel_threshold = 0.1;
        default_options.robust_num_attempts = 3;
        default_options.robust_max_voxel_neighborhood = 4;
        default_options.robust_threshold_relative_orientation = 2;
        default_options.robust_threshold_ego_orientation = 2;

        auto &ct_icp_options = default_options.ct_icp_options;
        ct_icp_options.size_voxel_map = 0.8;
        ct_icp_options.num_iters_icp = 30;
        ct_icp_options.threshold_voxel_occupancy = 5;
        ct_icp_options.min_number_neighbors = 20;
        ct_icp_options.voxel_neighborhood = 1;

        ct_icp_options.init_num_frames = 20;
        ct_icp_options.max_number_neighbors = 20;
        ct_icp_options.min_number_neighbors = 20;
        ct_icp_options.max_dist_to_plane_ct_icp = 0.5;
        ct_icp_options.threshold_orientation_norm = 0.1;
        ct_icp_options.threshold_orientation_norm = 0.01;
        ct_icp_options.point_to_plane_with_distortion = true;
        ct_icp_options.distance = CT_POINT_TO_PLANE;
        ct_icp_options.num_closest_neighbors = 1;
        ct_icp_options.beta_constant_velocity = 0.0;
        ct_icp_options.beta_location_consistency = 0.001;
        ct_icp_options.beta_small_velocity = 0.01;
        ct_icp_options.loss_function = CAUCHY;
        ct_icp_options.solver = CERES;
        ct_icp_options.ls_max_num_iters = 10;
        ct_icp_options.ls_num_threads = 8;
        ct_icp_options.ls_sigma = 0.2;
        ct_icp_options.ls_tolerant_min_threshold = 0.05;
        ct_icp_options.weight_neighborhood = 0.2;
        ct_icp_options.weight_alpha = 0.8;
        ct_icp_options.weighting_scheme = ALL;
        ct_icp_options.max_num_residuals = 600;
        ct_icp_options.min_num_residuals = 200;


        return default_options;
    }

    /* -------------------------------------------------------------------------------------------------------------- */
    size_t Odometry::MapSize() const {
        return ::ct_icp::MapSize(voxel_map_);
    }

    /* -------------------------------------------------------------------------------------------------------------- */
    void DistortFrame(std::vector<Point3D> &points, Eigen::Quaterniond &begin_quat, Eigen::Quaterniond &end_quat,
                      Eigen::Vector3d &begin_t, Eigen::Vector3d &end_t) {
        Eigen::Quaterniond end_quat_I = end_quat.inverse(); // Rotation of the inverse pose
        Eigen::Vector3d end_t_I = -1.0 * (end_quat_I * end_t); // Translation of the inverse pose
        for (auto &point: points) {
            double alpha_timestamp = point.alpha_timestamp;
            Eigen::Quaterniond q_alpha = begin_quat.slerp(alpha_timestamp, end_quat);
            q_alpha.normalize();
            Eigen::Vector3d t = (1.0 - alpha_timestamp) * begin_t + alpha_timestamp * end_t;

            // Distort Raw Keypoints
            point.raw_pt = end_quat_I * (q_alpha * point.raw_pt + t) + end_t_I;
        }
    }

    inline void TransformPoint(MOTION_COMPENSATION compensation, Point3D &point3D,
                               Eigen::Quaterniond &q_begin, Eigen::Quaterniond &q_end,
                               Eigen::Vector3d &t_begin, Eigen::Vector3d &t_end) {
        Eigen::Vector3d t;
        Eigen::Matrix3d R;
        double alpha_timestamp = point3D.alpha_timestamp;
        switch (compensation) {
            case MOTION_COMPENSATION::NONE:
            case MOTION_COMPENSATION::CONSTANT_VELOCITY:   //已经对点的坐标进行了修正，因此每个点的R和t都可以看做是帧最后的motion
                R = q_end.toRotationMatrix();
                t = t_end;
                break;
            case MOTION_COMPENSATION::CONTINUOUS:
            case MOTION_COMPENSATION::ITERATIVE:  // 计算每个点对应的motion
                R = q_begin.slerp(alpha_timestamp, q_end).normalized().toRotationMatrix();
                t = (1.0 - alpha_timestamp) * t_begin + alpha_timestamp * t_end;
                break;
        }
        point3D.pt = R * point3D.raw_pt + t;  // 把点投影到世界坐标系下
    }

    /* -------------------------------------------------------------------------------------------------------------- */
    Odometry::RegistrationSummary Odometry::RegisterFrameWithEstimate(const std::vector<Point3D> &frame,
                                                                      const TrajectoryFrame &initial_estimate) {
        auto frame_index = InitializeMotion(&initial_estimate);
        return DoRegister(frame, frame_index);
    }

    /* -------------------------------------------------------------------------------------------------------------- */
    //  * * * * 配准当前帧
    Odometry::RegistrationSummary Odometry::RegisterFrame(const std::vector<Point3D> &frame) {
        auto frame_index = InitializeMotion();  // 没有参数的话，表示initial_eatimate为空指针
        return DoRegister(frame, frame_index);
    }

    /* -------------------------------------------------------------------------------------------------------------- */
    //  初始化 motion
    int Odometry::InitializeMotion(const TrajectoryFrame *initial_estimate) {
        //  每处理一帧, registered_frames_就会加1
        int index_frame = registered_frames_++;
        if (initial_estimate != nullptr) {
            // Insert previous estimate
            trajectory_.emplace_back(*initial_estimate);
            return index_frame;
        }

        // Initial Trajectory Estimate
        trajectory_.emplace_back(TrajectoryFrame());  // TrajectoryFrame 中的参数包含了 帧的初始和结束pose，以及相关的一些函数

        if (index_frame <= 1) {
            // Initialize first pose at Identity
            trajectory_[index_frame].begin_R = Eigen::MatrixXd::Identity(3, 3);
            trajectory_[index_frame].begin_t = Eigen::Vector3d(0., 0., 0.);
            trajectory_[index_frame].end_R = Eigen::MatrixXd::Identity(3, 3);
            trajectory_[index_frame].end_t = Eigen::Vector3d(0., 0., 0.);
        } else if (index_frame == 2) {
            if (options_.initialization == INIT_CONSTANT_VELOCITY) {
                // Different regimen for the second frame due to the bootstrapped elasticity
                // 由前两帧之间的运动，推出前一帧到当前帧的运动，和前一帧的运动相乘，得到当前帧的初始motion
                Eigen::Matrix3d R_next_end = 
                        trajectory_[index_frame - 1].end_R * trajectory_[index_frame - 2].end_R.inverse() *
                        trajectory_[index_frame - 1].end_R;
                Eigen::Vector3d t_next_end = trajectory_[index_frame - 1].end_t +
                                             trajectory_[index_frame - 1].end_R *
                                             trajectory_[index_frame - 2].end_R.inverse() *
                                             (trajectory_[index_frame - 1].end_t -
                                              trajectory_[index_frame - 2].end_t);

                trajectory_[index_frame].begin_R = trajectory_[index_frame - 1].end_R;
                trajectory_[index_frame].begin_t = trajectory_[index_frame - 1].end_t;
                trajectory_[index_frame].end_R = R_next_end;
                trajectory_[index_frame].end_t = t_next_end;
            } else {
                // Important ! Start with a rigid frame and let the ICP distort it !
                trajectory_[index_frame] = trajectory_[index_frame - 1];
                trajectory_[index_frame].end_t = trajectory_[index_frame].begin_t;
                trajectory_[index_frame].end_R = trajectory_[index_frame].begin_R;
            }
        } else {
            if (options_.initialization == INIT_CONSTANT_VELOCITY) {
                if (options_.motion_compensation == CONTINUOUS) {
                    // When continuous: use the previous begin_pose as reference  
                    // 当配置文件中的motion_compensation为CONTINUOUS时， 当前帧的初始pose由前两帧的pose得到
                    Eigen::Matrix3d R_next_begin =
                            trajectory_[index_frame - 1].begin_R * trajectory_[index_frame - 2].begin_R.inverse() *
                            trajectory_[index_frame - 1].begin_R;
                    Eigen::Vector3d t_next_begin = trajectory_[index_frame - 1].begin_t +
                                                   trajectory_[index_frame - 1].begin_R *
                                                   trajectory_[index_frame - 2].begin_R.inverse() *
                                                   (trajectory_[index_frame - 1].begin_t -
                                                    trajectory_[index_frame - 2].begin_t);

                    trajectory_[index_frame].begin_R = R_next_begin;; //trajectory_[index_frame - 1].end_R;
                    trajectory_[index_frame].begin_t = t_next_begin;; //trajectory_[index_frame - 1].end_t;
                } else {
                    // When not continuous: set the new begin and previous end pose to be consistent
                    // 如果运动补偿不是CONTINUOUS时，设置当前帧的初始pose和前一帧的结束pose一致。
                    trajectory_[index_frame].begin_R = trajectory_[index_frame - 1].end_R;
                    trajectory_[index_frame].begin_t = trajectory_[index_frame - 1].end_t;
                }

                // 无论有没有运动补偿， 当前帧的end_pose都是由前两帧计算出来的（基于常量motion）
                Eigen::Matrix3d R_next_end =
                        trajectory_[index_frame - 1].end_R * trajectory_[index_frame - 2].end_R.inverse() *
                        trajectory_[index_frame - 1].end_R;
                Eigen::Vector3d t_next_end = trajectory_[index_frame - 1].end_t +
                                             trajectory_[index_frame - 1].end_R *
                                             trajectory_[index_frame - 2].end_R.inverse() *
                                             (trajectory_[index_frame - 1].end_t -
                                              trajectory_[index_frame - 2].end_t);

                trajectory_[index_frame].end_R = R_next_end;
                trajectory_[index_frame].end_t = t_next_end;
            } else {
                trajectory_[index_frame] = trajectory_[index_frame - 1];
                // Important ! Start with a rigid frame and let the ICP distort it !
                trajectory_[index_frame] = trajectory_[index_frame - 1];
                trajectory_[index_frame].end_t = trajectory_[index_frame].begin_t;
                trajectory_[index_frame].end_R = trajectory_[index_frame].begin_R;
            }
        }
        return index_frame;
    }

    /* -------------------------------------------------------------------------------------------------------------- */
    std::vector<Point3D> Odometry::InitializeFrame(const std::vector<Point3D> &const_frame,
                                                   int index_frame) {

        /// PREPROCESS THE INITIAL FRAME
        // 判断当前帧是否小于 用于构造初始化map所需要的帧数， 若小于，则使用初始的voxel_size
        double sample_size = index_frame < options_.init_num_frames ? options_.init_voxel_size : options_.voxel_size;
        std::vector<Point3D> frame(const_frame);  //复制一个const_frame的复本

        std::mt19937_64 g;  //生成64位随机数
        std::shuffle(frame.begin(), frame.end(), g);  //对frame的元素进行随机排列
        //Subsample to keep one random point in every voxel of the current frame
        //在当前帧的每一个voxel中随机保留一个点
        sub_sample_frame(frame, sample_size);

        // No elastic ICP for first frame because no initialization of ego-motion
        // 对于第一帧，每个点的相对时间戳 都为1
        if (index_frame == 1) {
            for (auto &point3D: frame) {
                point3D.alpha_timestamp = 1.0;
            }
        }

        std::shuffle(frame.begin(), frame.end(), g);  //再次对frame中的元素进行随机打乱（此时frame中每个voxel中至多有一个点）

        if (index_frame > 1) {
            // 如果运动补偿是 CONSTANT_VELOCITY ，则对原始的点进行运动补偿
            if (options_.motion_compensation == CONSTANT_VELOCITY) {
                // The motion compensation of Constant velocity modifies the raw points of the point cloud
                auto &tr_frame = trajectory_[index_frame];
                Eigen::Quaterniond begin_quat(tr_frame.begin_R);
                Eigen::Quaterniond end_quat(tr_frame.end_R);
                // 利用每个点的时间戳，对每个点进行运动失真校正
                DistortFrame(frame, begin_quat, end_quat, tr_frame.begin_t, tr_frame.end_t);
            }

            auto q_begin = Eigen::Quaterniond(trajectory_[index_frame].begin_R);
            auto q_end = Eigen::Quaterniond(trajectory_[index_frame].end_R);
            Eigen::Vector3d t_begin = trajectory_[index_frame].begin_t;
            Eigen::Vector3d t_end = trajectory_[index_frame].end_t;
            for (auto &point3D: frame) {
                TransformPoint(options_.motion_compensation, point3D, q_begin, q_end, t_begin, t_end);
            }
            //此时frame中的点，已经通过begin和end pose投射到世界坐标系下了
        }

        double min_timestamp = std::numeric_limits<double>::max();
        double max_timestamp = std::numeric_limits<double>::min();
        // 计算当前帧中保留的点 对应的  开始和结束的时间戳
        for (auto &point: frame) {
            point.index_frame = index_frame;
            if (point.timestamp > max_timestamp)
                max_timestamp = point.timestamp;
            if (point.timestamp < min_timestamp)
                min_timestamp = point.timestamp;
        }

        trajectory_[index_frame].begin_timestamp = min_timestamp;
        trajectory_[index_frame].end_timestamp = max_timestamp;

        return frame;
    }

    /* -------------------------------------------------------------------------------------------------------------- */
    Odometry::RegistrationSummary Odometry::DoRegister(const std::vector<Point3D> &const_frame,
                                                       int index_frame) {
                                                           //const_frame表示的是帧中的点的集合；index_frame表示的是帧的序号
        auto start = std::chrono::steady_clock::now();
        auto &log_out = *log_out_;
        bool kDisplay = options_.debug_print;
        CTICPOptions ct_icp_options = options_.ct_icp_options; // Make a copy of the options

        //  关于voxel的配置， voxel的size以及voxel map的size
        const double kSizeVoxelInitSample = options_.voxel_size;
        const double kSizeVoxelMap = options_.ct_icp_options.size_voxel_map;
        const double kMinDistancePoints = options_.min_distance_points;
        const int kMaxNumPointsInVoxel = options_.max_num_points_in_voxel;

        if (kDisplay) {
            log_out << "/* ------------------------------------------------------------------------ */" << std::endl;
            log_out << "/* ------------------------------------------------------------------------ */" << std::endl;
            log_out << "REGISTRATION OF FRAME number " << index_frame <<
                    " with " << (options_.ct_icp_options.solver == CERES ? "CERES" : "GN") << " solver" << std::endl;
        }

        // ** setp1 初始化帧  （对帧中的点进行处理，使得每个voxel中最多保留一个点，此外根据运动补偿，把帧中的点投到世界坐标系下，并记录保留下来的点中的最大时间和最小时间）
        auto frame = InitializeFrame(const_frame, index_frame);
        if (kDisplay)
            log_out << "Number of points in sub-sampled frame: " << frame.size() << " / " << const_frame.size()
                    << std::endl;
        if (index_frame > 0) {
            Eigen::Vector3d t_diff = trajectory_[index_frame].end_t - trajectory_[index_frame].begin_t;
            if (kDisplay)
                log_out << "Initial ego-motion distance: " << t_diff.norm() << std::endl;
        }

        const auto initial_estimate = trajectory_.back();
        RegistrationSummary summary;
        summary.frame = initial_estimate;
        auto previous_frame = initial_estimate;


        if (index_frame > 0) {
            bool good_enough_registration = false;
            summary.number_of_attempts = 1;  // 只尝试配准一次
            // 对于前init_num_frames，使用init_sample_voxel_size
            double sample_voxel_size = index_frame < options_.init_num_frames ?
                                       options_.init_sample_voxel_size : options_.sample_voxel_size;
            double min_voxel_size = std::min(options_.init_voxel_size, options_.voxel_size);

            // 下面的一系列操作，貌似是通过调整CT-ICP相关的参数，提升鲁棒性
            auto increase_robustness_level = [&]() {
                previous_frame = summary.frame;
                // Handle the failure cases
                trajectory_[index_frame] = initial_estimate;
                ct_icp_options.voxel_neighborhood = std::min(++ct_icp_options.voxel_neighborhood,
                                                             options_.robust_max_voxel_neighborhood);
                ct_icp_options.ls_max_num_iters += 30;
                if (ct_icp_options.max_num_residuals > 0)
                    ct_icp_options.max_num_residuals = ct_icp_options.max_num_residuals * 2;
                ct_icp_options.num_iters_icp = min(ct_icp_options.num_iters_icp + 20, 50);
                ct_icp_options.threshold_orientation_norm = max(
                        ct_icp_options.threshold_orientation_norm / 10, 1.e-5);
                ct_icp_options.threshold_translation_norm = max(
                        ct_icp_options.threshold_orientation_norm / 10, 1.e-4);
                sample_voxel_size = std::max(sample_voxel_size / 1.5, min_voxel_size);
                ct_icp_options.ls_sigma *= 1.2;
                ct_icp_options.max_dist_to_plane_ct_icp *= 1.5;
            };

            summary.robust_level = 0;
            do {
                if (summary.robust_level < next_robust_level_) {
                    // Increase the robustness for the first iteration after a failure
                    summary.robust_level++;
                    increase_robustness_level();
                    continue;
                }
                auto start_ct_icp = std::chrono::steady_clock::now();

                // ** step2: 尝试配准
                TryRegister(frame, index_frame, ct_icp_options, summary, sample_voxel_size);
                auto end_ct_icp = std::chrono::steady_clock::now();
                std::chrono::duration<double> elapsed_icp = (end_ct_icp - start);
                if (kDisplay) {
                    log_out << "Elapsed Elastic_ICP: " << (elapsed_icp.count()) * 1000.0 << std::endl;
                    log_out << "Number of Keypoints extracted: " << summary.sample_size <<
                            " / Actual number of residuals: " << summary.number_of_residuals << std::endl;
                }

                // Compute Modification of trajectory  
                if (index_frame > 0) {
                    summary.distance_correction = (trajectory_[index_frame].begin_t -
                                                   trajectory_[index_frame - 1].end_t).norm();

                    Eigen::Matrix3d delta_R = (trajectory_[index_frame - 1].end_R *
                                               trajectory_[index_frame].begin_R.inverse());
                    summary.relative_orientation = AngularDistance(trajectory_[index_frame - 1].end_R,
                                                                   trajectory_[index_frame].end_R);
                    summary.ego_orientation = summary.frame.EgoAngularDistance();

                }
                summary.relative_distance = (trajectory_[index_frame].end_t - trajectory_[index_frame].begin_t).norm();

                good_enough_registration = AssessRegistration(frame, summary,
                                                              kDisplay ? &log_out : nullptr);
                if (options_.robust_fail_early)
                    summary.success = good_enough_registration;

                if (!good_enough_registration) {
                    if (options_.robust_registration && summary.number_of_attempts < options_.robust_num_attempts) {
                        // Either fail or
                        if (kDisplay)
                            log_out << "Registration Attempt n°" << summary.number_of_attempts
                                    << " failed with message: "
                                    << summary.error_message << std::endl;
                        double trans_distance = previous_frame.TranslationDistance(summary.frame);
                        double rot_distance = previous_frame.RotationDistance(summary.frame);
                        if (kDisplay)
                            log_out << "Distance to previous trans : " << trans_distance <<
                                    " rot distance " << rot_distance << std::endl;
                        increase_robustness_level();
                        summary.robust_level++;
                        summary.number_of_attempts++;
                    } else {
                        good_enough_registration = true;
                    }
                }
            } while (!good_enough_registration);

            trajectory_[index_frame].success = summary.success;

            if (!summary.success) {
                if (kDisplay)
                    log_out << "Failure to register, after " << summary.number_of_attempts << std::endl;
                return summary;
            }

            if (summary.number_of_attempts >= options_.robust_num_attempts)
                robust_num_consecutive_failures_++;
            else
                robust_num_consecutive_failures_ = 0;
        }

        if (kDisplay) {
            if (index_frame > 0) {
                log_out << "Trajectory correction [begin(t) - end(t-1)]: "
                        << summary.distance_correction << std::endl;
                log_out << "Final ego-motion distance: " << summary.relative_distance << std::endl;
            }
        }

        if (index_frame == 0) {
            voxel_map_.clear();
        }

        bool add_points = true;

        if (options_.robust_registration) {

            // Communicate whether we suspect an error due to too many attempts
            suspect_registration_error_ = summary.number_of_attempts >= options_.robust_num_attempts;
            if (kDisplay) {
                log_out << "[Robust Registration] "
                        << (suspect_registration_error_ ? "Suspect Registration due to a large number of attempts."
                                                        : "")
                        << "Might be failing. Consecutive failures: " << robust_num_consecutive_failures_ << std::endl;
                log_out << "[Robust Registration] The rotation ego motion is "
                        << summary.ego_orientation << " (deg)/ " << " relative orientation "
                        << summary.relative_orientation << " (deg) " << std::endl;
            }

            if (summary.ego_orientation > options_.robust_threshold_ego_orientation ||
                summary.relative_orientation > options_.robust_threshold_relative_orientation) {
                if (kDisplay)
                    log_out << "[Robust Registration] Change in orientation too important. "
                               "Points will not be added." << std::endl;
                add_points = false;
            }

            if (suspect_registration_error_) {
                if (robust_num_consecutive_failures_ > 5) {
                    if (kDisplay)
                        log_out << "Adding points despite failure" << std::endl;
                }
                add_points |= (robust_num_consecutive_failures_ > 5);
            }

            next_robust_level_ = add_points ? options_.robust_minimal_level : options_.robust_minimal_level + 1;
            if (!summary.success)
                next_robust_level_ = options_.robust_minimal_level + 2;
            else {
                if (summary.relative_orientation > options_.robust_threshold_relative_orientation ||
                    summary.ego_orientation > options_.robust_threshold_ego_orientation) {
                    next_robust_level_ = options_.robust_minimal_level + 1;
                }
                if (summary.number_of_attempts > 1) {
                    next_robust_level_ = options_.robust_minimal_level + 1;
                }
            }

        }

        if (add_points) {
            //Update Voxel Map+
            AddPointsToMap(voxel_map_, frame, kSizeVoxelMap,
                           kMaxNumPointsInVoxel, kMinDistancePoints);
        }

#ifdef CT_ICP_WITH_VIZ
        if (options_.debug_viz) {

            auto &instance = viz::ExplorationEngine::Instance();
            auto model_ptr = std::make_shared<viz::PointCloudModel>();
            auto &model_data = model_ptr->ModelData();
            model_data.xyz.reserve(MapSize());
            for (auto &voxel: voxel_map_) {
                for (int i(0); i < voxel.second.NumPoints(); ++i)
                    model_data.xyz.push_back(voxel.second.points[i].cast<float>());
            }
            model_data.point_size = 1;
            model_data.default_color = Eigen::Vector3f::Zero();
            instance.AddModel(-3, model_ptr);
        }
#endif


        // Remove voxels too far from actual position of the vehicule
        const double kMaxDistance = options_.max_distance;
        const Eigen::Vector3d location = trajectory_[index_frame].end_t;
        RemovePointsFarFromLocation(voxel_map_, location, kMaxDistance);


        if (kDisplay) {
            log_out << "Average Load Factor (Map): " << voxel_map_.load_factor() << std::endl;
            log_out << "Number of Buckets (Map): " << voxel_map_.bucket_count() << std::endl;
            log_out << "Number of points (Map): " << MapSize() << std::endl;
        }

        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        if (kDisplay) {
            log_out << "Elapsed Time: " << elapsed_seconds.count() * 1000.0 << " (ms)" << std::endl;
        }

        summary.corrected_points = frame;
        summary.all_corrected_points = const_frame;

        Eigen::Quaterniond q_begin(summary.frame.begin_R);
        Eigen::Quaterniond q_end(summary.frame.end_R);

        for (auto &point3D: summary.all_corrected_points) {
            double timestamp = point3D.alpha_timestamp;
            Eigen::Quaterniond slerp = q_begin.slerp(timestamp, q_end).normalized();
            point3D.pt = slerp.toRotationMatrix() * point3D.raw_pt +
                         summary.frame.begin_t * (1.0 - timestamp) + timestamp * summary.frame.end_t;
        }

        return summary;
    }


    /* -------------------------------------------------------------------------------------------------------------- */
    Odometry::RegistrationSummary Odometry::TryRegister(vector<Point3D> &frame, int index_frame,
                                                        const CTICPOptions &options,
                                                        RegistrationSummary &registration_summary,
                                                        double sample_voxel_size) {
        // Use new sub_sample frame as keypoints
        std::vector<Point3D> keypoints;
        grid_sampling(frame, keypoints, sample_voxel_size);  // 再次根据sample_voxel_size提取点（也是每个voxel中至多有一个点）

        auto num_keypoints = (int) keypoints.size();
        registration_summary.sample_size = num_keypoints;

        {

            //CT ICP
            ICPSummary icp_summary;
            // voxel_map_里面保存的是什么？？？？
            if (options_.ct_icp_options.solver == CT_ICP_SOLVER::GN) {

                icp_summary = CT_ICP_GN(options, voxel_map_, keypoints, trajectory_, index_frame);
            } else {
                icp_summary = CT_ICP_CERES(options, voxel_map_, keypoints, trajectory_, index_frame);
            }
            registration_summary.success = icp_summary.success;
            registration_summary.number_of_residuals = icp_summary.num_residuals_used;

            if (!registration_summary.success) {
                registration_summary.success = false;
                return registration_summary;
            }

            //Update frame   根据优化结果更新当前帧的motion
            auto q_begin = Eigen::Quaterniond(trajectory_[index_frame].begin_R);
            auto q_end = Eigen::Quaterniond(trajectory_[index_frame].end_R);
            Eigen::Vector3d t_begin = trajectory_[index_frame].begin_t;
            Eigen::Vector3d t_end = trajectory_[index_frame].end_t;
            for (auto &point: frame) {
                // Modifies the world point of the frame based on the raw_pt
                TransformPoint(options_.motion_compensation, point, q_begin, q_end, t_begin, t_end);
            }
        }
        registration_summary.keypoints = keypoints;
        registration_summary.frame = trajectory_[index_frame];
        return registration_summary;
    }

    /* -------------------------------------------------------------------------------------------------------------- */
    // 判断当前帧是否配准成功
    bool Odometry::AssessRegistration(const vector<Point3D> &points,
                                      RegistrationSummary &summary, std::ostream *log_stream) const {

        bool success = summary.success;
        if (summary.robust_level == 0 &&
            (summary.relative_orientation > options_.robust_threshold_relative_orientation ||
             summary.ego_orientation > options_.robust_threshold_ego_orientation)) {
            if (summary.robust_level < options_.robust_num_attempts_when_rotation) {
                summary.error_message = "Large rotations require at a robust_level of at least 1 (got:" +
                                        std::to_string(summary.robust_level) + ").";
                return false;
            }
        }

        if (summary.relative_distance > options_.robust_relative_trans_threshold) {
            summary.error_message = "The relative distance is too important";
            return false;
        }

        // Only do neighbor assessment if enough motion
        // 如果当前帧的motion > 0.1,进行评估
        bool do_neighbor_assessment = summary.distance_correction > 0.1;
        do_neighbor_assessment |= summary.relative_distance > options_.robust_neighborhood_min_dist;
        do_neighbor_assessment |= summary.relative_orientation > options_.robust_neighborhood_min_orientation;

        if (do_neighbor_assessment && registered_frames_ > options_.init_num_frames) {
            if (options_.robust_registration) {
                const double kSizeVoxelMap = options_.ct_icp_options.size_voxel_map;
                Voxel voxel;
                double ratio_empty_voxel = 0;
                double ratio_half_full_voxel = 0;

                for (auto &point: points) {
                    voxel = Voxel::Coordinates(point.pt, kSizeVoxelMap);
                    if (voxel_map_.find(voxel) == voxel_map_.end())
                        ratio_empty_voxel += 1;
                    if (voxel_map_.find(voxel) != voxel_map_.end() &&
                        voxel_map_.at(voxel).NumPoints() > options_.max_num_points_in_voxel / 2) {
                        // Only count voxels which have at least
                        ratio_half_full_voxel += 1;
                    }
                }

                ratio_empty_voxel /= points.size();
                ratio_half_full_voxel /= points.size();

                if (log_stream != nullptr)
                    *log_stream << "[Quality Assessment] Keypoint Ratio of voxel half occupied: " <<
                                ratio_half_full_voxel << std::endl
                                << "[Quality Assessment] Keypoint Ratio of empty voxel " <<
                                ratio_empty_voxel << std::endl;
                if (ratio_half_full_voxel < options_.robust_full_voxel_threshold ||
                    ratio_empty_voxel > options_.robust_empty_voxel_threshold) {
                    success = false;
                    if (ratio_empty_voxel > options_.robust_empty_voxel_threshold)
                        summary.error_message = "[Odometry::AssessRegistration] Ratio of empty voxels " +
                                                std::to_string(ratio_empty_voxel) + "above threshold.";
                    else
                        summary.error_message = "[Odometry::AssessRegistration] Ratio of half full voxels " +
                                                std::to_string(ratio_half_full_voxel) + "below threshold.";

                }
            }
        }

        if (summary.relative_distance > options_.distance_error_threshold) {
            if (log_stream != nullptr)
                *log_stream << "Error in ego-motion distance !" << std::endl;
            return false;
        }

        return success;
    }

    /* -------------------------------------------------------------------------------------------------------------- */
    std::vector<TrajectoryFrame> Odometry::Trajectory() const {
        return trajectory_;
    }

    /* -------------------------------------------------------------------------------------------------------------- */
    ArrayVector3d Odometry::GetLocalMap() const {
        return MapAsPointcloud(voxel_map_);
    }

    /* -------------------------------------------------------------------------------------------------------------- */
    Odometry::Odometry(
            const OdometryOptions &options) {
        options_ = options;
        options_.ct_icp_options.init_num_frames = options_.init_num_frames;  //构成初始地图的帧数目
        // Update the motion compensation
        switch (options_.motion_compensation) {
            case MOTION_COMPENSATION::NONE:
            case MOTION_COMPENSATION::CONSTANT_VELOCITY:
                // ElasticICP does not compensate the motion
                // 当运动补偿选项为  NONE或者CONSTANT_VELOCITY时，不进行运动补偿
                options_.ct_icp_options.point_to_plane_with_distortion = false;
                options_.ct_icp_options.distance = POINT_TO_PLANE;
                break;
            case MOTION_COMPENSATION::ITERATIVE:
                // ElasticICP compensates the motion at each ICP iteration
                // ITERATIVE指的是在每一次ICP优化之后，都对点进行运动失真补偿
                options_.ct_icp_options.point_to_plane_with_distortion = true;
                options_.ct_icp_options.distance = POINT_TO_PLANE;
                break;
            case MOTION_COMPENSATION::CONTINUOUS:
                // ElasticICP compensates continuously the motion
                // CONTINUOUS表示连续补偿运动（怎么做的？），其中distance固定为CT_POINT_TO_PLANE
                options_.ct_icp_options.point_to_plane_with_distortion = true;
                options_.ct_icp_options.distance = CT_POINT_TO_PLANE;
                break;
        }
        next_robust_level_ = options.robust_minimal_level;

        if (options_.log_to_file) {
            log_file_ = std::make_unique<std::ofstream>(options_.log_file_destination.c_str(),
                                                        std::ofstream::trunc);
            log_out_ = log_file_.get();
            *log_out_ << "Debug Print ?" << options_.debug_print << std::endl;
        } else
            log_out_ = &std::cout;
    }


    /* -------------------------------------------------------------------------------------------------------------- */
    ArrayVector3d MapAsPointcloud(const VoxelHashMap &map) {
        ArrayVector3d points;
        points.reserve(MapSize(map));
        for (auto &voxel: map) {
            for (int i(0); i < voxel.second.NumPoints(); ++i)
                points.push_back(voxel.second.points[i]);
        }
        return points;
    }

    /* -------------------------------------------------------------------------------------------------------------- */
    size_t MapSize(const VoxelHashMap &map) {
        size_t map_size(0);
        for (auto &itr_voxel_map: map) {
            map_size += (itr_voxel_map.second).NumPoints();
        }
        return map_size;
    }

    /* -------------------------------------------------------------------------------------------------------------- */
    void RemovePointsFarFromLocation(VoxelHashMap &map, const Eigen::Vector3d &location, double distance) {
        std::vector<Voxel> voxels_to_erase;
        for (auto &pair: map) {
            Eigen::Vector3d pt = pair.second.points[0];
            if ((pt - location).squaredNorm() > (distance * distance)) {
                voxels_to_erase.push_back(pair.first);
            }
        }
        for (auto &vox: voxels_to_erase)
            map.erase(vox);
    }

    /* -------------------------------------------------------------------------------------------------------------- */
    inline void AddPointToMap(VoxelHashMap &map, const Eigen::Vector3d &point, double voxel_size,
                              int max_num_points_in_voxel, double min_distance_points, int min_num_points = 0) {
        short kx = static_cast<short>(point[0] / voxel_size);
        short ky = static_cast<short>(point[1] / voxel_size);
        short kz = static_cast<short>(point[2] / voxel_size);

        VoxelHashMap::iterator search = map.find(Voxel(kx, ky, kz));
        if (search != map.end()) {
            auto &voxel_block = (search.value());

            if (!voxel_block.IsFull()) {
                double sq_dist_min_to_points = 10 * voxel_size * voxel_size;
                for (int i(0); i < voxel_block.NumPoints(); ++i) {
                    auto &_point = voxel_block.points[i];
                    double sq_dist = (_point - point).squaredNorm();
                    if (sq_dist < sq_dist_min_to_points) {
                        sq_dist_min_to_points = sq_dist;
                    }
                }
                if (sq_dist_min_to_points > (min_distance_points * min_distance_points)) {
                    if (min_num_points <= 0 || voxel_block.NumPoints() >= min_num_points) {
                        voxel_block.AddPoint(point);
                    }
                }
            }
        } else {
            if (min_num_points <= 0) {
                // Do not add points (avoids polluting the map)
                VoxelBlock block(max_num_points_in_voxel);
                block.AddPoint(point);
                map[Voxel(kx, ky, kz)] = std::move(block);
            }

        }

    }

    /* -------------------------------------------------------------------------------------------------------------- */
    void AddPointsToMap(VoxelHashMap &map, const vector<Point3D> &points, double voxel_size,
                        int max_num_points_in_voxel, double min_distance_points, int min_num_points) {
        //Update Voxel Map
        for (const auto &point: points) {
            AddPointToMap(map, point.pt, voxel_size, max_num_points_in_voxel, min_distance_points, min_num_points);
        }
    }

    /* -------------------------------------------------------------------------------------------------------------- */
    void AddPointsToMap(VoxelHashMap &map, const ArrayVector3d &points, double voxel_size,
                        int max_num_points_in_voxel, double min_distance_points) {
        for (const auto &point: points)
            AddPointToMap(map, point, voxel_size, max_num_points_in_voxel, min_distance_points);
    }
    /* -------------------------------------------------------------------------------------------------------------- */


} // namespace ct_icp