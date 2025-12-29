/**
 * @file etsvi_node.cpp
 * @brief Energy-Tracking Space-Time Variational Integrator (ETSVI) Node
 * 能量跟踪时空变分积分器节点
 * * Implements a variational integrator with adaptive time-stepping based on energy error.
 * Uses CppAD for automatic differentiation of the Discrete Lagrangian.
 * 实现了一个基于能量误差自适应调整时间步长的变分积分器。
 * 使用 CppAD 对离散拉格朗日量进行自动微分。
 */

#include <cppad/cppad.hpp>
#include <cppad/example/cppad_eigen.hpp>

#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/aba.hpp>
#include <pinocchio/algorithm/energy.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>

#include <Eigen/Dense>
#include <rclcpp/rclcpp.hpp>

#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <numeric>
#include <string>
#include <iomanip>
#include <tuple>
#include <algorithm> // for std::clamp

// -----------------------------------------------------------------------------
// Compatibility Hacks | 兼容性补丁
// -----------------------------------------------------------------------------
// Fix Eigen <-> CppAD compatibility for isfinite()
// 修复 Eigen 与 CppAD 在 isfinite 函数上的兼容性问题
namespace Eigen {
    namespace numext {
        template<>
        EIGEN_STRONG_INLINE bool isfinite(const CppAD::AD<double>&) {
            return true;
        }
    }
}

// -----------------------------------------------------------------------------
// Type Definitions | 类型定义
// -----------------------------------------------------------------------------
using ADScalar = CppAD::AD<double>;                                       // 自动微分标量类型
using VecAD    = Eigen::Matrix<ADScalar, Eigen::Dynamic, 1>;              // AD 向量
using MatAD    = Eigen::Matrix<ADScalar, Eigen::Dynamic, Eigen::Dynamic>; // AD 矩阵
using Vec      = Eigen::VectorXd;                                         // 标准双精度向量
using Mat      = Eigen::MatrixXd;                                         // 标准双精度矩阵
using Model    = pinocchio::Model;
using Data     = pinocchio::Data;
using ModelAD  = pinocchio::ModelTpl<ADScalar>;                           // Pinocchio AD 模型
using DataAD   = pinocchio::DataTpl<ADScalar>;

// -----------------------------------------------------------------------------
// Helper Classes | 辅助类
// -----------------------------------------------------------------------------

/**
 * @brief Simple PID Controller for time-step adaptation.
 * 用于时间步长自适应的简单 PID 控制器。
 */
class PIDController {
public:
    PIDController(double kp, double ki, double kd, double i_min = -1e6, double i_max = 1e6)
        : kp_(kp), ki_(ki), kd_(kd), i_min_(i_min), i_max_(i_max) {}

    // 计算 PID 输出，用于调整 h (calculate delta_h)
    double update(double error, double dt) {
        if (dt <= 0.0) return 0.0;

        integral_ += error * dt;
        // Anti-windup clamping | 积分抗饱和截断
        integral_ = std::clamp(integral_, i_min_, i_max_);

        double deriv = (error - last_error_) / dt;
        last_error_ = error;

        return kp_ * error + ki_ * integral_ + kd_ * deriv;
    }

    void reset() {
        integral_ = 0.0;
        last_error_ = 0.0;
    }

private:
    double kp_, ki_, kd_;
    double integral_{0.0};
    double last_error_{0.0};
    double i_min_, i_max_;
};

// Solver execution statistics | 求解器统计信息
struct SolverStats {
    bool converged{false};     // 是否收敛
    std::string reason;        // 结束原因
    int iterations{0};         // 迭代次数
    double residual_norm{0.0}; // 残差范数
};

// -----------------------------------------------------------------------------
// CSV Utility | CSV 文件读写工具
// -----------------------------------------------------------------------------
namespace csv_utils {
    void write_matrix(const std::string &filename, const std::vector<Vec> &rows) {
        if (rows.empty()) return;
        std::ofstream ofs(filename);
        if (!ofs.is_open()) return;
        int ncols = static_cast<int>(rows[0].size());
        for (const auto &row : rows) {
            for (int c = 0; c < ncols; ++c) {
                ofs << std::setprecision(15) << row(c);
                if (c + 1 < ncols) ofs << ",";
            }
            ofs << "\n";
        }
    }

    void write_scalars(const std::string &filename, const std::vector<double> &rows) {
        std::ofstream ofs(filename);
        if (!ofs.is_open()) return;
        for (double v : rows) ofs << std::setprecision(15) << v << "\n";
    }

    void write_vectors3d(const std::string &filename, const std::vector<Eigen::Vector3d> &rows) {
        std::ofstream ofs(filename);
        if (!ofs.is_open()) return;
        for (const auto &v : rows) {
            ofs << std::setprecision(15) << v.x() << "," << v.y() << "," << v.z() << "\n";
        }
    }
}

// -----------------------------------------------------------------------------
// Core Solver Class | 核心求解器类
// -----------------------------------------------------------------------------
class ETSVISolver {
public:
    ETSVISolver(const std::string& urdf_path) {
        // 1. Load standard model | 加载标准物理模型
        pinocchio::urdf::buildModel(urdf_path, model_);
        model_.gravity.linear(Eigen::Vector3d(0, 0, -9.81)); // 设置重力
        data_ = std::make_unique<Data>(model_);

        // 2. Create AD model | 创建自动微分模型
        // cast<ADScalar>() transforms the double model to an AD model
        model_ad_ = model_.cast<ADScalar>();
        data_ad_ = std::make_unique<DataAD>(model_ad_);
    }

    Model& getModel() { return model_; }
    Data& getData() { return *data_; }
    int getNq() const { return model_.nq; }

    /**
     * @brief Computes Discrete Lagrangian L_d(q0, q1, h) using AD.
     * 利用自动微分计算离散拉格朗日量。
     * Formula: L_d = h * (T(q_mid, dq) - U(q_mid))
     */
    ADScalar discreteLagrangianAD(const VecAD &q0_ad, const VecAD &q1_ad, double h) {
        // Midpoint rule approximation | 中点法近似
        VecAD q_mid = ADScalar(0.5) * (q0_ad + q1_ad);
        VecAD dq = (q1_ad - q0_ad) / ADScalar(h);

        // Update kinematics and dynamics (CRBA computes Mass Matrix M)
        // 更新运动学和动力学 (CRBA 计算质量矩阵 M)
        pinocchio::crba(model_ad_, *data_ad_, q_mid);

        // Kinetic Energy: 0.5 * dq^T * M * dq
        MatAD M = data_ad_->M;
        ADScalar T = ADScalar(0.5) * dq.transpose() * (M * dq);

        // Potential Energy
        ADScalar U = pinocchio::computePotentialEnergy(model_ad_, *data_ad_, q_mid);

        return ADScalar(h) * (T - U);
    }

    /**
     * @brief Computes D1 = dL_d/dq0. Returns double vector.
     * 计算离散拉格朗日量对第一个位置参数 q0 的偏导数。
     */
    Vec computeD1(const Vec &q0, const Vec &q1, double h) {
        return computeGradient(q0, q1, h, true); // true indicates differentiation w.r.t first arg
    }

    /**
     * @brief Computes D2 = dL_d/dq1. Returns double vector.
     * 计算离散拉格朗日量对第二个位置参数 q1 的偏导数。
     */
    Vec computeD2(const Vec &q0, const Vec &q1, double h) {
        return computeGradient(q0, q1, h, false);
    }

    /**
     * @brief Solves implicit step equation: D2(q_{k-1}, q_k) + D1(q_k, q_{k+1}) = 0
     * 求解隐式步进方程 (离散欧拉-拉格朗日方程)。
     * Uses Newton-Raphson method. | 使用牛顿-拉夫逊法。
     * * @param q_prev q_{k-1}
     * @param q_curr q_k
     * @param tau    Generalized forces (torques) | 广义力
     * @param h      Time step | 时间步长
     */
    std::pair<Vec, SolverStats> solveNextState(const Vec &q_prev, const Vec &q_curr,
                                               const Vec &tau, double h,
                                               int max_iters = 50, double tol = 1e-8) {
        int n = model_.nq;
        Vec q_next = q_curr; // Initial guess | 初始猜测
        SolverStats info;

        // Precompute D2 once as it depends on fixed history (q_{k-1}, q_k)
        // D2 依赖于历史状态，在迭代过程中是常数，提前计算
        Vec D2 = computeD2(q_prev, q_curr, h);

        for (int it = 0; it < max_iters; ++it) {
            Vec D1 = computeD1(q_curr, q_next, h);

            // Residual equation: R = D2(k-1, k) + D1(k, k+1) + forces_term
            // 注意：力项通常是 -h * tau (外力做功的离散形式)
            Vec R = D2 + D1 - h * tau;

            info.residual_norm = R.norm();
            if (info.residual_norm < tol) {
                info.converged = true;
                info.iterations = it;
                return {q_next, info};
            }

            // Compute Jacobian J = dR/dq_next = d(D1)/dq_next
            // 计算残差对未知量 q_next 的雅可比矩阵
            Mat J = computeJacobianFD(q_curr, q_next, h);

            // Damped Newton step | 阻尼牛顿步
            Mat A = J + 1e-9 * Mat::Identity(n, n); // Regularization to avoid singularity
            Vec delta = A.colPivHouseholderQr().solve(-R);
            q_next += delta;
        }

        info.converged = false;
        info.reason = "max_iters_exceeded";
        return {q_next, info};
    }

    /**
     * @brief Numeric Discrete Energy for error tracking.
     * 计算数值离散能量，用于误差跟踪。
     * Ed = - dL_d / dh (Discrete Noether's Theorem | 离散诺特定理)
     */
    double computeDiscreteEnergy(const Vec &q0, const Vec &q1, double h) {
        double eps = 1e-8;
        // Central difference for dL/dh | 中心差分计算偏导
        double Lp = computeDiscreteLagrangianDouble(q0, q1, h + eps);
        double Lm = computeDiscreteLagrangianDouble(q0, q1, h - eps);
        return -(Lp - Lm) / (2.0 * eps);
    }

    // Compute standard Total Energy (T + U) | 计算标准机械能
    double computeTotalEnergy(const Vec &q, const Vec &v) {
        pinocchio::crba(model_, *data_, q);
        double T = 0.5 * v.transpose() * data_->M * v;
        double U = pinocchio::computePotentialEnergy(model_, *data_, q);
        return T + U;
    }

    // Get End-Effector Pose | 获取末端执行器位姿
    std::pair<Eigen::Vector3d, Eigen::Matrix3d> getEEPose(int frame_id, const Vec &q) {
        pinocchio::forwardKinematics(model_, *data_, q);
        pinocchio::updateFramePlacements(model_, *data_);
        return {data_->oMf[frame_id].translation(), data_->oMf[frame_id].rotation()};
    }

private:
    Model model_;
    std::unique_ptr<Data> data_;
    ModelAD model_ad_;
    std::unique_ptr<DataAD> data_ad_;

    // Internal helper for Lagrangian (double version) | 内部辅助函数：双精度拉格朗日量
    double computeDiscreteLagrangianDouble(const Vec &q0, const Vec &q1, double h) {
        Vec q_mid = 0.5 * (q0 + q1);
        Vec dq = (q1 - q0) / h;

        pinocchio::crba(model_, *data_, q_mid);
        double T = 0.5 * dq.transpose() * data_->M * dq;
        double U = pinocchio::computePotentialEnergy(model_, *data_, q_mid);
        return h * (T - U);
    }

    /**
     * @brief Generalized Gradient computation using CppAD
     * 通用的 CppAD 梯度计算函数
     * @param diff_first_arg If true, differentiate w.r.t q0 (D1), else q1 (D2)
     */
    Vec computeGradient(const Vec &q_const, const Vec &q_var, double h, bool diff_first_arg) {
        int n = model_.nq;
        CppAD::vector<ADScalar> x_ad(n);

        // Setup independent variables | 设置自变量
        // Depending on whether we compute D1 or D2, the "variable" is either q0 or q1
        Vec target_q = diff_first_arg ? q_const : q_var;
        for (int i=0; i<n; ++i) x_ad[i] = target_q[i];

        CppAD::Independent(x_ad); // Start recording tape | 开始录制计算图

        VecAD q_ad_in = Eigen::Map<VecAD>(x_ad.data(), n);
        VecAD q_other = (diff_first_arg ? q_var : q_const).cast<ADScalar>();

        ADScalar Ld;
        if (diff_first_arg) {
            // Calculating d(Ld(x, q_other))/dx -> D1
            Ld = discreteLagrangianAD(q_ad_in, q_other, h);
        } else {
            // Calculating d(Ld(q_other, x))/dx -> D2
            Ld = discreteLagrangianAD(q_other, q_ad_in, h);
        }

        CppAD::vector<ADScalar> y(1);
        y[0] = Ld;
        CppAD::ADFun<double> f(x_ad, y); // Define function f: x -> y

        CppAD::vector<double> x_val(n);
        for(int i=0; i<n; ++i) x_val[i] = target_q[i];

        // Compute gradient (Jacobian of scalar function is gradient)
        // 计算梯度 (标量函数的雅可比即为梯度)
        CppAD::vector<double> jac = f.Jacobian(x_val);

        return Eigen::Map<Vec>(jac.data(), n);
    }

    // Finite Difference Jacobian for Newton Solver | 牛顿法所需的有限差分雅可比
    Mat computeJacobianFD(const Vec &q_curr, const Vec &q_next, double h) {
        int n = model_.nq;
        Mat J(n, n);
        double eps = 1e-6;
        for (int j = 0; j < n; ++j) {
            Vec dq = Vec::Zero(n);
            dq(j) = eps;
            // Central difference: (D1(q+dq) - D1(q-dq)) / 2eps
            Vec D1p = computeD1(q_curr, q_next + dq, h);
            Vec D1m = computeD1(q_curr, q_next - dq, h);
            J.col(j) = (D1p - D1m) / (2.0 * eps);
        }
        return J;
    }
};

// -----------------------------------------------------------------------------
// Utils | 通用工具函数
// -----------------------------------------------------------------------------
std::string expand_user_path(const std::string &path) {
    if (!path.empty() && path[0] == '~') {
        const char* home = std::getenv("HOME");
        if (home) return std::string(home) + path.substr(1);
    }
    return path;
}

// Progress bar display | 进度条显示
void print_progress(double t_cur, double duration, double avg_step_time, double h) {
    constexpr int bar_width = 40;
    double progress = std::min(1.0, t_cur / duration);
    int pos = static_cast<int>(bar_width * progress);

    // Estimate remaining time | 预估剩余时间
    double time_left = (avg_step_time > 0 && h > 1e-9)
                       ? avg_step_time * (duration - t_cur) / h
                       : 0.0;

    std::cout << "\r[";
    for (int i = 0; i < bar_width; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << "% "
              << "| REM: " << int(time_left/60) << "m " << int(time_left)%60 << "s ";
    std::cout.flush();
}

// -----------------------------------------------------------------------------
// Main Node | 主节点入口
// -----------------------------------------------------------------------------
int main(int argc, char** argv) {
    std::cout.setf(std::ios::unitbuf); // Disable buffering for real-time log
    rclcpp::init(argc, argv);

    auto node = std::make_shared<rclcpp::Node>("etsvi_node",
        rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true)
    );

    // --- Parameter Loading | 参数加载 ---
    double q_init_val = node->get_parameter("q_init").as_double();
    double timestep   = node->get_parameter("timestep").as_double();
    double duration   = node->get_parameter("duration").as_double();
    std::string urdf  = expand_user_path(node->get_parameter("urdf_path").as_string());

    double pid_Kp = node->get_parameter("pid_Kp").as_double();
    double pid_Ki = node->get_parameter("pid_Ki").as_double();
    double pid_Kd = node->get_parameter("pid_Kd").as_double();

    RCLCPP_INFO(node->get_logger(), "ETSVI Init | URDF: %s", urdf.c_str());
    RCLCPP_INFO(node->get_logger(), "Sim Params | T=%.2fs, dt=%.4fs", duration, timestep);

    // --- Solver Initialization | 求解器初始化 ---
    ETSVISolver solver(urdf);
    int nq = solver.getNq();
    int tcp_id = solver.getModel().getFrameId("link_tcp"); // End-effector ID

    if (tcp_id < 0) {
        RCLCPP_ERROR(node->get_logger(), "Frame 'link_tcp' not found in URDF.");
        return 1;
    }

    // --- Initial Conditions | 初始条件 ---
    Vec q_prev = Vec::Constant(nq, q_init_val);
    Vec v_init = Vec::Zero(nq);
    Vec tau    = Vec::Zero(nq);

    // Compute q1 using fixed step first to bootstrap
    // 变分积分器需要 q_{k-1} 和 q_k 才能启动，第一步通常用定步长求解
    auto [q_curr, info_init] = solver.solveNextState(q_prev, q_prev + v_init * timestep, tau, timestep);

    if (!info_init.converged) {
        RCLCPP_FATAL(node->get_logger(), "Failed to initialize first step.");
        return 1;
    }

    // --- Data Logging Setup | 数据记录设置 ---
    std::vector<Vec> q_log;
    std::vector<double> t_log, h_log, E_log, dE_log;
    std::vector<Eigen::Vector3d> ee_log;
    std::vector<double> runtimes;

    // Initial log entry (t=0)
    q_log.push_back(q_prev);
    t_log.push_back(0.0);
    h_log.push_back(timestep); // Placeholder

    double E0 = solver.computeTotalEnergy(q_prev, v_init);
    E_log.push_back(E0);
    dE_log.push_back(0.0);
    ee_log.push_back(solver.getEEPose(tcp_id, q_prev).first);

    // Second log entry (from bootstrap step, t=dt)
    q_log.push_back(q_curr);
    t_log.push_back(timestep);
    h_log.push_back(timestep);

    double E1 = solver.computeTotalEnergy((q_prev + q_curr)/2.0, (q_curr - q_prev)/timestep);
    E_log.push_back(E1);
    dE_log.push_back(E1 - E0);
    ee_log.push_back(solver.getEEPose(tcp_id, q_curr).first);

    // Target discrete energy (constant for conservative systems)
    // 目标离散能量（对于保守系统应保持常数）
    double E_d_target = solver.computeDiscreteEnergy(q_prev, q_curr, timestep);

    // --- Simulation Loop | 仿真主循环 ---
    PIDController pid(pid_Kp, pid_Ki, pid_Kd);
    double t_now = timestep;

    auto t_start_wall = std::chrono::high_resolution_clock::now();

    RCLCPP_INFO(node->get_logger(), "Starting simulation loop...");

    while (t_now < duration && rclcpp::ok()) {
        auto t_step_start = std::chrono::high_resolution_clock::now();

        // 1. Get history | 获取历史状态
        const Vec& q_k_minus_1 = q_log[q_log.size() - 2];
        const Vec& q_k         = q_log.back();
        double h_prev          = h_log.back();

        // 2. Solve for q_{k+1} using h_prev (Trial Step) | 试探步
        // 猜测值 q_guess 基于当前速度
        Vec q_guess = q_k + (q_k - q_k_minus_1) * (h_prev / h_log[h_log.size()-2]);
        auto [q_next, info] = solver.solveNextState(q_k_minus_1, q_k, tau, h_prev);

        // 3. Compute Energy Error | 计算能量误差
        // 计算当前数值能量 E_numeric
        double E_numeric = solver.computeDiscreteEnergy(q_k, q_next, h_prev);

        // Scale invariant error normalization | 误差归一化（使其具有尺度不变性）
        double scale = std::max(std::abs(E_numeric), std::abs(E_d_target));
        double norm_factor = (scale < 1e-12) ? 1.0 : scale;
        double error = (E_numeric - E_d_target) / norm_factor;

        // 4. Update Time Step (h) via PID | 通过 PID 更新下一步的 h
        // Output delta_h represents the correction needed for the timestep
        double delta_h = pid.update(error, 0.01);
        double h_next = h_prev + delta_h;

        // Safety clamps (h > 1us) | 最小步长保护
        if (h_next < 1e-6) h_next = 1e-6;

        if (!info.converged) {
             RCLCPP_WARN_THROTTLE(node->get_logger(), *node->get_clock(), 1000,
                "Solver failed to converge at t=%.3f. Residual: %.2e", t_now, info.residual_norm);
        }

        // 5. Update State | 更新状态
        q_log.push_back(q_next);
        h_log.push_back(h_next);
        t_now += h_next;
        t_log.push_back(t_now);

        // 6. Logging Data | 记录数据
        double E_curr = solver.computeTotalEnergy((q_k + q_next)/2.0, (q_next - q_k)/h_next);
        E_log.push_back(E_curr);
        dE_log.push_back(E_curr - E_log[0]);
        ee_log.push_back(solver.getEEPose(tcp_id, q_next).first);

        // Timing stats | 时间统计
        auto t_step_end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(t_step_end - t_step_start).count();
        runtimes.push_back(elapsed);

        // Update UI every 10 steps | 每10步刷新一次进度
        double avg_time = std::accumulate(runtimes.begin(), runtimes.end(), 0.0) / runtimes.size();
        if (runtimes.size() % 10 == 0) {
            print_progress(t_now, duration, avg_time, h_next);
        }
    }
    std::cout << std::endl;

    // --- Save Results | 保存结果 ---
    std::string csv_dir = "src/vi/csv/etsvi/";
    RCLCPP_INFO(node->get_logger(), "Saving CSVs to %s ...", csv_dir.c_str());

    csv_utils::write_matrix(csv_dir + "q_history.csv", q_log);
    csv_utils::write_scalars(csv_dir + "time_history.csv", t_log);
    csv_utils::write_scalars(csv_dir + "energy_history.csv", E_log);
    csv_utils::write_scalars(csv_dir + "h_history.csv", h_log);
    csv_utils::write_scalars(csv_dir + "delta_energy_history.csv", dE_log);
    csv_utils::write_vectors3d(csv_dir + "ee_history.csv", ee_log);

    auto t_end_wall = std::chrono::high_resolution_clock::now();
    double total_wall = std::chrono::duration<double>(t_end_wall - t_start_wall).count();
    RCLCPP_INFO(node->get_logger(), "Done. Total Wall Time: %.2fs", total_wall);

    return 0;
}