/**
 * @file etsvi_node.cpp
 * @brief Energy-Tracking Space-Time Variational Integrator with Lyapunov Control
 * 修复 ModelAD 类型定义缺失的编译错误
 */

#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/aba.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/energy.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>

#include <cppad/cppad.hpp>
#include <cppad/example/cppad_eigen.hpp>

#include <eigen3/Eigen/Dense>
#include "rclcpp/rclcpp.hpp"

#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <string>
#include <iomanip>
#include <numeric>
#include <cstdlib> // for std::getenv

using namespace pinocchio;
using namespace std::chrono;

// ---------- 类型定义与兼容性补丁 ----------
namespace Eigen {
    namespace numext {
        template<> EIGEN_STRONG_INLINE bool isfinite(const CppAD::AD<double>&) { return true; }
    }
}

using ADScalar = CppAD::AD<double>;
using VecAD    = Eigen::Matrix<ADScalar, Eigen::Dynamic, 1>;
using Vec      = Eigen::VectorXd;
using Mat      = Eigen::MatrixXd;

// 【修复点】：显式定义 ModelAD 和 DataAD
using ModelAD  = pinocchio::ModelTpl<ADScalar>;
using DataAD   = pinocchio::DataTpl<ADScalar>;

// ---------- 辅助函数：路径处理 ----------
std::string expand_user_path(const std::string& path) {
    if (!path.empty() && path[0] == '~') {
        const char* home = std::getenv("HOME");
        if (home) return std::string(home) + path.substr(1);
    }
    return path;
}

// ---------- 辅助函数：CSV 写入 (保持模板风格) ----------
void write_csv(const std::string &filename, const std::vector<Vec> &rows) {
    if (rows.empty()) return;
    std::ofstream ofs(filename);
    int ncols = (int)rows[0].size();
    for (size_t r = 0; r < rows.size(); ++r) {
        for (int c = 0; c < ncols; ++c) {
            ofs << std::setprecision(15) << rows[r](c);
            if (c + 1 < ncols) ofs << ',';
        }
        ofs << '\n';
    }
}

void write_csv_scalar_series(const std::string &filename, const std::vector<double> &rows) {
    std::ofstream ofs(filename);
    for (double v : rows) ofs << std::setprecision(15) << v << '\n';
}

void write_csv_3d(const std::string &filename,
                  const std::vector<Eigen::Vector3d> &rows)
{
    std::ofstream ofs(filename);
    for (const auto &v : rows)
    {
        ofs << std::setprecision(15)
            << v(0) << "," << v(1) << "," << v(2) << "\n";
    }
}

// ---------- 核心求解器 (封装 CppAD 逻辑) ----------
class ETSVISolver {
public:
    ETSVISolver(const std::string& urdf_path) {
        pinocchio::urdf::buildModel(urdf_path, model);
        model.gravity.linear(Eigen::Vector3d(0, 0, -9.81));
        data = std::make_unique<Data>(model);

        // 初始化自动微分模型
        model_ad = model.cast<ADScalar>();
        data_ad = std::make_unique<DataAD>(model_ad);
    }

    Model& getModel() { return model; }
    Data& getData() { return *data; }
    int getNq() const { return model.nq; }

    // 计算能量对步长的梯度 dE/dh (Lyapunov 核心)
    double compute_energy_gradient(const Vec &q0, const Vec &q1, double h) {
        CppAD::vector<ADScalar> h_ad(1);
        h_ad[0] = h;
        CppAD::Independent(h_ad);

        VecAD q0_ad = q0.cast<ADScalar>();
        VecAD q1_ad = q1.cast<ADScalar>();
        VecAD q_mid = (q0_ad + q1_ad) * ADScalar(0.5);
        VecAD dq = (q1_ad - q0_ad) / h_ad[0];

        pinocchio::crba(model_ad, *data_ad, q_mid);
        data_ad->M.template triangularView<Eigen::Upper>() = data_ad->M.transpose().template triangularView<Eigen::Upper>();

        // 【关键修复】使用 .dot() 确保返回标量，避免 Eigen 表达式错误
        ADScalar T = ADScalar(0.5) * dq.dot(data_ad->M * dq);
        ADScalar U = pinocchio::computePotentialEnergy(model_ad, *data_ad, q_mid);

        CppAD::vector<ADScalar> y(1);
        y[0] = T + U;

        CppAD::ADFun<double> f(h_ad, y);
        CppAD::vector<double> h_val(1); h_val[0] = h;
        return f.Jacobian(h_val)[0];
    }

    // 计算离散拉格朗日梯度 D1/D2
    Vec compute_discrete_gradient(const Vec &q_c, const Vec &q_v, double h, bool first) {
        int n = model.nq;
        CppAD::vector<ADScalar> x(n);
        for(int i=0; i<n; ++i) x[i] = first ? q_c[i] : q_v[i];
        CppAD::Independent(x);

        VecAD qv_ad = Eigen::Map<VecAD>(x.data(), n);
        VecAD qc_ad = (first ? q_v : q_c).cast<ADScalar>();
        VecAD q_mid = (qv_ad + qc_ad) * ADScalar(0.5);
        VecAD dq = (first ? (qc_ad - qv_ad) : (qv_ad - qc_ad)) / ADScalar(h);

        pinocchio::crba(model_ad, *data_ad, q_mid);
        data_ad->M.template triangularView<Eigen::Upper>() = data_ad->M.transpose().template triangularView<Eigen::Upper>();

        ADScalar Ld = ADScalar(h) * (ADScalar(0.5) * dq.dot(data_ad->M * dq) - pinocchio::computePotentialEnergy(model_ad, *data_ad, q_mid));

        CppAD::vector<ADScalar> y(1); y[0] = Ld;
        CppAD::ADFun<double> f(x, y);
        CppAD::vector<double> xv(n);
        for(int i=0; i<n; ++i) xv[i] = (first ? q_c[i] : q_v[i]);
        return Eigen::Map<Vec>(f.Jacobian(xv).data(), n);
    }

    // 隐式牛顿迭代求解下一步
    std::pair<Vec, bool> solve_q_next(const Vec &q0, const Vec &q1, double h) {
        Vec q2 = q1; // 初始猜测
        Vec D2 = compute_discrete_gradient(q0, q1, h, false);

        for (int i = 0; i < 20; i++) {
            Vec D1 = compute_discrete_gradient(q1, q2, h, true);
            Vec R = D2 + D1;
            if (R.norm() < 1e-9) return {q2, true};

            // 有限差分 Jacobian (与模板保持一致)
            Mat J(model.nq, model.nq);
            double eps = 1e-8;
            for (int j = 0; j < model.nq; j++) {
                Vec dq = Vec::Zero(model.nq); dq[j] = eps;
                Vec D1_p = compute_discrete_gradient(q1, q2 + dq, h, true);
                Vec D1_m = compute_discrete_gradient(q1, q2 - dq, h, true);
                J.col(j) = (D1_p - D1_m) / (2.0 * eps);
            }
            // 简单的阻尼牛顿步
            Mat A = J + 1e-9 * Mat::Identity(model.nq, model.nq);
            q2 -= A.colPivHouseholderQr().solve(R);
        }
        return {q2, false};
    }

    // 计算总能量 (T + U)
    double compute_total_energy(const Vec &q, const Vec &v) {
        pinocchio::crba(model, *data, q);
        return 0.5 * v.transpose() * data->M * v + pinocchio::computePotentialEnergy(model, *data, q);
    }

    // 计算离散能量 (用于误差评估)
    double compute_discrete_energy(const Vec &q0, const Vec &q1, double h) {
        Vec q_mid = 0.5 * (q0 + q1);
        Vec dq = (q1 - q0) / h;
        pinocchio::crba(model, *data, q_mid);
        data->M.template triangularView<Eigen::Upper>() = data->M.transpose().template triangularView<Eigen::Upper>();
        return 0.5 * dq.dot(data->M * dq) + pinocchio::computePotentialEnergy(model, *data, q_mid);
    }

    // Get End-Effector Pose | 获取末端执行器位姿
    std::pair<Eigen::Vector3d, Eigen::Matrix3d> getEEPose(int frame_id, const Vec &q) {
        pinocchio::forwardKinematics(model, *data, q);
        pinocchio::updateFramePlacements(model, *data);
        return {data->oMf[frame_id].translation(), data->oMf[frame_id].rotation()};
    }

    Model model;
    std::unique_ptr<Data> data;
    ModelAD model_ad;
    std::unique_ptr<DataAD> data_ad;
};

// ---------- 主节点 ----------
int main(int argc, char** argv) {
    std::cout.setf(std::ios::unitbuf);
    rclcpp::init(argc, argv);

    auto node = std::make_shared<rclcpp::Node>("etsvi_node",
        rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true)
    );

    // 参数加载
    std::string urdf_raw = node->get_parameter("urdf_path").as_string();
    std::string urdf_path = expand_user_path(urdf_raw);

    double q_init     = node->get_parameter("q_init").as_double();
    double timestep   = node->get_parameter("timestep").as_double();
    double duration   = node->get_parameter("duration").as_double();

    // Lyapunov 控制增益
    double alpha = node->get_parameter("lyap_alpha").as_double();
    double beta = node->get_parameter("lyap_beta").as_double();

    RCLCPP_INFO(node->get_logger(), "Loading URDF: %s", urdf_path.c_str());
    RCLCPP_INFO(node->get_logger(), "Duration = %.3f, Timestep(init) = %.6f", duration, timestep);
    RCLCPP_INFO(node->get_logger(), "Lyapunov Gains: alpha=%.2f, beta=%.2f", alpha, beta);

    ETSVISolver solver(urdf_path);
    int n = solver.model.nq;
    int n_steps = static_cast<int>(duration / timestep);


    // 获取末端 frame id
    int link_tcp_id = solver.model.getFrameId("link_tcp");
    RCLCPP_INFO(node->get_logger(), "link_tcp_id=%d:", link_tcp_id);
    if (link_tcp_id == -1) {
        RCLCPP_ERROR(node->get_logger(), "TCP not found!");
        return 1;
    }

    RCLCPP_INFO(node->get_logger(), "Model has nq=%d nv=%d", solver.model.nq, solver.model.nv);

    // 初始状态
    Vec q_prev = Vec::Constant(n, q_init);
    Vec v_init = Vec::Zero(n);
    // 第一步：用定步长启动
    auto [q_curr, success] = solver.solve_q_next(q_prev, q_prev + v_init * timestep, timestep);
    if (!success) {
        RCLCPP_ERROR(node->get_logger(), "Failed to initialize first step.");
        return 1;
    }

    // 数据日志容器
    std::vector<Vec> q_history;
    std::vector<double> time_history;
    std::vector<double> energy_history;
    std::vector<double> delta_energy_history;
    std::vector<double> h_history;
    std::vector<double> runtimes;
    std::vector<Eigen::Vector3d> ee_history;

    q_history.push_back(q_prev);
    h_history.push_back(timestep);
    time_history.push_back(0.0);
    // 初始能量基准
    double E_ref = solver.compute_discrete_energy(q_prev, q_curr, timestep);
    energy_history.push_back(E_ref);
    delta_energy_history.push_back(energy_history.back()-E_ref);
    ee_history.push_back(solver.getEEPose(link_tcp_id, q_prev).first);

    q_history.push_back(q_curr);
    h_history.push_back(timestep);
    time_history.push_back(timestep);
    energy_history.push_back(E_ref);
    delta_energy_history.push_back(energy_history.back()-E_ref);
    ee_history.push_back(solver.getEEPose(link_tcp_id, q_curr).first);

    double t_cur = timestep;
    double h_next = timestep;
    double xi = 0.0; // 能量误差积分项

    auto t_start = high_resolution_clock::now();

    RCLCPP_INFO(node->get_logger(), "Starting Simulation Loop...");

    for (int step=0; step < n_steps -1  && t_cur < duration - 1e-12; ++step){
        auto t0 = high_resolution_clock::now();

        const Vec& q_km1 = q_history[q_history.size() - 2];
        const Vec& q_k   = q_history.back();

        // 1. 求解下一步 q_{k+1}
        auto [q_next, converged] = solver.solve_q_next(q_km1, q_k, h_next);

        if (!converged) {
            RCLCPP_WARN(node->get_logger(), "Solver diverged at t=%.3f, reducing step.", t_cur);
            h_next *= 0.5; // 简单回退策略
            continue;
        }

        // 2. Lyapunov 反馈控制计算
        double E_curr = solver.compute_discrete_energy(q_k, q_next, h_next);
        double e_k = E_curr - E_ref;
        xi += e_k; // 更新内模积分项

        double de_dh = solver.compute_energy_gradient(q_k, q_next, h_next);

        // 控制律: u = (de/dh)^+ * (-alpha * e - beta * xi)
        double reg = 1e-8; // 正则化
        double u_k = (de_dh / (de_dh * de_dh + reg)) * (-alpha * e_k - beta * xi);

        // 应用控制量并限幅
        h_next = std::clamp(h_next + u_k, 1e-5, 0.05);

        // 3. 记录数据
        q_history.push_back(q_next);
        h_history.push_back(h_next);
        energy_history.push_back(E_curr);
        delta_energy_history.push_back(energy_history.back()-E_ref);

        ee_history.push_back(solver.getEEPose(link_tcp_id, q_next).first);

        t_cur += h_next;
        time_history.push_back(t_cur);

        // 计时与进度显示 (完全复刻模板逻辑)
        auto t1 = high_resolution_clock::now();
        double elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count();
        runtimes.push_back(elapsed);

        double progress = t_cur / duration;
        if (progress > 1.0) progress = 1.0;

        if (step % 10 == 0) { // 降低刷新频率避免刷屏
            int bar_width = 50;
            int pos = int(bar_width * progress);
            double avg_time = !runtimes.empty() ? std::accumulate(runtimes.begin(), runtimes.end(), 0.0) / runtimes.size() : 0.0;
            double time_left = avg_time * (duration - t_cur) / h_next;

            std::cout << "\r[";
            for (int i = 0; i < bar_width; ++i) {
                if (i < pos) std::cout << "=";
                else if (i == pos) std::cout << ">";
                else std::cout << " ";
            }
            std::cout << "] " << int(progress * 100.0) << "%, approx "
                      << int(time_left/60) << "m " << int(time_left) % 60 << "s left... ";
            std::cout.flush();
        }
    }
    std::cout << "\n";

    auto t_end = high_resolution_clock::now();
    double total_elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_start).count();
    double avg_time = !runtimes.empty() ? std::accumulate(runtimes.begin(), runtimes.end(), 0.0) / runtimes.size() : 0.0;

    RCLCPP_INFO(node->get_logger(), "Simulation finished. Wall time: %.2fs, Avg step: %.2fms", total_elapsed, avg_time*1000);

    // 保存 CSV
    std::string csv_dir = "src/vi/csv/etsvi/";
    std::string cmd = "mkdir -p " + csv_dir;
    int unused = system(cmd.c_str()); (void)unused;

    write_csv(csv_dir + "q_history.csv", q_history);
    write_csv_scalar_series(csv_dir + "time_history.csv", time_history);
    write_csv_scalar_series(csv_dir + "energy_history.csv", energy_history);
    write_csv_scalar_series(csv_dir + "delta_energy_history.csv", delta_energy_history);
    write_csv_scalar_series(csv_dir + "h_history.csv", h_history);
    write_csv_3d(csv_dir + "ee_history.csv", ee_history);

    RCLCPP_INFO(node->get_logger(), "Data saved to %s", csv_dir.c_str());

    return 0;
}