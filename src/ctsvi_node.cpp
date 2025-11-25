//
// Created by space on 2025/11/7.
//
// variational_integrator.cpp
// C++ translation of your Python variational integrator example using Pinocchio + Eigen.
// Includes: discrete Lagrangian, finite-difference gradients D1/D2, Newton solver for implicit step,
// optional ABA-based forward dynamics step, logging to CSV for plotting in Python/Matplotlib.
//
// Build: a companion CMakeLists.txt is provided below in this same document.
// Requirements: Pinocchio (C++), Eigen3, a C++17 compiler.
// Typical build commands (from project root):
//   mkdir -p build && cd build
//   cmake ..
//   cmake --build . -- -j
// Run:
//   ./variational_integrator /path/to/your_robot.urdf
// The program writes q_history.csv, energy_history.csv, momentum_history.csv in the working dir.

#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/aba.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/energy.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>

#include <eigen3/Eigen/Dense>

#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <string>
#include <iomanip>
#include "rclcpp/rclcpp.hpp"


using namespace pinocchio;
using namespace std::chrono;

// ---------- Helper aliases ----------
using Vec = Eigen::VectorXd;
using Mat = Eigen::MatrixXd;

// Compute inertia (mass) matrix at configuration q
Mat inertia_matrix(const Model &model, Data &data, const Vec &q)
{
    pinocchio::crba(model, data, q);
    return data.M; // data.M is the inertia matrix
}

// Kinetic energy using midpoint q_mid and qdot
double kinetic_energy(const Model &model, Data &data, const Vec &q_mid, const Vec &qdot)
{
    // compute inertia at q_mid
    pinocchio::crba(model, data, q_mid);
    Mat M = data.M;
    return 0.5 * qdot.transpose() * M * qdot;
}

// Potential energy at q using Pinocchio helper
double potential_energy(const Model &model, Data &data, const Vec &q)
{
    // compute potential energy (fills data.potential_energy and returns it)
    double U = pinocchio::computePotentialEnergy(model, data, q);
    return U;
}

// Discrete Lagrangian L_d(q0, q1; h)
double discrete_lagrangian(const Model &model, Data &data, const Vec &q0, const Vec &q1, double h)
{
    Vec q_mid = 0.5 * (q0 + q1);
    Vec qdot = (q1 - q0) / h;
    double T = kinetic_energy(model, data, q_mid, qdot);
    double U = potential_energy(model, data, q_mid);
    return h * (T - U);
}

// Finite-difference gradient D1 = d L_d / d q0 (central difference)
Vec D_1(const Model &model, Data &data, const Vec &q0, const Vec &q1, double h, double eps=1e-6)
{
    int n = q0.size();
    Vec grad = Vec::Zero(n);
    for (int i = 0; i < n; ++i)
    {
        Vec dq = Vec::Zero(n);
        dq(i) = eps;
        double Lp = discrete_lagrangian(model, data, q0 + dq, q1, h);
        double Lm = discrete_lagrangian(model, data, q0 - dq, q1, h);
        grad(i) = (Lp - Lm) / (2.0 * eps);
    }
    return grad;
}

// Finite-difference gradient D2 = d L_d / d q1 (central difference)
Vec D_2(const Model &model, Data &data, const Vec &q0, const Vec &q1, double h, double eps=1e-6)
{
    int n = q1.size();
    Vec grad = Vec::Zero(n);
    for (int i = 0; i < n; ++i)
    {
        Vec dq = Vec::Zero(n);
        dq(i) = eps;
        double Lp = discrete_lagrangian(model, data, q0, q1 + dq, h);
        double Lm = discrete_lagrangian(model, data, q0, q1 - dq, h);
        grad(i) = (Lp - Lm) / (2.0 * eps);
    }
    return grad;
}

double compute_discrete_energy(
    const Model &model, Data &data,
    const Vec &qk, const Vec &qk1, double h)
{
    Vec v_mid = (qk1 - qk) / h;

    Vec D1 = D_1(model, data, qk, qk1, h);  // 已经实现了
    double Ld = discrete_lagrangian(model, data, qk, qk1, h);

    double Ed = - D1.dot(v_mid) - Ld;
    return Ed;
}

// Variational integrator initial step solver: solve q1 given q0, v0
struct SolverInfo { bool converged; std::string reason; int iterations; double residual_norm; };

std::pair<Vec, SolverInfo> VI_init(const Model &model, Data &data, const Vec &q0, const Vec &v0, const Vec &tau_k, double h, double eps=1e-6, int max_iters=50, double tol=1e-8)
{
    int n = q0.size();
    Vec q1 = q0;
    SolverInfo info{false, "", 0, 0.0};

    for (int it = 0; it < max_iters; ++it)
    {
        Mat M = inertia_matrix(model, data, q0);
        Vec D1 = D_1(model, data, q0, q1, h, eps);
        Vec R = M * v0 + D1 - h * tau_k;
        double normR = R.norm();
        info.iterations = it;
        info.residual_norm = normR;
        if (normR < tol) { info.converged = true; return {q1, info}; }

        // numerical Jacobian dR/dq1 = dD1/dq1
        Mat J = Mat::Zero(n,n);
        // double eps = 1e-6;
        for (int j = 0; j < n; ++j)
        {
            Vec dq = Vec::Zero(n);
            dq(j) = eps;
            Vec D1p = D_1(model, data, q0, q1 + dq, h, eps);
            Vec D1m = D_1(model, data, q0, q1 - dq, h, eps);
            J.col(j) = (D1p - D1m) / (2.0 * eps);
        }
        // regularize and solve
        Mat A = J + 1e-9 * Mat::Identity(n,n);
        Eigen::ColPivHouseholderQR<Mat> solver(A);
        if (solver.rank() < n)
        {
            info.converged = false;
            info.reason = "singular_jacobian";
            return {q1, info};
        }
        Vec delta = solver.solve(-R);
        q1 += delta;
    }
    info.converged = false;
    info.reason = "max_iters";
    return {q1, info};
}

// Solve q_next given q_prev, q_curr, tau_k
std::pair<Vec, SolverInfo> solve_q_next(const Model &model, Data &data, const Vec &q_prev, const Vec &q_curr, const Vec &tau_k, double h, double eps=1e-6, int max_iters=50, double tol=1e-8)
{
    int n = q_curr.size();
    Vec q_next = q_curr;
    SolverInfo info{false, "", 0, 0.0};

    for (int it = 0; it < max_iters; ++it)
    {
        Vec D2 = D_2(model, data, q_prev, q_curr, h, eps);
        Vec D1 = D_1(model, data, q_curr, q_next, h, eps);
        Vec R = D2 + D1 - h * tau_k;
        double normR = R.norm();
        info.iterations = it;
        info.residual_norm = normR;
        if (normR < tol) { info.converged = true; return {q_next, info}; }

        // numerical Jacobian dR/dq_next = dD1/dq_next
        Mat J = Mat::Zero(n,n);
        // double eps = 1e-6;
        for (int j = 0; j < n; ++j)
        {
            Vec dq = Vec::Zero(n);
            dq(j) = eps;
            Vec D1p = D_1(model, data, q_curr, q_next + dq, h, eps);
            Vec D1m = D_1(model, data, q_curr, q_next - dq, h, eps);
            J.col(j) = (D1p - D1m) / (2.0 * eps);
        }
        Mat A = J + 1e-9 * Mat::Identity(n,n);
        Eigen::ColPivHouseholderQR<Mat> solver(A);
        if (solver.rank() < n)
        {
            info.converged = false;
            info.reason = "singular_jacobian";
            return {q_next, info};
        }
        Vec delta = solver.solve(-R);
        q_next += delta;
    }
    info.converged = false;
    info.reason = "max_iters";
    return {q_next, info};
}

// Optional: Pinocchio ABA forward dynamics step (returns q_next, v_next)
std::pair<Vec, Vec> pinocchio_dynamics_step(const Model &model, Data &data, const Vec &q, const Vec &v, const Vec &tau, double h)
{
    // ABA: fills data.ddq
    pinocchio::aba(model, data, q, v, tau);
    Vec qdd = data.ddq;
    Vec v_next = v + h * qdd;
    Vec q_next = q + h * v_next;
    return {q_next, v_next};
}

// Write matrix (rows = time, cols = dofs) to CSV
void write_csv(const std::string &filename, const std::vector<Vec> &rows)
{
    if (rows.empty()) return;
    std::ofstream ofs(filename);
    int ncols = rows[0].size();
    for (size_t r = 0; r < rows.size(); ++r)
    {
        for (int c = 0; c < ncols; ++c)
        {
            ofs << std::setprecision(15) << rows[r](c);
            if (c + 1 < ncols) ofs << ',';
        }
        ofs << '\n';
    }
}

void write_csv_scalar_series(const std::string &filename, const std::vector<double> &rows)
{
    std::ofstream ofs(filename);
    for (double v : rows) ofs << std::setprecision(15) << v << '\n';
}

int main(int argc, char** argv)
{
    std::cout.setf(std::ios::unitbuf);
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>(
        "ctsvi_node",
        rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true)
    );

    // 获取参数值
    double q_init = node->get_parameter("q_init").as_double();
    double timestep = node->get_parameter("timestep").as_double();
    double duration = node->get_parameter("duration").as_double();
    double eps_diff = node->get_parameter("eps_diff").as_double();
    std::string urdf_path = node->get_parameter("urdf_path").as_string();

    // std::cout << "Using URDF: " << urdf_path << "\n";
    // std::cout << "Timestep: " << timestep << ", Duration: " << duration << "\n";

    RCLCPP_INFO(node->get_logger(), "Using URDF: %s", urdf_path.c_str());
    RCLCPP_INFO(node->get_logger(), "Initial = %.2f rad, Duration = %.1f s, Timestep = %.3f s, Eps_diff = %.1e", q_init, duration, timestep, eps_diff);

    // ====== 原 main 的剩余部分 ======
    Model model;
    try {
        pinocchio::urdf::buildModel(urdf_path, model);
    } catch (const std::exception &e) {
        RCLCPP_ERROR(node->get_logger(), "Error loading URDF");
        return 1;
    }
    Data data(model);

    // Set gravity vector (z negative)
    model.gravity.linear(Eigen::Vector3d(0,0,-9.81));

    // Print joints
    // std::cout << "Model has nq=" << model.nq << " nv=" << model.nv << " joints: \n";
    RCLCPP_INFO(node->get_logger(), "Model has nq=%d nv=%d joints:", model.nq, model.nv);
    // for (std::size_t i=0;i<model.names.size();++i)
    // {
    //     std::cout << i << " " << model.names[i] << " nq=" << model.joints[i].nq() << " nv=" << model.joints[i].nv() << "\n";
    // }

    int n = model.nq;
    int n_steps = static_cast<int>(duration / timestep);

    Vec q_prev = Vec::Constant(n, q_init);
    Vec v_prev = Vec::Zero(n);
    Vec tau_k = Vec::Zero(n);

    std::vector<Vec> q_history;
    std::vector<double> time_history;
    std::vector<double> energy_history;
    std::vector<double> energy_T_history;
    std::vector<double> energy_U_history;
    std::vector<double> delta_energy_history;
    std::vector<Vec> momentum_history;

    q_history.push_back(q_prev);

    auto t_start = high_resolution_clock::now();

    // 初始时刻
    Mat M0 = inertia_matrix(model, data, q_prev);
    Vec qdot0 = v_prev;
    double T0 = 0.5 * qdot0.transpose() * M0 * qdot0;
    double U0 = potential_energy(model, data, q_prev);
    double total_energy = T0 + U0;
    energy_history.push_back(total_energy);
    energy_T_history.push_back(T0);
    energy_U_history.push_back(U0);
    delta_energy_history.push_back(energy_history.back() - energy_history.front());

    // initial VI step: find q_curr from q_prev and v_prev
    auto [q_curr, info_init] = VI_init(model, data, q_prev, v_prev, tau_k, timestep, eps_diff);
    // std::cout << "VI_init: converged=" << info_init.converged << " it=" << info_init.iterations << " res=" << info_init.residual_norm << " reason=" << info_init.reason << "\n";
    // RCLCPP_INFO(node->get_logger(),
    //     "VI_init: converged=%d it=%d res=%f reason=%s",
    //     info_init.converged,
    //     info_init.iterations,
    //     info_init.residual_norm,
    //     info_init.reason.c_str());
    q_history.push_back(q_curr);

    // 可选：计算 q_curr 时刻的能量
    // Vec qdot1 = (q_curr - q_prev)/timestep;
    // Mat M1 = inertia_matrix(model, data, q_curr);
    // double T1 = 0.5 * qdot1.transpose() * M1 * qdot1;
    // double U1 = potential_energy(model, data, q_curr);
    // total_energy = T1 + U1;
    // energy_history.push_back(total_energy);
    // delta_energy_history.push_back(total_energy - energy_history.front());

    // double Ed = compute_discrete_energy(model, data, q_history[0], q_history[1], timestep);
    // energy_history.push_back(Ed);

    double T = kinetic_energy(model, data, (q_curr + q_prev) / 2.0, (q_curr - q_prev)/timestep);
    double U = potential_energy(model, data, (q_curr + q_prev) / 2.0);
    energy_T_history.push_back(T);
    energy_U_history.push_back(U);
    energy_history.push_back(T+U);
    delta_energy_history.push_back(energy_history.back() - energy_history.front());

    std::vector<double> runtimes;

    int bar_width = 50;
    double avg_time = 0.0;
    double time_left = 0.0;

    for (int step = 0; step < n_steps-1; ++step)
    {
        auto t0 = high_resolution_clock::now();
        auto [q_next, info] = solve_q_next(model, data, q_history[q_history.size()-2], q_history[q_history.size()-1], tau_k, timestep, eps_diff);
        q_history.push_back(q_next);
        time_history.push_back(step * timestep);

        // compute velocity qdot at current step
        // Vec qdot = (q_history.back() - q_history[q_history.size()-2]) / timestep;

        // inertia and kinetic
        // Mat M = inertia_matrix(model, data, q_history.back());
        // double T = 0.5 * qdot.transpose() * M * qdot;
        // double U = potential_energy(model, data, q_history.back());
        // total_energy = T + U;
        // energy_history.push_back(total_energy);
        // if (energy_history.size() == 1) delta_energy_history.push_back(0.0);
        // else delta_energy_history.push_back(total_energy - energy_history.front());
        // Vec p = M * qdot;
        // momentum_history.push_back(p);

        // Ed = compute_discrete_energy(model, data, q_history[q_history.size()-2], q_history.back(), timestep);
        // energy_history.push_back(Ed);

        T = kinetic_energy(model, data, (q_history.back() + q_history[q_history.size()-2]) / 2.0, (q_history.back() - q_history[q_history.size()-2])/timestep);
        U = potential_energy(model, data, (q_history.back() + q_history[q_history.size()-2]) / 2.0);
        energy_T_history.push_back(T);
        energy_U_history.push_back(U);
        energy_history.push_back(T+U);
        delta_energy_history.push_back(energy_history.back() - energy_history.front());

        auto t1 = high_resolution_clock::now();
        double elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count();
        runtimes.push_back(elapsed);

        // Debug prints
        // Vec p_plus = D_2(model, data, q_history[q_history.size()-2], q_history[q_history.size()-1], timestep);
        // std::cout << "step: " << step << " momentum norm: " << p.norm() << " p_plus norm: " << p_plus.norm() << " info.converged=" << info.converged << " it=" << info.iterations << "\n";
        // RCLCPP_INFO(node->get_logger(),
        //     "step: %d momentum norm: %f p_plus norm: %f info.converged=%d it=%d",
        //     step,
        //     p.norm(),
        //     p_plus.norm(),
        //     info.converged,
        //     info.iterations);

        // 计算完成比例
        double progress = double(step + 1) / n_steps;
        int pos = int(bar_width * progress);

        if (!runtimes.empty()) {
            avg_time = std::accumulate(runtimes.begin(), runtimes.end(), 0.0) / runtimes.size();
        }
        time_left = avg_time * (n_steps - step);

        // 构建进度条字符串
        std::cout << "\r[";
        for (int i = 0; i < bar_width; ++i) {
            if (i < pos) std::cout << "=";
            else if (i == pos) std::cout << ">";
            else std::cout << " ";
        }
        std::cout << "] " << int(progress * 100.0) << "%, " << int(time_left / 60) << "mins " << int(time_left) % 60 << "s left...";
        std::cout.flush();  // 强制输出
    }
    std::cout << "\n";
    time_history.push_back((n_steps-1) * timestep);
    time_history.push_back(n_steps * timestep);

    auto t_end = high_resolution_clock::now();
    double total_elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_start).count();
    if (!runtimes.empty()) {
        avg_time = std::accumulate(runtimes.begin(), runtimes.end(), 0.0) / runtimes.size();
    }

    // std::cout << "Simulation finished, wall time: " << total_elapsed << " s\n";
    // std::cout << "Average step time: " << avg_time*1e6 << " microseconds\n";
    RCLCPP_INFO(node->get_logger(),
        "Simulation finished, wall time: %f s, Average step time: %f ms",
        total_elapsed,
        avg_time*1e3);

    // Save CSVs
    write_csv("src/ctsvi/csv/ctsvi/q_history.csv", q_history);
    write_csv_scalar_series("src/ctsvi/csv/ctsvi/time_history.csv", time_history);
    write_csv_scalar_series("src/ctsvi/csv/ctsvi/energy_history.csv", energy_history);
    write_csv_scalar_series("src/ctsvi/csv/ctsvi/energy_T_history.csv", energy_T_history);
    write_csv_scalar_series("src/ctsvi/csv/ctsvi/energy_U_history.csv", energy_U_history);
    write_csv_scalar_series("src/ctsvi/csv/ctsvi/delta_energy_history.csv", delta_energy_history);
    // momentum: write per-row
    // write_csv("src/ctsvi/csv/ctsvi/momentum_history.csv", momentum_history);

    // std::cout << "Saved q_history.csv, energy_history.csv, momentum_history.csv\n";
    RCLCPP_INFO(node->get_logger(), "Saved csv.");

    // rclcpp::spin(node);
    // rclcpp::shutdown();

    return 0;
}
