//
// Created by space on 2025/11/18.
//
// variational_integrator_adaptive.cpp
// Migrated SEM adaptive-step solver from your python prototype into C++ (Pinocchio + Eigen + rclcpp).
// Replace your existing variational_integrator.cpp with this file (or merge changes).

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
#include <numeric>

using namespace pinocchio;
using namespace std::chrono;

// ---------- Helper aliases ----------
using Vec = Eigen::VectorXd;
using Mat = Eigen::MatrixXd;

// ---------- existing helpers (unchanged logic) ----------

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
    int n = (int)q0.size();
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
    int n = (int)q1.size();
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

// discrete energy Ed = -dL/dh  (numerical central difference)
///////////gai
double discrete_energy(const Model &model, Data &data, const Vec &q0, const Vec &q1, double h, double eps_h=1e-6)
{
    double Lp = discrete_lagrangian(model, data, q0, q1, h + eps_h);
    double Lm = discrete_lagrangian(model, data, q0, q1, h - eps_h);
    double dLdh = (Lp - Lm) / (2.0 * eps_h);
    return -dLdh;
}

// Solver info
struct SolverInfo { bool converged; std::string reason; int iterations; double residual_norm; };

// Your original VI_init (kept for initial step / fallback)
std::pair<Vec, SolverInfo> VI_init(const Model &model, Data &data, const Vec &q0, const Vec &v0, const Vec &tau_k, double h, double eps=1e-6, int max_iters=50, double tol=1e-8)
{
    int n = (int)q0.size();
    Vec q1 = q0 + h * v0; // slightly better initial guess
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

        Mat J = Mat::Zero(n,n);
        for (int j = 0; j < n; ++j)
        {
            Vec dq = Vec::Zero(n);
            dq(j) = eps;
            Vec D1p = D_1(model, data, q0, q1 + dq, h, eps);
            Vec D1m = D_1(model, data, q0, q1 - dq, h, eps);
            J.col(j) = (D1p - D1m) / (2.0 * eps);
        }
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

// Fixed-step fallback solver (your previous solve_q_next)
std::pair<Vec, SolverInfo> solve_q_next_fixed(const Model &model, Data &data, const Vec &q_prev, const Vec &q_curr, double h, double eps=1e-6, int max_iters=50, double tol=1e-8)
{
    int n = (int)q_curr.size();
    Vec q_next = q_curr;
    SolverInfo info{false, "", 0, 0.0};

    for (int it = 0; it < max_iters; ++it)
    {
        Vec D2 = D_2(model, data, q_prev, q_curr, h, eps);
        Vec D1 = D_1(model, data, q_curr, q_next, h, eps);
        Vec R = h * D2 + h * D1;
        double normR = R.norm();
        info.iterations = it;
        info.residual_norm = normR;
        if (normR < tol) { info.converged = true; return {q_next, info}; }

        Mat J = Mat::Zero(n, n);
        for (int j = 0; j < n; ++j)
        {
            Vec dq = Vec::Zero(n);
            dq(j) = eps;
            Vec D1p = D_1(model, data, q_curr, q_next + dq, h, eps);
            Vec D1m = D_1(model, data, q_curr, q_next - dq, h, eps);
            J.col(j) = h * (D1p - D1m) / (2.0 * eps);
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

// ---------- SEM adaptive solver (C++ port of your python solve_q_next_sem) ----------
std::tuple<Vec, double, SolverInfo> solve_q_next_sem(
    const Model &model, Data &data,
    const Vec &q_prev, const Vec &q_curr,
    double h_prev,
    const Vec &qdot_guess,
    double eps_q = 1e-6,
    double eps_h = 1e-6,
    int max_iters = 100,
    double tol = 1e-8,
    double h_min = 1e-6,
    double h_max = 0.1)
{
    int n = (int)q_curr.size();
    // initial guesses
    Vec q_next = q_curr + h_prev * qdot_guess;
    double h_next = h_prev;

    double E_prev = discrete_energy(model, data, q_prev, q_curr, h_prev, eps_h);

    SolverInfo info{false, "", 0, 0.0};
    Vec Rvec; Rvec.setZero(n+1);

    for (int it = 0; it < max_iters; ++it)
    {
        Vec D2 = D_2(model, data, q_prev, q_curr, h_prev, eps_q); // depends on h_prev only
        Vec D1 = D_1(model, data, q_curr, q_next, h_next, eps_q);

        Vec gvec = h_prev * D2 + h_next * D1; // size n
        double E_next = discrete_energy(model, data, q_curr, q_next, h_next, eps_h);
        double fval = E_prev - E_next;

        Rvec.resize(n+1);
        for (int i=0;i<n;++i) Rvec(i) = gvec(i);
        Rvec(n) = fval;

        double normR = Rvec.norm();
        info.iterations = it;
        info.residual_norm = normR;
        if (normR < tol)
        {
            info.converged = true;
            return {q_next, h_next, info};
        }

        // Build Jacobian J (n+1 x n+1)
        Mat J = Mat::Zero(n+1, n+1);

        // columns 0..n-1: perturb q_next
        for (int j = 0; j < n; ++j)
        {
            Vec dq = Vec::Zero(n);
            dq(j) = eps_q;
            Vec D1p = D_1(model, data, q_curr, q_next + dq, h_next, eps_q);
            Vec D1m = D_1(model, data, q_curr, q_next - dq, h_next, eps_q);
            // derivative of gvec wrt q_next_j: h_next * d(D1)/dq_next_j
            Vec col = h_next * (D1p - D1m) / (2.0 * eps_q);
            for (int i=0;i<n;++i) J(i, j) = col(i);

            // derivative of f wrt q_next_j: - d(Ed(q_curr,q_next,h_next))/dq_next_j
            double Ep = discrete_energy(model, data, q_curr, q_next + dq, h_next, eps_h);
            double Em = discrete_energy(model, data, q_curr, q_next - dq, h_next, eps_h);
            double df_dqj = - (Ep - Em) / (2.0 * eps_q);
            J(n, j) = df_dqj;
        }

        // column n: perturb h_next
        double dh = eps_h;
        Vec D1p_h = D_1(model, data, q_curr, q_next, h_next + dh, eps_q);
        Vec D1m_h = D_1(model, data, q_curr, q_next, h_next - dh, eps_q);
        Vec dD1_dh = (D1p_h - D1m_h) / (2.0 * dh);
        // ∂g/∂h_next = D1 + h_next * ∂D1/∂h_next
        Vec col_h = D1 + h_next * dD1_dh;
        for (int i=0;i<n;++i) J(i, n) = col_h(i);

        double Ep_h = discrete_energy(model, data, q_curr, q_next, h_next + dh, eps_h);
        double Em_h = discrete_energy(model, data, q_curr, q_next, h_next - dh, eps_h);
        double dEd_dh = (Ep_h - Em_h) / (2.0 * dh);
        // derivative of f = E_prev - E_next w.r.t h_next is - dE_next/dh
        J(n, n) = - dEd_dh;

        // Solve linear system J * delta = -R
        Mat Jreg = J + 1e-9 * Mat::Identity(n+1, n+1);
        Eigen::ColPivHouseholderQR<Mat> solver(Jreg);
        if (solver.rank() < n+1)
        {
            info.converged = false;
            info.reason = "singular_jacobian";
            return {q_next, h_next, info};
        }
        Vec delta = solver.solve(-Rvec);

        Vec dq = delta.head(n);
        double dh_scalar = delta(n);

        // line search / backtracking for stability and bounds
        double alpha = 1.0;
        bool accept = false;
        for (int ls = 0; ls < 10; ++ls)
        {
            Vec q_next_trial = q_next + alpha * dq;
            double h_next_trial = h_next + alpha * dh_scalar;
            if (h_next_trial <= h_min || h_next_trial >= h_max)
            {
                alpha *= 0.5;
                continue;
            }
            Vec D1_trial = D_1(model, data, q_curr, q_next_trial, h_next_trial, eps_q);
            Vec gvec_trial = h_prev * D2 + h_next_trial * D1_trial;
            double E_trial = discrete_energy(model, data, q_curr, q_next_trial, h_next_trial, eps_h);
            double f_trial = E_prev - E_trial;
            Vec R_trial(n+1);
            for (int i=0;i<n;++i) R_trial(i) = gvec_trial(i);
            R_trial(n) = f_trial;
            if (R_trial.norm() < (1.0 - 1e-4 * alpha) * normR || alpha < 1e-3)
            {
                q_next = q_next_trial;
                h_next = h_next_trial;
                accept = true;
                break;
            }
            alpha *= 0.5;
        }
        if (!accept)
        {
            info.converged = false;
            info.reason = "line_search_failed";
            info.iterations = it;
            info.residual_norm = normR;
            return {q_next, h_next, info};
        }
    }

    info.converged = false;
    info.reason = "max_iters";
    return {q_next, h_next, info};
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
    int ncols = (int)rows[0].size();
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
        "atsvi_node",
        rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true)
    );

    double q_init = node->get_parameter("q_init").as_double();
    double timestep = node->get_parameter("timestep").as_double();
    double duration = node->get_parameter("duration").as_double();
    double eps_diff = node->get_parameter("eps_diff").as_double();
    std::string urdf_path = node->get_parameter("urdf_path").as_string();
    double h_min = node->get_parameter("h_min").as_double();
    double h_max = node->get_parameter("h_max").as_double();
    int max_adapt_iters = node->get_parameter("max_adapt_iters").as_int();

    RCLCPP_INFO(node->get_logger(), "Using URDF: %s", urdf_path.c_str());
    RCLCPP_INFO(node->get_logger(), "Duration = %.3f, Timestep (initial) = %.6f, Eps_diff = %.1e", duration, timestep, eps_diff);
    RCLCPP_INFO(node->get_logger(), "h_min=%.1e h_max=%.6f max_adapt_iters=%d", h_min, h_max, max_adapt_iters);

    Model model;
    try {
        pinocchio::urdf::buildModel(urdf_path, model);
    } catch (const std::exception &e) {
        RCLCPP_ERROR(node->get_logger(), "Error loading URDF");
        return 1;
    }
    Data data(model);

    model.gravity.linear(Eigen::Vector3d(0,0,-9.81));
    RCLCPP_INFO(node->get_logger(), "Model has nq=%d nv=%d", model.nq, model.nv);

    int n = model.nq;

    // initial conditions
    Vec q_prev = Vec::Constant(n, q_init);
    Vec v_prev = Vec::Zero(n);
    Vec tau_k = Vec::Zero(n);

    std::vector<Vec> q_history;
    std::vector<double> time_history;
    std::vector<double> energy_history;
    std::vector<double> delta_energy_history;
    std::vector<Vec> momentum_history;
    std::vector<double> h_history;
    std::vector<double> runtimes;

    q_history.push_back(q_prev);

    auto t_start = high_resolution_clock::now();

    // initial energy
    Mat M0 = inertia_matrix(model, data, q_prev);
    Vec qdot0 = v_prev;
    double T0 = 0.5 * qdot0.transpose() * M0 * qdot0;
    double U0 = potential_energy(model, data, q_prev);
    double total_energy = T0 + U0;
    energy_history.push_back(total_energy);
    delta_energy_history.push_back(energy_history.back() - energy_history.front());

    // initial VI step to find q_curr (you may optionally change to an SEM first step)
    auto [q_curr, info_init] = VI_init(model, data, q_prev, v_prev, tau_k, timestep, eps_diff);
    // RCLCPP_INFO(node->get_logger(), "VI_init: converged=%d it=%d res=%f reason=%s", info_init.converged, info_init.iterations, info_init.residual_norm, info_init.reason.c_str());
    q_history.push_back(q_curr);

    // compute energy at q_curr
    // Vec qdot1 = (q_curr - q_prev)/timestep;
    // Mat M1 = inertia_matrix(model, data, q_curr);
    // double T1 = 0.5 * qdot1.transpose() * M1 * qdot1;
    // double U1 = potential_energy(model, data, q_curr);
    // total_energy = T1 + U1;
    // energy_history.push_back(total_energy);
    // delta_energy_history.push_back(total_energy - energy_history.front());

    double T = kinetic_energy(model, data, (q_curr + q_prev) / 2.0, (q_curr - q_prev)/timestep);
    double U = potential_energy(model, data, (q_curr + q_prev) / 2.0);
    total_energy = T+U;
    energy_history.push_back(total_energy);
    delta_energy_history.push_back(energy_history.back() - energy_history.front());

    // adaptive integration loop: advance until time >= duration
    double t_cur = 0.0;
    t_cur += timestep; // we have advanced to q_curr at t = timestep
    time_history.push_back(0.0);
    time_history.push_back(t_cur);
    h_history.push_back(timestep);

    int max_steps = std::max( (int)(duration / (h_min)), 1000000); // safeguard
    int step = 0;

    for (step = 0; step < max_steps && t_cur < duration - 1e-12; ++step)
    {
        auto t0 = high_resolution_clock::now();

        Vec qdot_guess = (q_history.back() - q_history[q_history.size()-2]) / h_history.back();

        // call SEM adaptive solver
        auto [q_next, h_next, info_adapt] = solve_q_next_sem(
            model, data,
            q_history[q_history.size()-2],
            q_history[q_history.size()-1],
            h_history.back(),
            qdot_guess,
            /*eps_q=*/eps_diff,
            /*eps_h=*/eps_diff,
            /*max_iters=*/max_adapt_iters,
            /*tol=*/1e-8,
            h_min,
            h_max
        );

        if (!info_adapt.converged)
        {
            // fallback to fixed-step DEL with previous step h
            RCLCPP_WARN(node->get_logger(), "Step %d: SEM solver failed (%s). Falling back to fixed-step DEL.", step, info_adapt.reason.c_str());
            auto [q_fix, info_fix] = solve_q_next_fixed(model, data, q_history[q_history.size()-2], q_history[q_history.size()-1], h_history.back(), eps_diff);
            q_next = q_fix;
            h_next = h_history.back();
            RCLCPP_INFO(node->get_logger(), "Fixed-step info: converged=%d it=%d res=%f reason=%s", info_fix.converged, info_fix.iterations, info_fix.residual_norm, info_fix.reason.c_str());
        }

        q_history.push_back(q_next);
        h_history.push_back(h_next);
        t_cur += h_next;
        time_history.push_back(t_cur);

        // energy & momentum record
        // Vec qdot = (q_history.back() - q_history[q_history.size()-2]) / h_next;
        // Mat M = inertia_matrix(model, data, q_history.back());
        // double T = 0.5 * qdot.transpose() * M * qdot;
        // double U = potential_energy(model, data, q_history.back());
        // total_energy = T + U;
        // energy_history.push_back(total_energy);
        // delta_energy_history.push_back(total_energy - energy_history.front());
        //
        // Vec p = M * qdot;
        // momentum_history.push_back(p);

        T = kinetic_energy(model, data, (q_history.back() + q_history[q_history.size()-2]) / 2.0, (q_history.back() - q_history[q_history.size()-2])/timestep);
        U = potential_energy(model, data, (q_history.back() + q_history[q_history.size()-2]) / 2.0);
        total_energy = T+U;
        energy_history.push_back(total_energy);
        delta_energy_history.push_back(energy_history.back() - energy_history.front());

        auto t1 = high_resolution_clock::now();
        double elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count();
        runtimes.push_back(elapsed);

        // progress print
        double progress = t_cur / duration;
        if (progress > 1.0) progress = 1.0;
        int bar_width = 50;
        int pos = int(bar_width * progress);
        double avg_time = !runtimes.empty() ? std::accumulate(runtimes.begin(), runtimes.end(), 0.0) / runtimes.size() : 0.0;
        double time_left = avg_time * (duration - t_cur) / std::max(h_min, h_next);
        std::cout << "\r[";
        for (int i = 0; i < bar_width; ++i) {
            if (i < pos) std::cout << "=";
            else if (i == pos) std::cout << ">";
            else std::cout << " ";
        }
        std::cout << "] " << int(progress * 100.0) << "%, approx " << int(time_left/60) << "mins " << int(time_left) % 60 << "s left... ";
        std::cout.flush();
    }
    std::cout << "\n";

    auto t_end = high_resolution_clock::now();
    double total_elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_start).count();
    double avg_time = !runtimes.empty() ? std::accumulate(runtimes.begin(), runtimes.end(), 0.0) / runtimes.size() : 0.0;
    RCLCPP_INFO(node->get_logger(), "Simulation finished, wall time: %f s, Average step time: %f ms", total_elapsed, avg_time*1e3);

    // Save CSVs
    write_csv("src/ctsvi/csv/atsvi/q_history.csv", q_history);
    write_csv_scalar_series("src/ctsvi/csv/atsvi/time_history.csv", time_history);
    write_csv_scalar_series("src/ctsvi/csv/atsvi/energy_history.csv", energy_history);
    write_csv_scalar_series("src/ctsvi/csv/atsvi/h_history.csv", h_history);
    write_csv_scalar_series("src/ctsvi/csv/atsvi/delta_energy_history.csv", delta_energy_history);
    // write_csv("src/ctsvi/csv/atsvi/momentum_history.csv", momentum_history);

    RCLCPP_INFO(node->get_logger(), "Saved csv");

    // rclcpp::shutdown();
    return 0;
}
