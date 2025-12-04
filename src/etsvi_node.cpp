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

// Fix Eigen <-> CppAD compatibility for isfinite()
namespace Eigen {
    namespace numext {
        template<>
        EIGEN_STRONG_INLINE bool isfinite(const CppAD::AD<double>&) {
            return true;
        }
    }
}

using namespace pinocchio;
using namespace std::chrono;

// ---------- type aliases ----------
using ADScalar = CppAD::AD<double>;
using VecAD = Eigen::Matrix<ADScalar, Eigen::Dynamic, 1>;
using MatAD = Eigen::Matrix<ADScalar, Eigen::Dynamic, Eigen::Dynamic>;
using Vec  = Eigen::VectorXd;
using Mat  = Eigen::MatrixXd;
using ModelAD = pinocchio::ModelTpl<ADScalar>;
using DataAD  = pinocchio::DataTpl<ADScalar>;

// ---------- Simple PID controller for step-size ----------
// Controls h using energy error (E_next - E_prev). Output is additive delta_h.
struct PIDController {
    double Kp{0.0}, Ki{0.0}, Kd{0.0};
    double integral{0.0};
    double last_error{0.0};
    double integral_min{-1e100}, integral_max{1e100}; // anti-windup clamps

    PIDController() = default;
    PIDController(double p, double i, double d, double i_min=-1e100, double i_max=1e100)
      : Kp(p), Ki(i), Kd(d), integral_min(i_min), integral_max(i_max) {}

    // dt: physical time corresponding to the step-size used for derivative/integral scaling.
    double update(double error, double dt) {
        // handle dt near zero
        double deriv = (dt > 0.0) ? (error - last_error) / dt : 0.0;
        integral += error * dt;
        // clamp integral to prevent wind-up
        if (integral > integral_max) integral = integral_max;
        if (integral < integral_min) integral = integral_min;
        last_error = error;
        return Kp * error + Ki * integral + Kd * deriv;
    }

    void reset() { integral = 0.0; last_error = 0.0; }
};

// 计算末端位姿（位置 + 旋转矩阵）
std::pair<Eigen::Vector3d, Eigen::Matrix3d>
compute_end_effector_pose(const Model &model,
                          Data &data,
                          int frame_id,
                          const Vec &q)
{
    // Update joint FK
    pinocchio::forwardKinematics(model, data, q);

    // Update placements of all joints and frames
    // pinocchio::updateGlobalPlacements(model, data);
    pinocchio::updateFramePlacements(model, data);

    Eigen::Vector3d ee_pos = data.oMf[frame_id].translation();
    Eigen::Matrix3d ee_rot = data.oMf[frame_id].rotation();

    return {ee_pos, ee_rot};
}

// ---------- double helpers (non-AD) ----------
Mat inertia_matrix(const Model &model, Data &data, const Vec &q)
{
    pinocchio::crba(model, data, q);
    return data.M;
}

double kinetic_energy(const Model &model, Data &data, const Vec &q_mid, const Vec &qdot)
{
    pinocchio::crba(model, data, q_mid);
    Mat M = data.M;
    return 0.5 * qdot.transpose() * M * qdot;
}

double potential_energy(const Model &model, Data &data, const Vec &q)
{
    return pinocchio::computePotentialEnergy(model, data, q);
}

// double discrete Lagrangian (used for Ed numeric)
double discrete_lagrangian_double(const Model &model, Data &data, const Vec &q0, const Vec &q1, double h)
{
    Vec q_mid = 0.5 * (q0 + q1);
    Vec dq = (q1 - q0) / h;
    double T = kinetic_energy(model, data, q_mid, dq);
    double U = potential_energy(model, data, q_mid);
    return h * (T - U);
}

// numeric discrete energy Ed = -dL/dh (centered difference)
double discrete_energy_numeric(const Model &model, Data &data, const Vec &q0, const Vec &q1, double h, double eps_h = 1e-8)
{
    double Lp = discrete_lagrangian_double(model, data, q0, q1, h + eps_h);
    double Lm = discrete_lagrangian_double(model, data, q0, q1, h - eps_h);
    double dLdh = (Lp - Lm) / (2.0 * eps_h);
    return -dLdh;
}

// ---------- AD discrete Lagrangian ----------
ADScalar discreteLagrangian_ad(const ModelAD &model_ad, DataAD &data_ad,
                               const VecAD &q0_ad, const VecAD &q1_ad, double h)
{
    VecAD q_mid = ADScalar(0.5) * (q0_ad + q1_ad);
    VecAD dq = (q1_ad - q0_ad) / ADScalar(h);

    pinocchio::crba(model_ad, data_ad, q_mid); // fills data_ad.M (AD)
    MatAD M = data_ad.M;

    ADScalar T = ADScalar(0.5) * dq.transpose() * (M * dq);
    ADScalar U = pinocchio::computePotentialEnergy(model_ad, data_ad, q_mid);

    return ADScalar(h) * (T - U);
}

// ---------- AD gradients D1, D2 (returns double vector evaluated at q) ----------
Vec D_1(const ModelAD &model_ad, DataAD &data_ad, const Vec &q0, const Vec &q1, double h)
{
    const int n = q0.size();
    // independent variables: q0
    CppAD::vector<ADScalar> x_ad(n);
    for (int i=0;i<n;++i) x_ad[i] = q0[i];
    CppAD::Independent(x_ad);

    VecAD q0_ad = Eigen::Map<VecAD>(x_ad.data(), n);
    VecAD q1_ad = q1.cast<ADScalar>();

    ADScalar Ld = discreteLagrangian_ad(model_ad, data_ad, q0_ad, q1_ad, h);
    CppAD::vector<ADScalar> y(1);
    y[0] = Ld;

    CppAD::ADFun<double> f;
    f.Dependent(x_ad, y);

    CppAD::vector<double> x0(n);
    for (int i=0;i<n;++i) x0[i] = q0[i];

    CppAD::vector<double> jac = f.Jacobian(x0); // y dim 1 -> gradient length n

    Vec grad(n);
    for (int i=0;i<n;++i) grad[i] = jac[i];
    return grad;
}

Vec D_2(const ModelAD &model_ad, DataAD &data_ad, const Vec &q0, const Vec &q1, double h)
{
    const int n = q1.size();
    // independent variables: q1
    CppAD::vector<ADScalar> x_ad(n);
    for (int i=0;i<n;++i) x_ad[i] = q1[i];
    CppAD::Independent(x_ad);

    VecAD q0_ad = q0.cast<ADScalar>();
    VecAD q1_ad = Eigen::Map<VecAD>(x_ad.data(), n);

    ADScalar Ld = discreteLagrangian_ad(model_ad, data_ad, q0_ad, q1_ad, h);
    CppAD::vector<ADScalar> y(1);
    y[0] = Ld;

    CppAD::ADFun<double> f;
    f.Dependent(x_ad, y);

    CppAD::vector<double> x0(n);
    for (int i=0;i<n;++i) x0[i] = q1[i];

    CppAD::vector<double> jac = f.Jacobian(x0);

    Vec grad(n);
    for (int i=0;i<n;++i) grad[i] = jac[i];
    return grad;
}

// ---------- Solver info ----------
struct SolverInfo { bool converged; std::string reason; int iterations; double residual_norm; };

// ---------- Fixed-step DEL solver (Newton on q_next) ----------
std::pair<Vec, SolverInfo> solve_q_next(const ModelAD &model_ad,
                                        DataAD &data_ad,
                                        const Vec &q_prev,
                                        const Vec &q_curr,
                                        const Vec &tau_k,
                                        double h,
                                        int max_iters=50,
                                        double tol=1e-8)
{
    int n = q_curr.size();
    Vec q_next = q_curr;
    SolverInfo info{false, "", 0, 0.0};

    for (int it = 0; it < max_iters; ++it)
    {
        Vec D2 = D_2(model_ad, data_ad, q_prev, q_curr, h);
        Vec D1 = D_1(model_ad, data_ad, q_curr, q_next, h);
        Vec R = D2 + D1 - h * tau_k;
        double normR = R.norm();
        info.iterations = it;
        info.residual_norm = normR;
        if (normR < tol) { info.converged = true; return {q_next, info}; }

        // numeric Jacobian of D1 wrt q_next (could be replaced by AD Jacobian)
        Mat J = Mat::Zero(n, n);
        double eps = 1e-6;
        for (int j = 0; j < n; ++j)
        {
            Vec dq = Vec::Zero(n);
            dq(j) = eps;
            Vec D1p = D_1(model_ad, data_ad, q_curr, q_next + dq, h);
            Vec D1m = D_1(model_ad, data_ad, q_curr, q_next - dq, h);
            J.col(j) = (D1p - D1m) / (2.0 * eps);
        }

        Mat A = J + 1e-9 * Mat::Identity(n, n);
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

// ---------- Energy-tracking time steps solver ----------
std::tuple<Vec, double, SolverInfo> solve_q_next_et(const Model &model, Data &data,
                                                    const ModelAD &model_ad, DataAD &data_ad,
                                                    const Vec &q_prev, const Vec &q_curr,
                                                    double h_prev,
                                                    const Vec &tau_k,
                                                    PIDController &pid)
{
    double E_d = discrete_energy_numeric(model, data, q_prev, q_curr, h_prev);

    auto [q_next, info] = solve_q_next(model_ad, data_ad, q_prev,
                                           q_curr, tau_k, h_prev);

    double E_r = discrete_energy_numeric(model, data, q_curr, q_next, h_prev);

    // define normalized error (scale invariant)
    double scale = std::max(std::abs(E_r), std::abs(E_d));
    double norm = (scale < 1e-12) ? 1.0 : scale;
    double E_err = (E_d - E_r) / norm; // want ~0

    double delta_h = pid.update(E_err, h_prev);
    double h_next = h_prev + delta_h;

    return {q_next, h_next, info};
}

// CSV writers
void write_csv(const std::string &filename, const std::vector<Vec> &rows)
{
    if (rows.empty()) return;
    std::ofstream ofs(filename);
    int ncols = rows[0].size();
    for (size_t r=0;r<rows.size();++r)
    {
        for (int c=0;c<ncols;++c)
        {
            ofs << std::setprecision(15) << rows[r](c);
            if (c+1 < ncols) ofs << ",";
        }
        ofs << "\n";
    }
}

void write_csv_scalar_series(const std::string &filename, const std::vector<double> &rows)
{
    std::ofstream ofs(filename);
    for (double v : rows) ofs << std::setprecision(15) << v << "\n";
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

void print_progress(double t_cur, double duration, std::vector<double> runtimes, double h_next) {
    int bar_width = 50;
    double progress = t_cur / duration;
    if (progress > 1.0) progress = 1.0;
    int pos = int(bar_width * progress);
    double avg_time = !runtimes.empty() ? std::accumulate(runtimes.begin(), runtimes.end(), 0.0) / runtimes.size() : 0.0;
    double time_left = avg_time * (duration - t_cur) / h_next;
    std::cout << "\r[";
    for (int i = 0; i < bar_width; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << "%, approx " << int(time_left/60) << "mins " << int(time_left) % 60 << "s left... ";
    std::cout.flush();
}

// ---------- Main ----------
int main(int argc, char** argv)
{
    std::cout.setf(std::ios::unitbuf);
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>(
        "etsvi_node",
        rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true)
    );

    // Parameters (provide defaults via launch)
    double q_init = node->get_parameter("q_init").as_double();
    double timestep = node->get_parameter("timestep").as_double();
    double duration = node->get_parameter("duration").as_double();
    double eps_diff = node->get_parameter("eps_diff").as_double();
    std::string urdf_path = node->get_parameter("urdf_path").as_string();
    // double h_min = node->get_parameter("h_min").as_double();
    // double h_max = node->get_parameter("h_max").as_double();
    // int max_adapt_iters = node->get_parameter("max_adapt_iters").as_int();

    // PID params for energy control
    double pid_Kp = node->get_parameter("pid_Kp").as_double();
    double pid_Ki = node->get_parameter("pid_Ki").as_double();
    double pid_Kd = node->get_parameter("pid_Kd").as_double();

    RCLCPP_INFO(node->get_logger(), "Using URDF: %s", urdf_path.c_str());
    RCLCPP_INFO(node->get_logger(), "Duration = %.3f, Timestep(init) = %.6f, eps_diff = %.1e", duration, timestep, eps_diff);
    // RCLCPP_INFO(node->get_logger(), "h_min=%.1e h_max=%.6f max_adapt_iters=%d", h_min, h_max, max_adapt_iters);

    // load model (double) and data
    Model model;
    try {
        pinocchio::urdf::buildModel(urdf_path, model);
    } catch (const std::exception &e) {
        RCLCPP_ERROR(node->get_logger(), "Error loading URDF: %s", e.what());
        return 1;
    }
    Data data(model);
    model.gravity.linear(Eigen::Vector3d(0,0,-9.81));

    // 获取末端 frame id
    int link_tcp_id = model.getFrameId("link_tcp");
    // RCLCPP_INFO(node->get_logger(), "link_tcp_id=%d:", link_tcp_id);
    if (link_tcp_id == -1) {
        RCLCPP_ERROR(node->get_logger(), "TCP not found!");
        return 1;
    }

    // RCLCPP_INFO(node->get_logger(), "Model has nq=%d nv=%d", model.nq, model.nv);

    // build AD model/data
    ModelAD model_ad = model.cast<ADScalar>();
    DataAD data_ad(model_ad);

    int n = model.nq;
    int n_steps = static_cast<int>(duration / timestep);

    // initial conditions
    Vec q_prev = Vec::Constant(n, q_init);
    Vec v_prev = Vec::Zero(n);
    Vec tau_k = Vec::Zero(n); // currently zero torques; modify as needed

    std::vector<Vec> q_history;
    std::vector<double> time_history;
    std::vector<double> energy_history;
    std::vector<double> delta_energy_history;
    std::vector<double> h_history;
    std::vector<double> runtimes;
    std::vector<Eigen::Vector3d> ee_history;

    time_history.push_back(0.0);
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

    auto [ee_pos0, ee_rot0] = compute_end_effector_pose(model, data, link_tcp_id, q_prev);
    ee_history.push_back(ee_pos0);

    // initial VI step
    auto [q_curr, info_init] = solve_q_next(model_ad, data_ad, q_prev, q_prev + v_prev * timestep, tau_k, timestep);
    q_history.push_back(q_curr);
    
    double T = kinetic_energy(model, data, (q_curr + q_prev) / 2.0, (q_curr - q_prev)/timestep);
    double U = potential_energy(model, data, (q_curr + q_prev) / 2.0);
    total_energy = T + U;
    energy_history.push_back(total_energy);
    delta_energy_history.push_back(energy_history.back() - energy_history.front());

    auto [ee_pos1, _ee_rot1] = compute_end_effector_pose(model, data, link_tcp_id, q_curr);
    ee_history.push_back(ee_pos1);

    double t_cur = 0.0;
    t_cur += timestep; // we have advanced to q_curr at t = timestep
    time_history.push_back(timestep);
    h_history.push_back(timestep);

    PIDController pid(pid_Kp, pid_Ki, pid_Kd, -1e6, 1e6);

    // int max_steps = std::max((int)(duration / h_min) + 10, 1000);
    for (int step=0; step < n_steps -1  && t_cur < duration - 1e-12; ++step)
    {
        auto t0 = high_resolution_clock::now();

        Vec qdot_guess = (q_history.back() - q_history[q_history.size()-2]) / h_history.back();

        auto [q_next, h_next, info_adapt] = solve_q_next_et(
            model, data, model_ad, data_ad,
            q_history[q_history.size()-2],
            q_history[q_history.size()-1],
            h_history.back(),
            tau_k,
            pid
        );

        if (!info_adapt.converged)
        {
            RCLCPP_WARN(node->get_logger(), "Step %d: SEM solver failed (%s). Falling back to fixed-step DEL.", step, info_adapt.reason.c_str());
            auto [q_fix, info_fix] = solve_q_next(model_ad, data_ad, q_history[q_history.size()-2], q_history[q_history.size()-1], tau_k, h_history.back(), eps_diff);
            q_next = q_fix;
            h_next = h_history.back();
            RCLCPP_INFO(node->get_logger(), "Fixed-step info: converged=%d it=%d res=%f reason=%s", info_fix.converged, info_fix.iterations, info_fix.residual_norm, info_fix.reason.c_str());
        }

        q_history.push_back(q_next);
        h_history.push_back(h_next);
        t_cur += h_next;
        time_history.push_back(t_cur);

        T = kinetic_energy(model, data, (q_history.back() + q_history[q_history.size()-2]) / 2.0, (q_history.back() - q_history[q_history.size()-2]) / h_next);
        U = potential_energy(model, data, (q_history.back() + q_history[q_history.size()-2]) / 2.0);
        total_energy = T + U;
        energy_history.push_back(total_energy);
        delta_energy_history.push_back(energy_history.back() - energy_history.front());

        auto [ee_pos, ee_rot] = compute_end_effector_pose(model, data, link_tcp_id, q_next);
        ee_history.push_back(ee_pos);

        auto t1 = high_resolution_clock::now();
        double elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count();
        runtimes.push_back(elapsed);

        print_progress(t_cur, duration, runtimes, h_next);
    }
    std::cout << "\n";

    auto t_end = high_resolution_clock::now();
    double total_elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_start).count();
    double avg_time = !runtimes.empty() ? std::accumulate(runtimes.begin(), runtimes.end(), 0.0) / runtimes.size() : 0.0;
    RCLCPP_INFO(node->get_logger(), "Simulation finished, wall time: %f s, Average step time: %f ms", total_elapsed, avg_time*1e3);

    // Save CSVs
    write_csv("src/vi/csv/etsvi/q_history.csv", q_history);
    write_csv_scalar_series("src/vi/csv/etsvi/time_history.csv", time_history);
    write_csv_scalar_series("src/vi/csv/etsvi/energy_history.csv", energy_history);
    write_csv_scalar_series("src/vi/csv/etsvi/h_history.csv", h_history);
    write_csv_scalar_series("src/vi/csv/etsvi/delta_energy_history.csv", delta_energy_history);
    write_csv_3d("src/vi/csv/etsvi/ee_history.csv", ee_history);

    RCLCPP_INFO(node->get_logger(), "Saved CSVs.");

    return 0;
}
