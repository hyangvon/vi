#!/usr/bin/env python3
import os
import yaml
import numpy as np
import pybullet as p
import pybullet_data

# Simple PyBullet simulator that reproduces dynamics and writes CSVs

def expand_user(path):
    return os.path.expanduser(path)


def main():
    cfg_path = expand_user('~/ros2_ws/dynamic_ws/src/vi/config/vi_params.yaml')
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # params under /**/ros__parameters
    params = cfg.get('/**', {}).get('ros__parameters', {})
    # fallback to top-level keys if not found
    q_init = params.get('q_init', 0.2)
    timestep = params.get('timestep', 0.01)
    duration = params.get('duration', 10.0)
    urdf_path = expand_user(params.get('urdf_path', '~/ros2_ws/dynamic_ws/src/vi/urdf/7_pendulum.urdf'))

    n_steps = int(duration / timestep)

    # Start PyBullet in DIRECT
    client = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation(physicsClientId=client)
    p.setGravity(0, 0, -9.81, physicsClientId=client)
    p.setTimeStep(timestep, physicsClientId=client)

    flags = p.URDF_USE_INERTIA_FROM_FILE
    body = p.loadURDF(urdf_path, useFixedBase=True, flags=flags, physicsClientId=client)

    num_joints = p.getNumJoints(body, physicsClientId=client)

    # find link index for 'link_tcp'
    link_tcp_idx = -1
    for i in range(num_joints):
        info = p.getJointInfo(body, i, physicsClientId=client)
        link_name = info[12].decode('utf-8')
        if link_name == 'link_tcp':
            link_tcp_idx = i
            break

    # initialize joints
    for i in range(num_joints):
        p.resetJointState(body, i, targetValue=q_init, targetVelocity=0.0, physicsClientId=client)
        p.setJointMotorControl2(body, i, p.VELOCITY_CONTROL, force=0)

    q_history = []
    time_history = []
    energy_history = []
    delta_energy_history = []
    ee_history = []

    def compute_ke(qdot_arr, M):
        """Compute kinetic energy robustly when M and qdot sizes may differ."""
        qdot_arr = np.asarray(qdot_arr).flatten()
        M = np.asarray(M)
        n_q = qdot_arr.size

        if M.ndim == 1:
            # treat as diagonal
            M_mat = np.diag(M)
        else:
            M_mat = M

        m0, m1 = M_mat.shape[0], M_mat.shape[1]
        # take top-left submatrix if M larger, or pad with zeros if smaller
        if m0 >= n_q and m1 >= n_q:
            M_sub = M_mat[:n_q, :n_q]
        else:
            M_sub = np.zeros((n_q, n_q))
            r = min(m0, n_q)
            c = min(m1, n_q)
            M_sub[:r, :c] = M_mat[:r, :c]

        return 0.5 * qdot_arr @ M_sub @ qdot_arr

    # initial energy
    q = [p.getJointState(body, i, physicsClientId=client)[0] for i in range(num_joints)]
    qdot = [p.getJointState(body, i, physicsClientId=client)[1] for i in range(num_joints)]

    try:
        M = np.array(p.calculateMassMatrix(body, q, physicsClientId=client))
    except Exception:
        M = np.eye(num_joints)

    qdot_arr = np.array(qdot)
    KE = compute_ke(qdot_arr, M)

    # potential energy: sum m * g * z (approx using link COM/world position)
    U = 0.0
    g = 9.81
    # base
    base_mass, *_ = p.getDynamicsInfo(body, -1, physicsClientId=client) if False else (0.0,)
    for i in range(-1, num_joints):
        if i == -1:
            pos, _ = p.getBasePositionAndOrientation(body, physicsClientId=client)
            mass = 0.0
        else:
            mass = p.getDynamicsInfo(body, i, physicsClientId=client)[0]
            pos = p.getLinkState(body, i, physicsClientId=client)[0]
        U += mass * g * pos[2]

    E_ref = KE + U

    # main loop
    t = 0.0
    for step in range(n_steps):
        p.stepSimulation(physicsClientId=client)

        q = [p.getJointState(body, i, physicsClientId=client)[0] for i in range(num_joints)]
        qdot = [p.getJointState(body, i, physicsClientId=client)[1] for i in range(num_joints)]

        try:
            M = np.array(p.calculateMassMatrix(body, q, physicsClientId=client))
        except Exception:
            M = np.eye(num_joints)

        qdot_arr = np.array(qdot)
        KE = compute_ke(qdot_arr, M)

        U = 0.0
        for i in range(-1, num_joints):
            if i == -1:
                # base
                continue
            mass = p.getDynamicsInfo(body, i, physicsClientId=client)[0]
            pos = p.getLinkState(body, i, physicsClientId=client)[0]
            U += mass * g * pos[2]

        E = KE + U

        # ee position
        if link_tcp_idx != -1:
            ee_pos = p.getLinkState(body, link_tcp_idx, physicsClientId=client)[0]
        else:
            ee_pos = (0.0, 0.0, 0.0)

        q_history.append(np.array(q))
        time_history.append(t)
        energy_history.append(E)
        delta_energy_history.append(E - E_ref)
        ee_history.append(np.array(ee_pos))

        t += timestep

    # save CSVs
    csv_dir = os.path.expanduser('~/ros2_ws/dynamic_ws/src/vi/csv/pybullet/')
    os.makedirs(csv_dir, exist_ok=True)

    np.savetxt(os.path.join(csv_dir, 'q_history.csv'), np.array(q_history), delimiter=',')
    np.savetxt(os.path.join(csv_dir, 'time_history.csv'), np.array(time_history), delimiter=',')
    np.savetxt(os.path.join(csv_dir, 'energy_history.csv'), np.array(energy_history), delimiter=',')
    np.savetxt(os.path.join(csv_dir, 'delta_energy_history.csv'), np.array(delta_energy_history), delimiter=',')
    np.savetxt(os.path.join(csv_dir, 'ee_history.csv'), np.array(ee_history), delimiter=',')

    p.disconnect(client)

if __name__ == '__main__':
    main()
