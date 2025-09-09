import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

def unicycle_model(state, v, omega, dt):
    """
    单轨车辆模型（Unicycle Model）
    状态更新方程:
        x_{k+1} = x_k + v_k * cos(theta_k) * dt
        y_{k+1} = y_k + v_k * sin(theta_k) * dt
        theta_{k+1} = theta_k + omega_k * dt

    参数:
        state: 当前状态 [x, y, theta]
        v: 线速度
        omega: 角速度
        dt: 时间步长

    返回:
        下一时刻状态 [x_next, y_next, theta_next]
    """
    x, y, theta = state
    x_next = x + v * np.cos(theta) * dt
    y_next = y + v * np.sin(theta) * dt
    theta_next = theta + omega * dt
    return np.array([x_next, y_next, theta_next])


def mpc_control(initial_state, trajectory, horizon, dt):
    """
    使用MPC控制器追踪轨迹。

    参数:
        initial_state: 初始状态 [x, y, theta]
        trajectory: 给定轨迹，形状为 (N, 5)，每个点包含 [x_ref, y_ref, theta_ref, v_ref, time]
        horizon: MPC预测步长
        dt: 时间步长

    返回:
        控制输入序列 [(v0, omega0), (v1, omega1), ...]
        以及相应的状态序列
    """
    num_points = len(trajectory)
    x_ref, y_ref, theta_ref = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]
    
    # 权重参数
    Q_x = 1.0  # x误差权重
    Q_y = 1.0  # y误差权重
    Q_theta = 0.5  # theta误差权重
    R_v = 0.1  # v控制输入权重
    R_omega = 0.1  # omega控制输入权重

    # 当前状态
    state = np.array(initial_state)
    states = [state]  # 保存所有状态
    controls = []  # 保存所有控制输入

    for t in range(num_points - horizon):
        # 当前预测的参考轨迹
        ref_x = x_ref[t:t + horizon]
        ref_y = y_ref[t:t + horizon]
        ref_theta = theta_ref[t:t + horizon]

        # 优化变量
        v = cp.Variable(horizon)  # 线速度
        omega = cp.Variable(horizon)  # 角速度
        x = cp.Variable(horizon + 1)
        y = cp.Variable(horizon + 1)
        theta = cp.Variable(horizon + 1)

        # 约束
        constraints = []
        constraints.append(x[0] == state[0])
        constraints.append(y[0] == state[1])
        constraints.append(theta[0] == state[2])

        for k in range(horizon):
            # 动力学约束
            constraints.append(x[k + 1] == x[k] + v[k] * cp.cos(theta[k]) * dt)
            constraints.append(y[k + 1] == y[k] + v[k] * cp.sin(theta[k]) * dt)
            constraints.append(theta[k + 1] == theta[k] + omega[k] * dt)
            # 控制输入约束
            constraints.append(v[k] >= -1.0)  # 最小速度
            constraints.append(v[k] <= 1.0)   # 最大速度
            constraints.append(omega[k] >= -1.0)  # 最小角速度
            constraints.append(omega[k] <= 1.0)   # 最大角速度

        # 目标函数
        cost = 0
        for k in range(horizon):
            cost += Q_x * (x[k] - ref_x[k]) ** 2 + Q_y * (y[k] - ref_y[k]) ** 2
            cost += Q_theta * (theta[k] - ref_theta[k]) ** 2
            cost += R_v * v[k] ** 2 + R_omega * omega[k] ** 2

        # 定义优化问题
        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve()

        # 当前时刻的最优控制输入
        v_opt = v.value[0]
        omega_opt = omega.value[0]
        controls.append((v_opt, omega_opt))

        # 更新状态
        state = unicycle_model(state, v_opt, omega_opt, dt)
        states.append(state)

    return np.array(controls), np.array(states)


def main():
    # 给定轨迹 (31 x 5)
    trajectory = np.array([
        [i, 2 * np.sin(0.1 * i), 0.1 * i, 1.0, i * 0.1] for i in range(31)
    ])
    initial_state = trajectory[0, :3]  # 初始状态 [x, y, theta]
    horizon = 10  # 预测步长
    dt = 0.1  # 时间间隔

    # 使用MPC追踪轨迹
    controls, states = mpc_control(initial_state, trajectory, horizon, dt)

    # 绘制轨迹
    plt.figure(figsize=(10, 6))
    plt.plot(trajectory[:, 0], trajectory[:, 1], "r--", label="Reference Trajectory")
    plt.plot(states[:, 0], states[:, 1], "b-", label="MPC Tracking")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend()
    plt.grid()
    plt.title("MPC Trajectory Tracking")
    plt.show()


if __name__ == "__main__":
    main()