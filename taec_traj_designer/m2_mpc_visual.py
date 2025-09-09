import matplotlib.pyplot as plt
import numpy as np
import math
import cvxpy

# -------------------- 车辆与 MPC 参数 --------------------
NX = 4  # 状态变量数量: [x, y, v, yaw]
NU = 2  # 控制变量数量: [a, delta]
T = 5  # 预测时域长度

# MPC 权重矩阵
R = np.diag([0.01, 0.01])  # 控制输入代价
Rd = np.diag([0.01, 1.0])  # 控制输入变化代价
Q = np.diag([1.0, 1.0, 0.5, 0.5])  # 状态误差代价
Qf = Q  # 最终状态误差代价

# 车辆参数
WB = 2.5  # 轴距 [m]
MAX_STEER = np.deg2rad(45.0)  # 最大转向角 [rad]
MAX_DSTEER = np.deg2rad(30.0)  # 最大转向角速度 [rad/s]
MAX_SPEED = 20.0 / 3.6  # 最大速度 [m/s]
MIN_SPEED = -5.0 / 3.6  # 最小速度 [m/s]
MAX_ACCEL = 1.0  # 最大加速度 [m/s^2]

DT = 0.2  # 采样时间 [s]
MAX_ITER = 3  # 最大迭代次数
DU_TH = 0.1  # 控制输入变化阈值
GOAL_DIS = 1.5  # 目标距离 [m]

show_animation = True


# -------------------- 车辆状态类 --------------------
class State:
    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v

    def update(self, a, delta):
        """
        更新车辆状态
        """
        delta = np.clip(delta, -MAX_STEER, MAX_STEER)  # 限制转向角
        self.x += self.v * math.cos(self.yaw) * DT
        self.y += self.v * math.sin(self.yaw) * DT
        self.yaw += self.v / WB * math.tan(delta) * DT
        self.v += a * DT
        self.v = np.clip(self.v, MIN_SPEED, MAX_SPEED)  # 限制速度


# -------------------- 辅助函数 --------------------
def pi_2_pi(angle):
    """
    将角度限制在 [-pi, pi] 之间
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi


def calc_nearest_index(state, cx, cy):
    """
    计算车辆状态与轨迹的最近点索引
    """
    dx = [state.x - icx for icx in cx]
    dy = [state.y - icy for icy in cy]
    distances = np.hypot(dx, dy)
    index = np.argmin(distances)
    return index, distances[index]


def get_linear_model_matrix(v, phi, delta):
    """
    获取线性化的车辆动力学模型矩阵
    """
    A = np.eye(NX)
    A[0, 2] = DT * math.cos(phi)
    A[0, 3] = -DT * v * math.sin(phi)
    A[1, 2] = DT * math.sin(phi)
    A[1, 3] = DT * v * math.cos(phi)
    A[3, 2] = DT * math.tan(delta) / WB

    B = np.zeros((NX, NU))
    B[2, 0] = DT
    B[3, 1] = DT * v / (WB * math.cos(delta) ** 2)

    C = np.zeros(NX)
    return A, B, C


def predict_motion(x0, oa, od):
    """
    预测车辆的状态序列
    """
    state = State(x=x0[0], y=x0[1], yaw=x0[3], v=x0[2])
    xbar = np.zeros((NX, T + 1))
    xbar[:, 0] = x0
    for i in range(T):
        state.update(oa[i], od[i])
        xbar[:, i + 1] = [state.x, state.y, state.v, state.yaw]
    return xbar


# -------------------- 核心 MPC 控制器 --------------------
def linear_mpc_control(xref, xbar, x0, oa, od):
    """
    线性化的 MPC 控制器
    """
    x = cvxpy.Variable((NX, T + 1))
    u = cvxpy.Variable((NU, T))

    cost = 0.0
    constraints = []

    # 初始状态约束
    constraints += [x[:, 0] == x0]

    for t in range(T):
        cost += cvxpy.quad_form(u[:, t], R)  # 控制输入代价
        cost += cvxpy.quad_form(xref[:, t] - x[:, t], Q)  # 状态误差代价

        if t != 0:
            cost += cvxpy.quad_form(u[:, t] - u[:, t - 1], Rd)  # 控制输入变化代价

        # 动力学约束
        A, B, C = get_linear_model_matrix(xbar[2, t], xbar[3, t], od[t])
        constraints += [x[:, t + 1] == A @ x[:, t] + B @ u[:, t] + C]

        # 控制输入限制
        constraints += [cvxpy.abs(u[0, t]) <= MAX_ACCEL]
        constraints += [cvxpy.abs(u[1, t]) <= MAX_STEER]

    cost += cvxpy.quad_form(xref[:, T] - x[:, T], Qf)  # 终端状态误差代价

    # 求解优化问题
    prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
    prob.solve(solver=cvxpy.OSQP)

    if prob.status == cvxpy.OPTIMAL or prob.status == cvxpy.OPTIMAL_INACCURATE:
        ox = get_nparray_from_matrix(x.value[0, :])
        oy = get_nparray_from_matrix(x.value[1, :])
        ov = get_nparray_from_matrix(x.value[2, :])
        oyaw = get_nparray_from_matrix(x.value[3, :])
        oa = get_nparray_from_matrix(u.value[0, :])
        odelta = get_nparray_from_matrix(u.value[1, :])
    else:
        print("MPC求解失败")
        oa, odelta, ox, oy, oyaw, ov = None, None, None, None, None, None

    return oa, odelta, ox, oy, oyaw, ov


def iterative_linear_mpc_control(xref, x0, oa, od):
    """
    迭代线性化 MPC 控制器
    """
    for _ in range(MAX_ITER):
        xbar = predict_motion(x0, oa, od)
        poa, pod = oa[:], od[:]
        oa, od, _, _, _, _ = linear_mpc_control(xref, xbar, x0, oa, od)
        du = np.sum(np.abs(oa - poa)) + np.sum(np.abs(od - pod))
        if du <= DU_TH:
            break
    return oa, od


# -------------------- 模拟主函数 --------------------
def do_simulation(cx, cy, cyaw, initial_state):
    """
    MPC 仿真主函数
    """
    state = initial_state
    x, y, yaw, v = [state.x], [state.y], [state.yaw], [state.v]
    oa, od = [0.0] * T, [0.0] * T

    for _ in range(500):
        target_ind, _ = calc_nearest_index(state, cx, cy)
        xref = np.zeros((NX, T + 1))
        for t in range(T + 1):
            ind = min(len(cx) - 1, target_ind + t)
            xref[:, t] = [cx[ind], cy[ind], MAX_SPEED, cyaw[ind]]

        oa, od = iterative_linear_mpc_control(xref, [state.x, state.y, state.v, state.yaw], oa, od)
        state.update(oa[0], od[0])

        x.append(state.x)
        y.append(state.y)
        yaw.append(state.yaw)
        v.append(state.v)

        if show_animation:
            plt.cla()
            plt.plot(cx, cy, "-r", label="Reference")
            plt.plot(x, y, "-b", label="MPC")
            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.001)

    return x, y, yaw, v


# -------------------- 主程序入口 --------------------
if __name__ == "__main__":
    cx = np.linspace(0, 50, 500)
    cy = np.sin(cx / 5.0) * cx / 5.0
    cyaw = np.arctan2(np.gradient(cy), np.gradient(cx))
    initial_state = State(x=cx[0], y=cy[0], yaw=cyaw[0], v=0.0)

    x, y, yaw, v = do_simulation(cx, cy, cyaw, initial_state)

    plt.figure()
    plt.plot(cx, cy, "-r", label="Reference Path")
    plt.plot(x, y, "-b", label="Tracked Path")
    plt.legend()
    plt.grid(True)
    plt.show()