import numpy as np
import matplotlib.pyplot as plt

# 已知参数
curvature = 0.5  # 曲率
cur_list = []
zt = curvature
for i in range(15):
    cur = curvature - i * 0.015
    cur_list.append(cur)
    zt = cur
for i in range(15):
    cur = zt + i * 0.015
    cur_list.append(cur)

radius = 1 / curvature  # 半径
arc_length = 3.0  # 轨迹长度（假设运动总时间为3s，匀速运动）
center = (radius, 0)  # 圆心 (假设圆心在 (R, 0))

def generate_arc_points(curvature, v):
    start_point = np.array([0,0,0])
    sx = start_point[0]
    sy = start_point[1]
    stheta = start_point[2]
    pt_list = []
    start_point = np.array([sx, sy, stheta])
    pt_list.append(start_point)
    for i in range(30):
        sx = sx + v * np.cos(stheta) * 0.1
        sy = sy + v * np.sin(stheta) * 0.1
        stheta = stheta + curvature[i] * v * 0.1
        cur_point = np.array([sx, sy, stheta])
        pt_list.append(cur_point)
    pt_list = np.array(pt_list)
    return pt_list 

# def generate_arc_points(curvature, v):
#     start_point = np.array([0,0,0])
#     sx = start_point[0]
#     sy = start_point[1]
#     stheta = start_point[2]
#     pt_list = []
#     start_point = np.array([sx, sy, stheta])
#     pt_list.append(start_point)
#     for i in range(30):
#         sx = sx + v * np.cos(stheta) * 0.1
#         sy = sy + v * np.sin(stheta) * 0.1
#         stheta = stheta + curvature * v * 0.1
#         cur_point = np.array([sx, sy, stheta])
#         pt_list.append(cur_point)
#     pt_list = np.array(pt_list)
#     return pt_list 

# # 计算圆弧上的点
# def generate_arc_points(radius, arc_length, speed, num_points):
#     total_time = arc_length / speed  # 总时间
#     time_points = np.linspace(0, total_time, num_points)  # 时间点
#     angles = time_points * speed / radius  # 对应圆弧上的角度 (s = r * theta -> theta = s / r)
#     x = radius * np.cos(angles)  # 圆弧上的 x 坐标
#     y = radius * np.sin(angles)  # 圆弧上的 y 坐标
#     return x, y, time_points

# 生成两种速度下的轨迹
speed_1 = 2.0  # 速度1（慢速）
speed_2 = 4.0  # 速度2（快速）
num_points = 30  # 每条轨迹采样30个点

traj1 = generate_arc_points(cur_list, speed_1)
traj2 = generate_arc_points(cur_list, speed_2)
x1 = traj1[:,0]
y1 = traj1[:,1]
x2 = traj2[:,0]
y2 = traj2[:,1]
# # 轨迹1（速度1）
# x1, y1, t1 = generate_arc_points(radius, arc_length, speed_1, num_points)
# # 轨迹2（速度2）
# x2, y2, t2 = generate_arc_points(radius, arc_length, speed_2, num_points)

# 绘制轨迹
plt.figure(figsize=(10, 6))
plt.plot(x1, y1, '-o', label=f"Speed = {speed_1} (slow)", markersize=5)
plt.plot(x2, y2, '-x', label=f"Speed = {speed_2} (fast)", markersize=5)

# 圆心和轨迹信息
plt.scatter(center[0], center[1], color='red', label="Arc Center", zorder=5)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Trajectories with Different Speeds (Constant Curvature = 0.5)")
plt.legend()
plt.axis("equal")
plt.grid()

# 显示图形
plt.show()