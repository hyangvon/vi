编译、仿真、画图
colcon build && ros2 run vi vi_compare_sim.py  && ros2 run vi rk4_vi_compare_sim.py

编译、画图（跳过仿真）
colcon build && ros2 run vi vi_compare_sim.p --skip-sim && ros2 run vi rk4_vi_compare_sim.py --skip-sim
