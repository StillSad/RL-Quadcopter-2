import numpy as np
from physics_sim import PhysicsSim

class TakeOff():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None,
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
            起点（0，0，0）
            终点 高度大于20
            路径 接近垂直上升

        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime)
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4
        self.last_post = self.sim.pose

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.])

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        #1、计算当前位置与目标位置差值的绝对值
        temp = abs(self.sim.pose[:3] - self.target_pos)
        #2、(self.sim.pose[2] - self.target_pos[2]) 当前位置与目标位置高度的差值作为奖励的第一部分，
        # 当前位置低于目标位置奖励为负值，当前位置高于目标位置奖励为正值；
        # 奖励的第二部分是当前位置与目标位置xy轴上的偏差，偏差越大奖励越小，防止上升时xy轴上有较大偏移
        reward = (self.sim.pose[2] - self.target_pos[2]) - 0.5 * temp[:2].sum()
        # 3、奖励的第三部分是当起飞高度大于目标位置高度时奖励增加20，小于目标位置高度时奖励减少30；可以让飞行器更快的达到起飞高度
        if self.sim.pose[2] >= self.target_pos[2]:  # agent has crossed the target height
            reward += 20.0  # bonus reward
        else:
            reward -= 30
        #4、奖励的第四部分是当前位置高度与上次位置高度的差值，若当前位置高度低于上次位置高度说明飞行器在下降奖励减少，
        # 当前位置高度等于上次位置高度奖励不变，当前位置高度高于上次位置高度说明飞行器在上升，奖励增加
        temp = self.sim.pose - self.last_post
        reward += temp[2]


        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward()
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat)
        return state