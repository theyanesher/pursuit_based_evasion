import numpy as np
import torch
# import matplotlib.pyplot as plt
from copy import deepcopy

from env import Env
from agent import Agent
from test_parameter import *
from utils import *
from model import PolicyNet
from node_manager import Node_manager
from ground_truth_node_manager import GroundTruthNodeManager

if not os.path.exists(gifs_path):
    os.makedirs(gifs_path)

class Test_worker:
    def __init__(self, meta_agent_id, policy_net, global_step, n_agents, device='cpu', save_image=False, greedy=True):
        self.meta_agent_id = meta_agent_id
        self.global_step = global_step    # Same as episode number
        self.save_image = save_image
        self.device = device
        self.greedy = greedy
        self.save_path = gifs_path

        self.n_agents = n_agents

        self.env = Env(global_step, self.n_agents, plot=self.save_image, save_path=self.save_path)
        self.robot_height = self.env.robot_height    # metres

        self.node_manager = Node_manager(self.robot_height, self.env.robot_height_cell, self.env.max_height_cell, quad_tree_size=QUAD_TREE_SIZE, 
                                         sensor_range=self.env.sensor_range, plot=self.save_image)
        
        # self.ground_truth_node_manager = GroundTruthNodeManager(self.node_manager, self.env.ground_truth_info, self.env.sensor_range, self.robot_height, self.env.robot_height_cell, self.env.max_height_cell,
        #                                                 device=self.device, plot=self.save_image)
        
        self.ground_truth_node_manager = None
        
        self.env.init_episode(self.node_manager)

        self.robot_list = [Agent(i, policy_net, self.node_manager, self.robot_height, self.env.robot_height_cell, 
                                 self.env.max_height_cell, self.device, self.save_image) for i in
                           range(self.n_agents)]

        self.perf_metrics = dict()

    def run_episode(self):
        for i, robot in enumerate(self.robot_list):
            robot.update_graph(self.env.flight_info, self.env.ground_info, deepcopy(self.env.robot_locations[robot.id]), 
                               self.env.safe_info, self.env.global_frontiers, self.env.global_frontiers_ground, self.env.global_safe_zone_frontiers, 
                               self.env.uncovered_safe_frontiers, self.env.robot_locations, k=i)

        for robot in self.robot_list:
            robot.update_past_locations()

        all_robot_past_locations = []  
        for robot in self.robot_list:  
            robot.update_planning_state(self.env.robot_locations)
            all_robot_past_locations.append(robot.past_positions_index)

        if self.save_image:
            self.env.plot_env_test(-1, -1, self.robot_list)  

        max_travel_dist = 0
        length_history = [max_travel_dist]
        safe_rate_history = [self.env.safe_rate]
        explored_rate_history = [self.env.explored_rate]
        ground_explored_rate_history = [self.env.explored_rate_ground]

        for i in range(MAX_EPISODE_STEP):
            print('Timestep:', i)
            selected_locations = []
            dist_list = []
            next_node_index_list = []
            safe_increase_record = []

            for robot in self.robot_list:
                # Get states
                observation = robot.get_observation(copy(all_robot_past_locations), pad=False)

                next_location, next_node_index, _ = robot.select_next_waypoint(observation, self.greedy)

                selected_locations.append(next_location)
                dist_list.append(np.linalg.norm(next_location - robot.location))
                next_node_index_list.append(next_node_index)

            selected_locations = self.solve_path_confict(selected_locations, dist_list)

            self.env.step(i , selected_locations, self.robot_list, test=True)

            for k, robot in enumerate(self.robot_list):
                robot.update_graph(self.env.flight_info, self.env.ground_info, deepcopy(self.env.robot_locations[robot.id]), 
                               self.env.safe_info, self.env.global_frontiers, self.env.global_frontiers_ground, self.env.global_safe_zone_frontiers, 
                               self.env.uncovered_safe_frontiers, k=k)
                
            for robot in self.robot_list:
                robot.update_past_locations()

            all_robot_past_locations = []
            for robot in self.robot_list:
                robot.update_planning_state(self.env.robot_locations)
                all_robot_past_locations.append(robot.past_positions_index)

            if self.save_image:
                self.env.plot_env_test(i, SIM_STEP, self.robot_list)

            # Check if episode is done
            if np.sum(self.robot_list[0].utility_safe) == 0:
                done = True
            else:
                done = False  

            _, safety_increase = self.env.calculate_team_reward(dist_list)
            if safety_increase > 0:
                safe_increase_record.append(1)
            else:
                safe_increase_record.append(0)

            max_travel_dist = max([robot.travel_dist for robot in self.robot_list])
            length_history.append(max_travel_dist)
            safe_rate_history.append(self.env.safe_rate)
            explored_rate_history.append(self.env.explored_rate)
            ground_explored_rate_history.append(self.env.explored_rate_ground)

            if done:
                if self.save_image:
                    self.env.plot_env_test(i+1, 0, self.robot_list)
                break
        
        if done:
            true_dist = max_travel_dist
        else:
            true_dist = np.nan

        # Record epsiode performance
        self.perf_metrics['travel_dist'] = max_travel_dist
        self.perf_metrics['true_travel_dist'] = true_dist
        self.perf_metrics['explored_rate'] = self.env.explored_rate
        self.perf_metrics['explored_rate_ground'] = self.env.explored_rate_ground
        self.perf_metrics['safe_rate'] = self.env.safe_rate
        self.perf_metrics['success_rate'] = done
        self.perf_metrics['safe_increase_rate'] = np.mean(safe_increase_record)
        self.perf_metrics['length_history'] = length_history
        self.perf_metrics['safe_rate_history'] = safe_rate_history
        self.perf_metrics['explored_rate_history'] = explored_rate_history
        self.perf_metrics['explored_rate_ground_history'] = ground_explored_rate_history

        if self.save_image:
            make_gif(self.save_path, self.global_step, self.env.frame_files, self.env.explored_rate)

    def solve_path_confict(self, selected_locations, dist_list):
        selected_locations = np.array(selected_locations).reshape(-1, 2)
        arriving_sequence = np.argsort(np.array(dist_list))
        selected_locations_in_arriving_sequence = np.array(selected_locations)[arriving_sequence]

        for j, selected_location in enumerate(selected_locations_in_arriving_sequence):
            solved_locations = selected_locations_in_arriving_sequence[:j]
            while selected_location[0] + selected_location[1] * 1j in solved_locations[:, 0] + solved_locations[:,
                                                                                               1] * 1j:
                id = arriving_sequence[j]
                nearby_nodes = self.robot_list[id].node_manager.nodes_dict.nearest_neighbors(
                    selected_location.tolist(), 25)
                for node in nearby_nodes:
                    coords = node.data.coords
                    if coords[0] + coords[1] * 1j in solved_locations[:, 0] + solved_locations[:, 1] * 1j:
                        continue
                    selected_location = coords
                    break

                selected_locations_in_arriving_sequence[j] = selected_location
                selected_locations[id] = selected_location
        return selected_locations
    
if __name__ == '__main__':
    from model import PolicyNet
    if LOAD_MODEL:
        checkpoint = torch.load(load_path + '/checkpoint.pth', map_location='cpu',  weights_only=True)
    net = PolicyNet(NODE_INPUT_DIM, FRONTIER_CLUSTER_INPUT_DIM, EMBEDDING_DIM)
    if LOAD_MODEL:
        net.load_state_dict(checkpoint['policy_model'])
    test_worker = Test_worker(0, net, 1, N_AGENTS, 'cpu', True)
    test_worker.run_episode()