import numpy as np
import torch
import matplotlib.pyplot as plt
from copy import deepcopy

from env import Env
from agent import Agent
from parameter import *
from utils import *
from model import PolicyNet
from node_manager import Node_manager
from ground_truth_node_manager import GroundTruthNodeManager
# from torchinfo import summary 
# import time

if not os.path.exists(gifs_path):
    os.makedirs(gifs_path)

class Multi_agent_worker:
    def __init__(self, meta_agent_id, policy_net, global_step, n_agents, device='cpu', save_image=False):
        self.meta_agent_id = meta_agent_id
        self.global_step = global_step    # Same as episode number
        self.save_image = save_image
        self.device = device

        self.n_agents = n_agents
        self.save_path = gifs_path

        self.env = Env(global_step, self.n_agents, plot=self.save_image, save_path=self.save_path)
        self.robot_height = self.env.robot_height    # metres

        self.node_manager = Node_manager(self.robot_height, self.env.robot_height_cell, self.env.max_height_cell, quad_tree_size=QUAD_TREE_SIZE, 
                                         sensor_range=self.env.sensor_range, plot=self.save_image)
        # self.ground_truth_node_manager = GroundTruthNodeManager(self.node_manager, self.env.ground_truth_info, self.env.sensor_range, self.robot_height, self.env.robot_height_cell, self.env.max_height_cell,
        #                                                         device=self.device, plot=self.save_image)
                                                                
        
        self.env.init_episode(self.node_manager)

        self.robot_list = [Agent(i, policy_net, self.node_manager, self.robot_height, self.env.robot_height_cell, 
                                 self.env.max_height_cell, self.device, self.save_image) for i in
                           range(self.n_agents)]

        self.episode_buffer = []
        self.perf_metrics = dict()
        self.episode_buffer = [[] for _ in range(EPISODE_BUFFER_SIZE)]

    def run_episode(self):
        actions = []

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
            self.env.plot_env(-1, -1, self.robot_list, self.node_manager)

        for i in range(MAX_EPISODE_STEP):
            # print('timestep', i)
            selected_locations = []
            dist_list = []
            next_node_index_list = []
            safe_increase_record = []

            for robot in self.robot_list:
                # Get states
                observation = robot.get_observation(copy(all_robot_past_locations))
                # ground_truth_observation = robot.ground_truth_node_manager.get_ground_truth_observation(robot.location, self.env.robot_locations)
                robot.save_observation(observation)
                # robot.save_global_state(ground_truth_observation)

                next_location, next_node_index, action_index = robot.select_next_waypoint(observation)
                robot.save_action(action_index)

                selected_locations.append(next_location)
                dist_list.append(np.linalg.norm(next_location - robot.location))
                next_node_index_list.append(next_node_index)

            selected_locations = self.solve_path_confict(selected_locations, dist_list)
            actions.append(selected_locations)

            curr_node_indices = np.array([robot.current_index for robot in self.robot_list]) # MAAC
            reward_list = []
            # for robot in self.robot_list:
            #     node = self.node_manager.nodes_dict.find((self.env.robot_locations[robot.id][0], self.env.robot_locations[robot.id][1])).data
            #     utility = node.utility / (2 * SENSOR_RANGE * 3.14 // FRONTIER_CELL_SIZE)
            #     utility_ground = node.utility_ground / (2 * SENSOR_RANGE * 3.14 // FRONTIER_CELL_SIZE)
            #     utility_safe = node.utility_safe / (2 * SENSOR_RANGE * 3.14 // FRONTIER_CELL_SIZE)
            #     utility_uncovered_safe = node.utility_safe_uncovered / ( 2 * SENSOR_RANGE * 3.14 // FRONTIER_CELL_SIZE)
            #     reward_list.append(utility_safe + 2 * utility_uncovered_safe)
            reward_list = np.array(reward_list)

            self.env.step(i , selected_locations, self.robot_list)

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
                self.env.plot_env(i, SIM_STEP, self.robot_list)

            # safe = self.env.check_done
            
            if np.sum(self.robot_list[0].utility_safe) == 0:
                done = True
            else:
                done = False  

            # Check reward again
            team_reward, safety_increase = self.env.calculate_team_reward(dist_list)
            # print(reward_list, '\n')
            reward_list = team_reward
            # Task completion reward
            max_travel_dist = max([robot.travel_dist for robot in self.robot_list])
            
            if done:
                reward_list += 20000 / max_travel_dist
                
            if safety_increase > 0:
                safe_increase_record.append(1)
            else:
                safe_increase_record.append(0)

            for robot, reward in zip(self.robot_list, reward_list):
                robot.save_reward(reward)
                robot.save_all_indices(np.array(curr_node_indices))
                robot.save_done(done)
     
            if done:
                if self.save_image:
                    self.env.plot_env(i+1, 0, self.robot_list)
                break

        self.perf_metrics['travel_dist'] = max_travel_dist
        if done:
            true_dist = max_travel_dist
        else:
            true_dist = np.nan

        safe = True
        self.perf_metrics['true_travel_dist'] = true_dist
        self.perf_metrics['explored_rate'] = self.env.explored_rate
        self.perf_metrics['explored_rate_ground'] = self.env.explored_rate_ground
        self.perf_metrics['success_rate'] = done
        self.perf_metrics['safe_success_rate'] = safe  # safe means there is no intrude of unsafe zone on the border
        self.perf_metrics['safe_increase_rate'] = np.mean(safe_increase_record)

        if self.global_step > 1500 and not done:
            print('Current epsiode:', self.global_step)
            print('File name:', self.env.map_name)
            print('Initial location:', self.env.initial_locations)
            print('Origin (x, y):', self.env.belief_origin_x, self.env.belief_origin_y)
            print('Robot height:', self.env.robot_height_cell)
            print('Done/Safe', done)
            actions_array = np.array(actions)

            # Save only the 10 latest npy files
            filename = f'actions_{self.global_step}.npy'
            np.save(filename, actions_array)
            
            # Get list of action files and keep only latest 10
            action_files = sorted([f for f in os.listdir('.') if f.startswith('actions_') and f.endswith('.npy')], 
                                key=lambda x: int(x.split('_')[1].split('.')[0]), 
                                reverse=True)
            for old_file in action_files[10:]:
                os.remove(old_file)

        # save episode buffer
        for robot in self.robot_list:
            observation = robot.get_observation(copy(all_robot_past_locations))
            robot.save_next_observations(observation, next_node_index_list) # MAAC
            # ground_truth_observation = robot.ground_truth_node_manager.get_ground_truth_observation(robot.location, self.env.robot_locations)
            # robot.save_global_next_state(ground_truth_observation)
            for i in range(len(self.episode_buffer)):
                self.episode_buffer[i] += robot.episode_buffer[i]

        # save gif
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
    # from model import PolicyNet
    # if LOAD_MODEL:
    #     checkpoint = torch.load(load_path + '/checkpoint.pth', map_location='cpu',  weights_only=True)
    # net = PolicyNet(NODE_INPUT_DIM, FRONTIER_CLUSTER_INPUT_DIM, EMBEDDING_DIM)
    # if LOAD_MODEL:
    #     net.load_state_dict(checkpoint['policy_model'])
    worker = Multi_agent_worker(0, None, 88, N_AGENTS, 'cpu', True)
    worker.run_episode()