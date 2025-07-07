from copy import deepcopy
import time
import numpy as np
import torch
from utils import *
from parameter import *


class Agent:
    def __init__(self, id, policy_net, node_manager, robot_height, robot_height_cell, max_height_cell, device='cpu',
                 plot=False):
        self.id = id
        self.device = device
        self.plot = plot
        self.policy_net = policy_net
        self.robot_height = robot_height
        self.robot_height_cell = robot_height_cell
        self.max_robot_height_cell = max_height_cell
        self.next_locations = None
        # location and global map
        self.location = None
        self.global_flight_info = None
        self.global_ground_info = None
        self.global_safe_info = None
        self.local_center = None

        # Map related parameters
        self.cell_size = CELL_SIZE
        self.downsample_size = NODE_RESOLUTION  # cell
        self.downsampled_cell_size = self.cell_size * self.downsample_size  # meter
        self.local_map_size = LOCAL_MAP_SIZE  # meter
        self.updating_map_size = UPDATING_MAP_SIZE

        # Map
        self.flight_info = None
        self.ground_info = None
        self.safe_info = None

        # Local frontiers
        self.frontier_flight, self.frontier_ground, self.frontier_safe = None, None, None

        # Managers
        self.node_manager = node_manager

        # Sensor
        self.sensor_range = SENSOR_RANGE
        self.sensor_range_cell = round(self.sensor_range / self.cell_size)

        # Local graph
        self.node_coords, self.utility, self.utility_ground, self.guidepost, self.occupancy, self.guidepost_ground = None, None, None, None, None, None
        self.visited_node, self.path_coords, self.path_coords_ground = None, None, None
        self.current_index, self.adjacent_matrix, self.neighbor_indices = None, None, None
        self.utility_safe, self.utility_uncovered_safe, self.safe_indicator, self.counter_safe_indicator = None, None, None, None
        self.guidepost_safe, self.guidepost_uncovered_safe = None, None
        self.frontiers_centroids, self.frontiers_clusters = None, None
        self.visitations = None
        self.past_locations = []

        self.travel_dist = 0
        self.episode_buffer = []

        for i in range(EPISODE_BUFFER_SIZE):
            self.episode_buffer.append([])

        if self.plot:
            self.trajectory_x = []
            self.trajectory_y = []

    def update_global_map(self, global_flight_info, global_ground_info, global_safe_zone):
        # no need in training because of shallow copy
        self.global_flight_info = global_flight_info
        self.global_ground_info = global_ground_info
        self.global_safe_info = global_safe_zone

    def update_map(self):
        self.flight_info = self.get_updating_map(self.location, self.global_flight_info)
        self.ground_info = self.get_updating_map(self.location, self.global_ground_info)
        self.safe_info = self.get_updating_map(self.location, self.global_safe_info)

    def update_location(self, location):
        if self.location is None:
            dist = 0
        else:
            dist = np.linalg.norm(self.location - location)
        self.travel_dist += dist

        self.location = location

        # if self.node_manager.nodes_dict.__len__() == 0:
        #     pass
        # else:
        if len(self.past_locations) >= TRAJECTORY_LEN:
            # node = self.node_manager.nodes_dict.find(self.past_locations[0].tolist())
            # node.data.visited = 0
            self.past_locations.pop(0)
        self.past_locations.append(location)

        if self.plot:
            self.trajectory_x.append(location[0])
            self.trajectory_y.append(location[1])

    def update_frontiers(self, flight_frontiers, ground_frontiers, safe_zone_frontiers):
        self.frontier_flight = flight_frontiers
        self.frontier_ground = ground_frontiers  # in 3d coordinates, meters
        self.frontier_safe = safe_zone_frontiers

    def update_safe_graph(self, safe_zone_info, uncovered_safe_frontiers, counter_safe_info):
        self.update_global_safe_zone(safe_zone_info)
        self.update_local_safe_zone(counter_safe_info)
        self.update_safe_frontiers()
        self.node_manager.update_local_safe_graph(self.location, self.safe_frontier, uncovered_safe_frontiers,
                                                  self.extended_local_counter_safe_zone_info,
                                                  self.extended_local_safe_zone_info, self.extended_local_map_info)

    def update_graph(self, global_flight_info, global_ground_info, location, global_safe_info, flight_frontiers,
                     ground_frontiers, safe_zone_frontiers, uncovered_safe_frontiers, robot_locations=None, k=None):
        self.update_global_map(global_flight_info, global_ground_info, global_safe_info)
        self.update_location(location)
        # self.update_map()
        self.update_frontiers(flight_frontiers, ground_frontiers, safe_zone_frontiers)  # For global frontiers

        # if self.frontier_safe == []:
        #     print(self.frontier_safe)
        #     plt.subplot(1, 2, 1)
        #     plt.axis('off')
        #     plt.title('Safe zone', y=-0.12, fontweight='bold')
        #     plt.imshow(global_flight_info.map, cmap='gray_r')
        #     plt.imshow(global_flight_info.map == -127, cmap=plt.cm.colors.ListedColormap(['none', 'lightblue']), alpha=0.8)
        #     plt.imshow(self.global_safe_info.map, cmap='Greens', alpha=0.55)
        #     robot_cell = get_cell_position_from_coords(self.location, self.global_flight_info)
        #     plt.plot(robot_cell[0], robot_cell[1], 'r'+'o', markersize=10, zorder=5)
        #     plt.imshow(self.safe_info.map, cmap='Oranges', alpha=0.55)

        #     #Plot safe frontiers
        #     # if len(covered_safe_frontiers) != 0:
        #     #     covered_safe_frontier_cells = get_cell_position_from_coords(np.array(list(self.env.covered_safe_frontiers))[:,:2], self.env.safe_info).reshape(-1, 2)
        #     #     plt.scatter(covered_safe_frontier_cells[:, 0], covered_safe_frontier_cells[:, 1], c='g', s=1, zorder=6)
        #     if len(uncovered_safe_frontiers) != 0:
        #         uncovered_safe_frontier_cells = get_cell_position_from_coords(np.array(list(uncovered_safe_frontiers))[:,:2], self.global_safe_info).reshape(-1, 2)
        #         plt.scatter(uncovered_safe_frontier_cells[:, 0], uncovered_safe_frontier_cells[:, 1], c='r', s=1, zorder=6)
        # Update graph
        self.node_manager.update_graph(self.location,  # in meters
                                       self.frontier_flight,
                                       self.frontier_ground,
                                       self.frontier_safe,
                                       uncovered_safe_frontiers,
                                       self.global_flight_info,
                                       self.global_ground_info,
                                       self.global_safe_info,
                                       robot_locations,
                                       k)

        # self.node_manager.update_past_locations(self.past_locations)

    def update_past_locations(self):
        self.node_manager.update_past_locations(self.past_locations)

    def update_planning_state(self, robot_locations):
        (self.node_coords, self.utility, self.guidepost, self.occupancy, self.adjacent_matrix, self.current_index,
         self.neighbor_indices,
         self.utility_ground, self.guidepost_ground, self.path_coords_ground,
         self.utility_safe, self.utility_uncovered_safe, self.safe_indicator, self.occluded_indicator,
         self.guidepost_safe,
         self.guidepost_uncovered_safe, self.visitations, self.past_positions_index, self.frontiers_centroids,
         self.frontiers_clusters) = self.node_manager.get_all_node_graph(self.location, robot_locations,
                                                                         self.past_locations)

    def get_observation(self, all_robot_past_locations, pad=True):
        node_coords = self.node_coords
        node_utility = self.utility.reshape(-1, 1)
        node_utility_ground = self.utility_ground.reshape(-1, 1)
        node_utility_safe = self.utility_safe.reshape(-1, 1)
        node_utility_uncovered_safe = self.utility_uncovered_safe.reshape(-1, 1)
        node_guidepost = self.guidepost.reshape(-1, 1)
        node_guidepost_ground = self.guidepost_ground.reshape(-1, 1)
        # node_dist_guidepost = self.dist_guidepost.reshape(-1, 1)
        # node_dist_guidepost_ground = self.dist_guidepost_ground.reshape(-1, 1)
        node_occupancy = self.occupancy.reshape(-1, 1)
        # neighboring_robot_indices = np.where(node_occupancy==1)[0]
        node_safe_indicator = self.safe_indicator.reshape(-1, 1)
        node_occluded_indicator = self.occluded_indicator.reshape(-1, 1)
        node_guidepost_safe = self.guidepost_safe.reshape(-1, 1)
        node_guidepost_uncovered_safe = self.guidepost_uncovered_safe.reshape(-1, 1)
        node_visitations = self.visitations.reshape(-1, 1)

        current_index = self.current_index
        edge_mask = self.adjacent_matrix
        current_edge = self.neighbor_indices  # indices of the neigbors of the current position
        n_node = node_coords.shape[0]
        # border_coords = get_coords_from_cell_position(self.border_points, self.global_flight_info)

        # Get all robot past locations data
        all_robot_past_locations.pop(self.id)
        all_robot_past_locations = np.array(all_robot_past_locations)
        all_robot_past_locations = torch.tensor(all_robot_past_locations).unsqueeze(0).to(self.device)
        # if len((all_robot_past_locations.shape) == 3:):
        #     # assert local_node_coords.shape[0] < NODE_PADDING_SIZE, print(local_node_coords.shape[0])
        #     padding = torch.nn.ZeroPad2d((0, 0, 0, NODE_PADDING_SIZE - n_node))
        #     node_inputs = padding(node_inputs)                                      

        current_robot_past_locations = torch.tensor(self.past_positions_index).unsqueeze(0).unsqueeze(0).to(self.device)

        current_node_coords = node_coords[self.current_index]
        local_node_coords = np.concatenate(
            (node_coords[:, 0].reshape(-1, 1) - current_node_coords[0],  # Scale to [-1, 1]
             node_coords[:, 1].reshape(-1, 1) - current_node_coords[1]),
            axis=-1) / UPDATING_MAP_SIZE / 2  # LOCAL_MAP_SIZE
        # print('Start', np.max(local_node_utility), np.max(local_node_utility_ground))
        # print(np.max(local_node_utility_safe), np.max(local_node_utility_uncovered_safe))
        # print(np.max(local_node_dist_guidepost), np.max(local_node_dist_guidepost_ground))
        # print(np.max(local_node_occupancy))      

        node_utility = node_utility / (2 * self.sensor_range * 3.14 // FRONTIER_CELL_SIZE)
        node_utility_ground = node_utility_ground / (2 * self.sensor_range * 3.14 // FRONTIER_CELL_SIZE)
        node_utility_safe = node_utility_safe / (2 * self.sensor_range * 3.14 // FRONTIER_CELL_SIZE)
        node_utility_uncovered_safe = node_utility_uncovered_safe / (2 * self.sensor_range * 3.14 // FRONTIER_CELL_SIZE)
        # node_dist_guidepost = node_dist_guidepost / UPDATING_MAP_SIZE 
        # node_dist_guidepost_ground = node_dist_guidepost_ground / UPDATING_MAP_SIZE 
        # print('Check', np.max(node_utility), np.max(node_utility_ground), np.max(node_utility_safe), np.max(node_utility_uncovered_safe), np.max(node_dist_guidepost), np.max(node_dist_guidepost_ground))
        # assert np.all(node_utility >= 0) and np.all(node_utility <= 1), "local_node_utility must be between 0 and 1"
        # assert np.all(node_utility_ground >= 0) and np.all(node_utility_ground <= 1), np.max(node_utility_ground)#"local_node_utility_ground must be between 0 and 1"
        # assert np.all(node_utility_safe >= 0) and np.all(node_utility_safe <= 1), "local_node_utility_safe must be between 0 and 1"       
        # assert np.all(node_utility_uncovered_safe >= 0) and np.all(node_utility_uncovered_safe <= 1), "local_node_utility-uncovered_safe must be between 0 and 1"
        # assert np.all(node_dist_guidepost >= 0) and np.all(node_dist_guidepost <= 1), "local_node_dist_guidepost must be between 0 and 1"
        # assert np.all(node_dist_guidepost_ground >= 0) and np.all(node_dist_guidepost_ground <= 1), "local_node_dist_guidepost_ground must be between 0 and 1"
        # assert np.all(node_guidepost >= 0) and np.all(node_guidepost <= 1), "local_node_guidepost must be between 0 and 1"
        # assert np.all(node_occupancy >= -1) and np.all(node_occupancy <= 1), "local_node_occupancy must be between -1 and 1"
        # assert np.all(node_guidepost_ground >= 0) and np.all(node_guidepost_ground <= 1), "local_node_guidepost must be between 0 and 1"
        # assert np.all(node_safe_indicator >= -1) and np.all(node_safe_indicator <= 1), "local_node_safe_indicator must be between 0 and 1"
        # assert np.all(local_node_counter_safe_indicator >= 0) and np.all(local_node_counter_safe_indicator <= 1), "local_node_counter_safe_indicator must be between 0 and 1"
        # local_node_inputs = np.concatenate((local_node_coords, local_node_utility, local_node_guidepost, local_node_occupancy, local_node_utility_ground, 
        #                                             local_node_guidepost_ground, local_node_dist_guidepost, local_node_dist_guidepost_ground, local_node_utility_safe, local_node_utility_uncovered_safe,
        #                                             local_node_safe_indicator, local_node_counter_safe_indicator), axis=1)    
        node_inputs = np.concatenate(
            (local_node_coords, node_occupancy, node_utility, node_utility_ground, node_guidepost,
             node_guidepost_ground, node_utility_safe, node_utility_uncovered_safe,
             node_safe_indicator, node_occluded_indicator, node_guidepost_safe, node_guidepost_uncovered_safe,
             node_visitations), axis=1)
        node_inputs = torch.FloatTensor(node_inputs).unsqueeze(0).to(self.device)

        # Node inputs
        if pad:
            # assert local_node_coords.shape[0] < NODE_PADDING_SIZE, print(local_node_coords.shape[0])
            padding = torch.nn.ZeroPad2d((0, 0, 0, NODE_PADDING_SIZE - n_node))
            node_inputs = padding(node_inputs)  # pad the nodes to the padding size for batch training

        # Padding mask to indicate where there is a node
        node_padding_mask = torch.zeros((1, 1, n_node), dtype=torch.int16).to(self.device)
        if pad:
            node_padding = torch.ones((1, 1, NODE_PADDING_SIZE - n_node), dtype=torch.int16).to(
                self.device)
            node_padding_mask = torch.cat((node_padding_mask, node_padding), dim=-1)

        # Current node index
        current_index = torch.tensor([current_index]).reshape(1, 1, 1).to(self.device)

        # Matrix to indicate where there is node neighbors
        edge_mask = torch.tensor(edge_mask).unsqueeze(0).to(self.device)

        # Matrix to pad the neighbor adjacency matrix with 1
        if pad:
            padding = torch.nn.ConstantPad2d((0, NODE_PADDING_SIZE - n_node, 0, NODE_PADDING_SIZE - n_node), 1)
            edge_mask = padding(edge_mask)

        current_in_edge = np.argwhere(current_edge == self.current_index)[0][0]
        # Current local edge mask for current node to pad till k-size with 0
        current_edge = torch.tensor(current_edge).unsqueeze(0)
        k_size = current_edge.size()[-1]
        if pad:
            padding = torch.nn.ConstantPad1d((0, K_SIZE - k_size), 0)
            current_edge = padding(current_edge)
        current_edge = current_edge.unsqueeze(-1)

        # Local edge padding mask to show where the current robot is with 1 and non nodes to be also 1
        edge_padding_mask = torch.zeros((1, 1, k_size), dtype=torch.int16).to(self.device)
        edge_padding_mask[0, 0, current_in_edge] = 1
        if pad:
            padding = torch.nn.ConstantPad1d((0, K_SIZE - k_size), 1)
            edge_padding_mask = padding(edge_padding_mask)

        # neighboring_robot_indices = torch.tensor(neighboring_robot_indices).reshape(1, -1, 1).to(self.device)

        frontiers_cluster_coords = np.concatenate(
            (self.frontiers_centroids[:, 0].reshape(-1, 1) - current_node_coords[0],  # Scale to [-1, 1]
             self.frontiers_centroids[:, 1].reshape(-1, 1) - current_node_coords[1]),
            axis=-1) / UPDATING_MAP_SIZE / 2

        frontiers_clusters_inputs = np.concatenate((frontiers_cluster_coords, self.frontiers_clusters), axis=1)
        frontiers_clusters_inputs = torch.FloatTensor(frontiers_clusters_inputs).unsqueeze(0).to(self.device)
        n_clusters = frontiers_cluster_coords.shape[0]
        if pad:
            # assert local_node_coords.shape[0] < NODE_PADDING_SIZE, print(local_node_coords.shape[0])
            padding = torch.nn.ZeroPad2d((0, 0, 0, CLUSTERS_PADDING_SIZE - n_clusters))
            frontiers_clusters_inputs = padding(frontiers_clusters_inputs)

            # Padding mask to indicate where there is a node
        frontiers_padding_mask = torch.zeros((1, 1, n_clusters), dtype=torch.int16).to(self.device)
        if pad:
            frontiers_padding = torch.ones((1, 1, CLUSTERS_PADDING_SIZE - n_clusters), dtype=torch.int16).to(
                self.device)
            frontiers_padding_mask = torch.cat((frontiers_padding_mask, frontiers_padding), dim=-1)

        return [node_inputs, node_padding_mask, edge_mask, current_index, current_edge, edge_padding_mask,
                frontiers_clusters_inputs, frontiers_padding_mask, all_robot_past_locations,
                current_robot_past_locations]

    # def update_safe_graph(self, safe_zone_info, uncovered_safe_frontiers, counter_safe_info):
    #     self.update_global_safe_zone(safe_zone_info)
    #     self.update_local_safe_zone(counter_safe_info)
    #     self.update_safe_frontiers()
    #     self.node_manager.update_local_safe_graph(self.location, self.safe_frontier, uncovered_safe_frontiers,
    #                                               self.extended_local_counter_safe_zone_info,
    #                                               self.extended_local_safe_zone_info, self.extended_local_map_info)

    def select_next_waypoint(self, observation, greedy=False):
        _, _, _, _, current_edge, _, _, _, _, _ = observation
        with torch.no_grad():
            logp = self.policy_net(*observation)

        if greedy:
            action_index = torch.argmax(logp, dim=1).long()
        else:
            action_index = torch.multinomial(logp.exp(), 1).long().squeeze(1)

        next_node_index = current_edge[0, action_index.item(), 0].item()
        next_position = self.node_coords[next_node_index]

        return next_position, next_node_index, action_index

    def get_updating_map(self, location, map_info):
        # the map includes all nodes that may be updating
        updating_map_origin_x = (location[0] - self.updating_map_size / 2)
        updating_map_origin_y = (location[1] - self.updating_map_size / 2)

        updating_map_top_x = updating_map_origin_x + self.updating_map_size
        updating_map_top_y = updating_map_origin_y + self.updating_map_size

        min_x = map_info.map_origin_x
        min_y = map_info.map_origin_y
        max_x = (map_info.map_origin_x + self.cell_size * (map_info.map.shape[1] - 1))
        max_y = (map_info.map_origin_y + self.cell_size * (map_info.map.shape[0] - 1))

        if updating_map_origin_x < min_x:
            updating_map_origin_x = min_x
        if updating_map_origin_y < min_y:
            updating_map_origin_y = min_y
        if updating_map_top_x > max_x:
            updating_map_top_x = max_x
        if updating_map_top_y > max_y:
            updating_map_top_y = max_y

        updating_map_origin_x = (updating_map_origin_x // self.cell_size + 1) * self.cell_size
        updating_map_origin_y = (updating_map_origin_y // self.cell_size + 1) * self.cell_size
        updating_map_top_x = (updating_map_top_x // self.cell_size) * self.cell_size
        updating_map_top_y = (updating_map_top_y // self.cell_size) * self.cell_size

        updating_map_origin_x = np.round(updating_map_origin_x, 1)
        updating_map_origin_y = np.round(updating_map_origin_y, 1)
        updating_map_top_x = np.round(updating_map_top_x, 1)
        updating_map_top_y = np.round(updating_map_top_y, 1)

        updating_map_origin = np.array([updating_map_origin_x, updating_map_origin_y])
        updating_map_origin_in_global_map = get_cell_position_from_coords(updating_map_origin, map_info)

        updating_map_top = np.array([updating_map_top_x, updating_map_top_y])
        updating_map_top_in_global_map = get_cell_position_from_coords(updating_map_top, map_info)

        updating_map = map_info.map[
                       updating_map_origin_in_global_map[1]:updating_map_top_in_global_map[1] + 1,
                       updating_map_origin_in_global_map[0]:updating_map_top_in_global_map[0] + 1]

        updating_map_info = Map_info(updating_map, updating_map_origin_x, updating_map_origin_y, self.cell_size)

        return updating_map_info

    def save_observation(self, observation):
        node_inputs, node_padding_mask, edge_mask, current_index, current_edge, edge_padding_mask, frontiers_clusters, frontiers_padding_mask, all_robot_past_locations, current_robot_past_locations = observation
        self.episode_buffer[0] += node_inputs
        self.episode_buffer[1] += node_padding_mask.bool()
        self.episode_buffer[2] += edge_mask.bool()
        self.episode_buffer[3] += current_index
        self.episode_buffer[4] += current_edge
        self.episode_buffer[5] += edge_padding_mask.bool()
        self.episode_buffer[6] += frontiers_clusters
        self.episode_buffer[7] += frontiers_padding_mask.bool()
        self.episode_buffer[8] += all_robot_past_locations
        self.episode_buffer[9] += current_robot_past_locations

    def save_action(self, action_index):
        self.episode_buffer[10] += action_index.reshape(1, 1, 1)

    def save_reward(self, reward):
        self.episode_buffer[11] += torch.FloatTensor([reward]).reshape(1, 1, 1).to(self.device)

    def save_done(self, done):
        self.episode_buffer[12] += torch.tensor([int(done)]).reshape(1, 1, 1).to(self.device)

    def save_all_indices(self, all_agent_curr_indices):
        self.episode_buffer[23] += torch.tensor(all_agent_curr_indices).reshape(1, -1, 1).to(self.device)

    def save_next_observations(self, observation, next_node_index_list):
        # Copy the current state (indices 0-7 and 13) to the next state block with an offset of +4.
        self.episode_buffer[13] = deepcopy(self.episode_buffer[0])[1:]
        self.episode_buffer[14] = deepcopy(self.episode_buffer[1])[1:]
        self.episode_buffer[15] = deepcopy(self.episode_buffer[2])[1:]
        self.episode_buffer[16] = deepcopy(self.episode_buffer[3])[1:]
        self.episode_buffer[17] = deepcopy(self.episode_buffer[4])[1:]
        self.episode_buffer[18] = deepcopy(self.episode_buffer[5])[1:]
        self.episode_buffer[19] = deepcopy(self.episode_buffer[6])[1:]
        self.episode_buffer[20] = deepcopy(self.episode_buffer[7])[1:]
        self.episode_buffer[21] = deepcopy(self.episode_buffer[8])[1:]
        self.episode_buffer[22] = deepcopy(self.episode_buffer[9])[1:]
        self.episode_buffer[24] = deepcopy(self.episode_buffer[23])[1:]

        node_inputs, node_padding_mask, edge_mask, current_index, current_edge, edge_padding_mask, frontiers_clusters, frontiers_padding_mask, all_robot_past_locations, current_robot_past_locations = observation
        self.episode_buffer[13] += node_inputs
        self.episode_buffer[14] += node_padding_mask.bool()
        self.episode_buffer[15] += edge_mask.bool()
        self.episode_buffer[16] += current_index
        self.episode_buffer[17] += current_edge
        self.episode_buffer[18] += edge_padding_mask.bool()
        self.episode_buffer[19] += frontiers_clusters
        self.episode_buffer[20] += frontiers_padding_mask.bool()
        self.episode_buffer[21] += all_robot_past_locations
        self.episode_buffer[22] += current_robot_past_locations

        self.episode_buffer[24] += torch.tensor(next_node_index_list).reshape(1, -1, 1).to(self.device)
        self.episode_buffer[25] = deepcopy(self.episode_buffer[24])[1:]
        self.episode_buffer[25] += deepcopy(self.episode_buffer[24])[-1:]

    def save_global_state(self, ground_truth_observation):
        node_inputs, node_padding_mask, edge_mask, current_index, current_edge, edge_padding_mask, frontiers_clusters, frontiers_padding_mask, all_robot_past_locations, current_robot_past_locations = ground_truth_observation
        self.episode_buffer[26] += node_inputs
        self.episode_buffer[27] += node_padding_mask.bool()
        self.episode_buffer[28] += edge_mask.bool()
        self.episode_buffer[29] += current_index
        self.episode_buffer[30] += current_edge
        self.episode_buffer[31] += edge_padding_mask.bool()
        self.episode_buffer[32] += frontiers_clusters
        self.episode_buffer[33] += frontiers_padding_mask.bool()
        self.episode_buffer[34] += all_robot_past_locations
        self.episode_buffer[35] += current_robot_past_locations

    def save_global_next_state(self, ground_truth_observation):
        self.episode_buffer[36] = deepcopy(self.episode_buffer[26])[1:]
        self.episode_buffer[37] = deepcopy(self.episode_buffer[27])[1:]
        self.episode_buffer[38] = deepcopy(self.episode_buffer[28])[1:]
        self.episode_buffer[39] = deepcopy(self.episode_buffer[29])[1:]
        self.episode_buffer[40] = deepcopy(self.episode_buffer[30])[1:]
        self.episode_buffer[41] = deepcopy(self.episode_buffer[31])[1:]
        self.episode_buffer[42] = deepcopy(self.episode_buffer[32])[1:]
        self.episode_buffer[43] = deepcopy(self.episode_buffer[33])[1:]
        self.episode_buffer[44] = deepcopy(self.episode_buffer[34])[1:]
        self.episode_buffer[45] = deepcopy(self.episode_buffer[35])[1:]

        node_inputs, node_padding_mask, edge_mask, current_index, current_edge, edge_padding_mask, frontiers_clusters, frontiers_padding_mask, all_robot_past_locations, current_robot_past_locations = ground_truth_observation
        self.episode_buffer[36] += node_inputs
        self.episode_buffer[37] += node_padding_mask.bool()
        self.episode_buffer[38] += edge_mask.bool()
        self.episode_buffer[39] += current_index
        self.episode_buffer[40] += current_edge
        self.episode_buffer[41] += edge_padding_mask.bool()
        self.episode_buffer[42] += frontiers_clusters
        self.episode_buffer[43] += frontiers_padding_mask.bool()
        self.episode_buffer[44] += all_robot_past_locations
        self.episode_buffer[45] += current_robot_past_locations


# from copy import deepcopy
# import time
# import numpy as np
# import torch
# from utils import *
# from parameter import *


# class Agent:
#     def __init__(self, id, policy_net, node_manager, ground_truth_node_manager, robot_height, robot_height_cell, max_height_cell, device='cpu', plot=False):
#         self.id = id
#         self.device = device
#         self.plot = plot
#         self.policy_net = policy_net
#         self.robot_height = robot_height
#         self.robot_height_cell = robot_height_cell
#         self.max_robot_height_cell = max_height_cell
        
#         # location and global map
#         self.location = None
#         self.global_flight_info = None
#         self.global_ground_info = None
#         self.global_safe_info = None
#         self.local_center = None

#         # Map related parameters
#         self.cell_size = CELL_SIZE
#         self.downsample_size = NODE_RESOLUTION  # cell
#         self.downsampled_cell_size = self.cell_size * self.downsample_size  # meter
#         self.local_map_size = LOCAL_MAP_SIZE  # meter
#         self.updating_map_size = UPDATING_MAP_SIZE

#         # Map
#         self.flight_info = None
#         self.ground_info = None
#         self.safe_info = None

#         # Local frontiers
#         self.frontier_flight, self.frontier_ground, self.frontier_safe = None, None, None

#         # Managers
#         self.node_manager = node_manager
#         self.ground_truth_node_manager = ground_truth_node_manager

#         # Sensor
#         self.sensor_range = SENSOR_RANGE
#         self.sensor_range_cell = round(self.sensor_range / self.cell_size)
        
#         # Local graph
#         self.node_coords, self.utility, self.utility_ground, self.guidepost, self.occupancy, self.guidepost_ground = None, None, None, None, None, None
#         self.visited_node, self.path_coords, self.path_coords_ground = None, None, None
#         self.current_index, self.adjacent_matrix, self.neighbor_indices = None, None, None
#         self.utility_safe, self.utility_uncovered_safe, self.safe_indicator, self.counter_safe_indicator = None, None, None, None
#         self.guidepost_safe, self.guidepost_uncovered_safe = None, None
#         self.frontiers_centroids, self.frontiers_clusters = None, None
    
#         self.travel_dist = 0
#         self.episode_buffer = []

#         for i in range(EPISODE_BUFFER_SIZE):
#             self.episode_buffer.append([])

#         if self.plot:
#             self.trajectory_x = []
#             self.trajectory_y = []

#     def update_global_map(self, global_flight_info, global_ground_info, global_safe_zone):
#         # no need in training because of shallow copy
#         self.global_flight_info = global_flight_info
#         self.global_ground_info = global_ground_info
#         self.global_safe_info = global_safe_zone
        
#     def update_map(self):
#         self.flight_info = self.get_updating_map(self.location, self.global_flight_info)
#         self.ground_info = self.get_updating_map(self.location, self.global_ground_info)
#         self.safe_info = self.get_updating_map(self.location, self.global_safe_info)
#     def update_location(self, location):
#         if self.location is None:
#             dist = 0
#         else:
#             dist = np.linalg.norm(self.location - location)
#         self.travel_dist += dist

#         self.location = location

#         node = self.node_manager.nodes_dict.find(location.tolist())

#         if self.node_manager.nodes_dict.__len__() == 0:
#             pass
#         # else:
#         #     if not node == None:
#         #         print('len of nodes',self.node_manager.nodes_dict.__len__())
#         #         print('nodes', node)
#         #         node.data.set_visited()

#         if self.plot:
#             self.trajectory_x.append(location[0])
#             self.trajectory_y.append(location[1])

#     def update_frontiers(self, flight_frontiers, ground_frontiers, safe_zone_frontiers):
#         self.frontier_flight = flight_frontiers
#         self.frontier_ground = ground_frontiers    # in 3d coordinates, meters
#         self.frontier_safe = safe_zone_frontiers
#     def update_graph(self, global_flight_info, global_ground_info, location, global_safe_info, flight_frontiers, ground_frontiers, safe_zone_frontiers, uncovered_safe_frontiers, ground_truth_manager, robot_locations=None, k=None):
#         self.update_global_map(global_flight_info, global_ground_info, global_safe_info)                             
#         self.update_location(location)
#         # self.update_map()
#         self.update_frontiers(flight_frontiers, ground_frontiers, safe_zone_frontiers)   # For global frontiers

#         # if self.frontier_safe == []:
#         #     print(self.frontier_safe)
#         #     plt.subplot(1, 2, 1)
#         #     plt.axis('off')
#         #     plt.title('Safe zone', y=-0.12, fontweight='bold')
#         #     plt.imshow(global_flight_info.map, cmap='gray_r')
#         #     plt.imshow(global_flight_info.map == -127, cmap=plt.cm.colors.ListedColormap(['none', 'lightblue']), alpha=0.8)
#         #     plt.imshow(self.global_safe_info.map, cmap='Greens', alpha=0.55)
#         #     robot_cell = get_cell_position_from_coords(self.location, self.global_flight_info)
#         #     plt.plot(robot_cell[0], robot_cell[1], 'r'+'o', markersize=10, zorder=5)
#         #     plt.imshow(self.safe_info.map, cmap='Oranges', alpha=0.55)

#         #     #Plot safe frontiers
#         #     # if len(covered_safe_frontiers) != 0:
#         #     #     covered_safe_frontier_cells = get_cell_position_from_coords(np.array(list(self.env.covered_safe_frontiers))[:,:2], self.env.safe_info).reshape(-1, 2)
#         #     #     plt.scatter(covered_safe_frontier_cells[:, 0], covered_safe_frontier_cells[:, 1], c='g', s=1, zorder=6)
#         #     if len(uncovered_safe_frontiers) != 0:
#         #         uncovered_safe_frontier_cells = get_cell_position_from_coords(np.array(list(uncovered_safe_frontiers))[:,:2], self.global_safe_info).reshape(-1, 2)
#         #         plt.scatter(uncovered_safe_frontier_cells[:, 0], uncovered_safe_frontier_cells[:, 1], c='r', s=1, zorder=6)
#         # Update graph
#         # self.node_manager.update_graph(self.location,                                     # in meters
#         #                                 self.frontier_flight,
#         #                                 self.frontier_ground,
#         #                                 self.frontier_safe,
#         #                                 uncovered_safe_frontiers,
#         #                                 self.global_flight_info,
#         #                                 self.global_ground_info,
#         #                                 self.global_safe_info,
#         #                                 self.global_flight_info,
#         #                                 self.global_ground_info,
#         #                                 ground_truth_manager,
#         #                                 robot_locations,
#         #                                 k)
#         self.node_manager.update_graph(self.location,  # in meters
#                                        self.frontier_flight,
#                                        self.frontier_ground,
#                                        self.frontier_safe,
#                                        uncovered_safe_frontiers,
#                                        self.global_flight_info,
#                                        self.global_ground_info,
#                                        self.global_safe_info,
#                                        robot_locations,
#                                        k)

#     def update_planning_state(self, robot_locations):
#         (self.node_coords, self.utility, self.guidepost, self.occupancy, self.adjacent_matrix, self.current_index, self.neighbor_indices, 
#          self.utility_ground, self.guidepost_ground, self.path_coords_ground, 
#          self.utility_safe, self.utility_uncovered_safe, self.safe_indicator, self.occluded_indicator, self.guidepost_safe, 
#          self.guidepost_uncovered_safe, self.frontiers_centroids, self.frontiers_clusters) = self.node_manager.get_all_node_graph(self.location, robot_locations, self.past_locations)

#     def get_observation(self, pad=True):
#         node_coords = self.node_coords
#         node_utility = self.utility.reshape(-1, 1)
#         node_utility_ground = self.utility_ground.reshape(-1, 1)
#         node_utility_safe = self.utility_safe.reshape(-1, 1)
#         node_utility_uncovered_safe = self.utility_uncovered_safe.reshape(-1, 1)
#         node_guidepost = self.guidepost.reshape(-1, 1)
#         node_guidepost_ground = self.guidepost_ground.reshape(-1, 1)
#         # node_dist_guidepost = self.dist_guidepost.reshape(-1, 1)
#         # node_dist_guidepost_ground = self.dist_guidepost_ground.reshape(-1, 1)
#         node_occupancy = self.occupancy.reshape(-1, 1)
#         # neighboring_robot_indices = np.where(node_occupancy==1)[0]
#         node_safe_indicator = self.safe_indicator.reshape(-1, 1)
#         node_occluded_indicator = self.occluded_indicator.reshape(-1, 1)
#         node_guidepost_safe = self.guidepost_safe.reshape(-1, 1)
#         node_guidepost_uncovered_safe = self.guidepost_uncovered_safe.reshape(-1, 1)

#         current_index = self.current_index
#         edge_mask = self.adjacent_matrix
#         current_edge = self.neighbor_indices           # indices of the neigbors of the current position
#         n_node = node_coords.shape[0]
#         # border_coords = get_coords_from_cell_position(self.border_points, self.global_flight_info)

#         current_node_coords = node_coords[self.current_index]
#         local_node_coords = np.concatenate((node_coords[:, 0].reshape(-1, 1) - current_node_coords[0],            # Scale to [-1, 1]
#                                             node_coords[:, 1].reshape(-1, 1) - current_node_coords[1]),
#                                            axis=-1) / UPDATING_MAP_SIZE / 2 #LOCAL_MAP_SIZE
#         # print('Start', np.max(local_node_utility), np.max(local_node_utility_ground))
#         # print(np.max(local_node_utility_safe), np.max(local_node_utility_uncovered_safe))
#         # print(np.max(local_node_dist_guidepost), np.max(local_node_dist_guidepost_ground))
#         # print(np.max(local_node_occupancy))      
        
#         node_utility = node_utility / (2 * self.sensor_range * 3.14 // FRONTIER_CELL_SIZE)
#         node_utility_ground = node_utility_ground / (2 * self.sensor_range * 3.14 // FRONTIER_CELL_SIZE)
#         node_utility_safe = node_utility_safe / (2 * self.sensor_range * 3.14 // FRONTIER_CELL_SIZE)
#         node_utility_uncovered_safe = node_utility_uncovered_safe / (2 * self.sensor_range * 3.14 // FRONTIER_CELL_SIZE)
#         # node_dist_guidepost = node_dist_guidepost / UPDATING_MAP_SIZE 
#         # node_dist_guidepost_ground = node_dist_guidepost_ground / UPDATING_MAP_SIZE 
#         # print('Check', np.max(node_utility), np.max(node_utility_ground), np.max(node_utility_safe), np.max(node_utility_uncovered_safe), np.max(node_dist_guidepost), np.max(node_dist_guidepost_ground))
#         # assert np.all(node_utility >= 0) and np.all(node_utility <= 1), "local_node_utility must be between 0 and 1"
#         # assert np.all(node_utility_ground >= 0) and np.all(node_utility_ground <= 1), np.max(node_utility_ground)#"local_node_utility_ground must be between 0 and 1"
#         # assert np.all(node_utility_safe >= 0) and np.all(node_utility_safe <= 1), "local_node_utility_safe must be between 0 and 1"       
#         # assert np.all(node_utility_uncovered_safe >= 0) and np.all(node_utility_uncovered_safe <= 1), "local_node_utility-uncovered_safe must be between 0 and 1"
#         # assert np.all(node_dist_guidepost >= 0) and np.all(node_dist_guidepost <= 1), "local_node_dist_guidepost must be between 0 and 1"
#         # assert np.all(node_dist_guidepost_ground >= 0) and np.all(node_dist_guidepost_ground <= 1), "local_node_dist_guidepost_ground must be between 0 and 1"
#         # assert np.all(node_guidepost >= 0) and np.all(node_guidepost <= 1), "local_node_guidepost must be between 0 and 1"
#         # assert np.all(node_occupancy >= -1) and np.all(node_occupancy <= 1), "local_node_occupancy must be between -1 and 1"
#         # assert np.all(node_guidepost_ground >= 0) and np.all(node_guidepost_ground <= 1), "local_node_guidepost must be between 0 and 1"
#         # assert np.all(node_safe_indicator >= -1) and np.all(node_safe_indicator <= 1), "local_node_safe_indicator must be between 0 and 1"
#         # assert np.all(local_node_counter_safe_indicator >= 0) and np.all(local_node_counter_safe_indicator <= 1), "local_node_counter_safe_indicator must be between 0 and 1"
#         # local_node_inputs = np.concatenate((local_node_coords, local_node_utility, local_node_guidepost, local_node_occupancy, local_node_utility_ground, 
#         #                                             local_node_guidepost_ground, local_node_dist_guidepost, local_node_dist_guidepost_ground, local_node_utility_safe, local_node_utility_uncovered_safe,
#         #                                             local_node_safe_indicator, local_node_counter_safe_indicator), axis=1)    
#         node_inputs = np.concatenate((local_node_coords, node_occupancy, node_utility, node_utility_ground, node_guidepost,
#                                         node_guidepost_ground, node_utility_safe, node_utility_uncovered_safe,
#                                         node_safe_indicator, node_occluded_indicator, node_guidepost_safe, node_guidepost_uncovered_safe), axis=1)        
#         node_inputs = torch.FloatTensor(node_inputs).unsqueeze(0).to(self.device)
        
#         # Node inputs
#         if pad:
#             # assert local_node_coords.shape[0] < NODE_PADDING_SIZE, print(local_node_coords.shape[0])
#             padding = torch.nn.ZeroPad2d((0, 0, 0, NODE_PADDING_SIZE - n_node))
#             node_inputs = padding(node_inputs)                                      # pad the nodes to the padding size for batch training
        
#         # Padding mask to indicate where there is a node
#         node_padding_mask = torch.zeros((1, 1, n_node), dtype=torch.int16).to(self.device)
#         if pad:
#             node_padding = torch.ones((1, 1, NODE_PADDING_SIZE - n_node), dtype=torch.int16).to(
#                 self.device)
#             node_padding_mask = torch.cat((node_padding_mask, node_padding), dim=-1)
        
#         # Current node index
#         current_index = torch.tensor([current_index]).reshape(1, 1, 1).to(self.device)
        
#         # Matrix to indicate where there is node neighbors
#         edge_mask = torch.tensor(edge_mask).unsqueeze(0).to(self.device)
        
#         # Matrix to pad the neighbor adjacency matrix with 1
#         if pad:
#             padding = torch.nn.ConstantPad2d((0, NODE_PADDING_SIZE - n_node, 0, NODE_PADDING_SIZE - n_node), 1)
#             edge_mask = padding(edge_mask)
        
#         current_in_edge = np.argwhere(current_edge == self.current_index)[0][0]
#         # Current local edge mask for current node to pad till k-size with 0
#         current_edge = torch.tensor(current_edge).unsqueeze(0)
#         k_size = current_edge.size()[-1]
#         if pad:
#             padding = torch.nn.ConstantPad1d((0, K_SIZE - k_size), 0)
#             current_edge = padding(current_edge)
#         current_edge = current_edge.unsqueeze(-1)
        
#         # Local edge padding mask to show where the current robot is with 1 and non nodes to be also 1
#         edge_padding_mask = torch.zeros((1, 1, k_size), dtype=torch.int16).to(self.device)
#         edge_padding_mask[0, 0, current_in_edge] = 1
#         if pad:
#             padding = torch.nn.ConstantPad1d((0, K_SIZE - k_size), 1)
#             edge_padding_mask = padding(edge_padding_mask)

#         # neighboring_robot_indices = torch.tensor(neighboring_robot_indices).reshape(1, -1, 1).to(self.device)

#         frontiers_cluster_coords = np.concatenate((self.frontiers_centroids[:, 0].reshape(-1, 1) - current_node_coords[0],            # Scale to [-1, 1]
#                                             self.frontiers_centroids[:, 1].reshape(-1, 1) - current_node_coords[1]),
#                                            axis=-1) / UPDATING_MAP_SIZE / 2
        
#         frontiers_clusters_inputs = np.concatenate((frontiers_cluster_coords, self.frontiers_clusters), axis=1)
#         frontiers_clusters_inputs = torch.FloatTensor(frontiers_clusters_inputs).unsqueeze(0).to(self.device)
#         n_clusters = frontiers_cluster_coords.shape[0]
#         if pad:
#             # assert local_node_coords.shape[0] < NODE_PADDING_SIZE, print(local_node_coords.shape[0])
#             padding = torch.nn.ZeroPad2d((0, 0, 0, CLUSTERS_PADDING_SIZE - n_clusters))
#             frontiers_clusters_inputs = padding(frontiers_clusters_inputs)  

#         # Padding mask to indicate where there is a node
#         frontiers_padding_mask = torch.zeros((1, 1, n_clusters), dtype=torch.int16).to(self.device)
#         if pad:
#             frontiers_padding = torch.ones((1, 1, CLUSTERS_PADDING_SIZE - n_clusters), dtype=torch.int16).to(
#                 self.device)
#             frontiers_padding_mask = torch.cat((frontiers_padding_mask, frontiers_padding), dim=-1)
        

#         return [node_inputs, node_padding_mask, edge_mask, current_index, current_edge, edge_padding_mask, frontiers_clusters_inputs, frontiers_padding_mask]
#     def select_next_waypoint(self, observation, greedy=False):
#         _, _, _, _, current_edge, _, _, _ = observation
#         with torch.no_grad():
#             logp = self.policy_net(*observation)
            
#         if greedy:
#             action_index = torch.argmax(logp, dim=1).long()
#         else:
#             action_index = torch.multinomial(logp.exp(), 1).long().squeeze(1)
            
#         next_node_index = current_edge[0, action_index.item(), 0].item()
#         next_position = self.node_coords[next_node_index]

#         return next_position, next_node_index, action_index

#     def get_updating_map(self, location, map_info):
#         # the map includes all nodes that may be updating
#         updating_map_origin_x = (location[0] - self.updating_map_size / 2)
#         updating_map_origin_y = (location[1] - self.updating_map_size / 2)

#         updating_map_top_x = updating_map_origin_x + self.updating_map_size
#         updating_map_top_y = updating_map_origin_y + self.updating_map_size

#         min_x = map_info.map_origin_x
#         min_y = map_info.map_origin_y
#         max_x = (map_info.map_origin_x + self.cell_size * (map_info.map.shape[1] - 1))
#         max_y = (map_info.map_origin_y + self.cell_size * (map_info.map.shape[0] - 1))

#         if updating_map_origin_x < min_x:
#             updating_map_origin_x = min_x
#         if updating_map_origin_y < min_y:
#             updating_map_origin_y = min_y
#         if updating_map_top_x > max_x:
#             updating_map_top_x = max_x
#         if updating_map_top_y > max_y:
#             updating_map_top_y = max_y

#         updating_map_origin_x = (updating_map_origin_x // self.cell_size + 1) * self.cell_size
#         updating_map_origin_y = (updating_map_origin_y // self.cell_size + 1) * self.cell_size
#         updating_map_top_x = (updating_map_top_x // self.cell_size) * self.cell_size
#         updating_map_top_y = (updating_map_top_y // self.cell_size) * self.cell_size

#         updating_map_origin_x = np.round(updating_map_origin_x, 1)
#         updating_map_origin_y = np.round(updating_map_origin_y, 1)
#         updating_map_top_x = np.round(updating_map_top_x, 1)
#         updating_map_top_y = np.round(updating_map_top_y, 1)

#         updating_map_origin = np.array([updating_map_origin_x, updating_map_origin_y])
#         updating_map_origin_in_global_map = get_cell_position_from_coords(updating_map_origin, map_info)

#         updating_map_top = np.array([updating_map_top_x, updating_map_top_y])
#         updating_map_top_in_global_map = get_cell_position_from_coords(updating_map_top, map_info)

#         updating_map = map_info.map[
#                     updating_map_origin_in_global_map[1]:updating_map_top_in_global_map[1]+1,
#                     updating_map_origin_in_global_map[0]:updating_map_top_in_global_map[0]+1]

#         updating_map_info = Map_info(updating_map, updating_map_origin_x, updating_map_origin_y, self.cell_size)

#         return updating_map_info
#     def save_observation(self, observation):
#         node_inputs, node_padding_mask, edge_mask, current_index, current_edge, edge_padding_mask, frontiers_clusters, frontiers_padding_mask = observation
#         self.episode_buffer[0] += node_inputs
#         self.episode_buffer[1] += node_padding_mask.bool()
#         self.episode_buffer[2] += edge_mask.bool()
#         self.episode_buffer[3] += current_index
#         self.episode_buffer[4] += current_edge
#         self.episode_buffer[5] += edge_padding_mask.bool()
#         self.episode_buffer[6] += frontiers_clusters
#         self.episode_buffer[7] += frontiers_padding_mask.bool()
        
#     def save_action(self, action_index):
#         self.episode_buffer[8] += action_index.reshape(1, 1, 1)

#     def save_reward(self, reward):
#         self.episode_buffer[9] += torch.FloatTensor([reward[0]]).reshape(1, 1, 1).to(self.device)
#         self.episode_buffer[10] += torch.FloatTensor([reward[1]]).reshape(1, 1, 1).to(self.device)
#         self.episode_buffer[11] += torch.FloatTensor([reward[2]]).reshape(1, 1, 1).to(self.device)
#         self.episode_buffer[12] += torch.FloatTensor([reward[3]]).reshape(1, 1, 1).to(self.device)
#         self.episode_buffer[13] += torch.FloatTensor([reward[4]]).reshape(1, 1, 1).to(self.device)
#     def save_done(self, done):
#         self.episode_buffer[14] += torch.tensor([int(done)]).reshape(1, 1, 1).to(self.device)
        
#     def save_all_indices(self, all_agent_curr_indices):
#         self.episode_buffer[23] += torch.tensor(all_agent_curr_indices).reshape(1, -1, 1).to(self.device)

#     def save_next_observations(self, observation, next_node_index_list):
#         self.episode_buffer[15] = deepcopy(self.episode_buffer[0])[1:]
#         self.episode_buffer[16] = deepcopy(self.episode_buffer[1])[1:]
#         self.episode_buffer[17] = deepcopy(self.episode_buffer[2])[1:]
#         self.episode_buffer[18] = deepcopy(self.episode_buffer[3])[1:]
#         self.episode_buffer[19] = deepcopy(self.episode_buffer[4])[1:]
#         self.episode_buffer[20] = deepcopy(self.episode_buffer[5])[1:]
#         self.episode_buffer[21] = deepcopy(self.episode_buffer[6])[1:]
#         self.episode_buffer[22] = deepcopy(self.episode_buffer[7])[1:]
#         self.episode_buffer[24] = deepcopy(self.episode_buffer[23])[1:]

#         node_inputs, node_padding_mask, edge_mask, current_index, current_edge, edge_padding_mask, frontiers_clusters, frontiers_padding_mask = observation
#         self.episode_buffer[15] += node_inputs
#         self.episode_buffer[16] += node_padding_mask.bool()
#         self.episode_buffer[17] += edge_mask.bool()
#         self.episode_buffer[18] += current_index
#         self.episode_buffer[19] += current_edge
#         self.episode_buffer[20] += edge_padding_mask.bool()
#         self.episode_buffer[21] += frontiers_clusters
#         self.episode_buffer[22] += frontiers_padding_mask.bool()

#         self.episode_buffer[24] += torch.tensor(next_node_index_list).reshape(1, -1, 1).to(self.device)
#         self.episode_buffer[25] = deepcopy(self.episode_buffer[24])[1:]   # exclude the first item
#         self.episode_buffer[25] += deepcopy(self.episode_buffer[24])[-1:] # add last element
         
#     def save_global_state(self, ground_truth_observation):
#         node_inputs, node_padding_mask, edge_mask, current_index, current_edge, edge_padding_mask, frontiers_clusters, frontiers_padding_mask = ground_truth_observation
#         self.episode_buffer[26] += node_inputs
#         self.episode_buffer[27] += node_padding_mask.bool()
#         self.episode_buffer[28] += edge_mask.bool()
#         self.episode_buffer[29] += current_index
#         self.episode_buffer[30] += current_edge
#         self.episode_buffer[31] += edge_padding_mask.bool()
#         self.episode_buffer[32] += frontiers_clusters
#         self.episode_buffer[33] += frontiers_padding_mask.bool()
        
#     def save_global_next_state(self, ground_truth_observation):
#         self.episode_buffer[34] = deepcopy(self.episode_buffer[26])[1:]
#         self.episode_buffer[35] = deepcopy(self.episode_buffer[27])[1:]
#         self.episode_buffer[36] = deepcopy(self.episode_buffer[28])[1:]
#         self.episode_buffer[37] = deepcopy(self.episode_buffer[29])[1:]
#         self.episode_buffer[38] = deepcopy(self.episode_buffer[30])[1:]
#         self.episode_buffer[39] = deepcopy(self.episode_buffer[31])[1:]
#         self.episode_buffer[40] = deepcopy(self.episode_buffer[32])[1:]
#         self.episode_buffer[41] = deepcopy(self.episode_buffer[33])[1:]

#         node_inputs, node_padding_mask, edge_mask, current_index, current_edge, edge_padding_mask, frontiers_clusters, frontiers_padding_mask = ground_truth_observation
#         self.episode_buffer[34] += node_inputs
#         self.episode_buffer[35] += node_padding_mask.bool()
#         self.episode_buffer[36] += edge_mask.bool()
#         self.episode_buffer[37] += current_index
#         self.episode_buffer[38] += current_edge
#         self.episode_buffer[39] += edge_padding_mask.bool()
#         self.episode_buffer[40] += frontiers_clusters
#         self.episode_buffer[41] += frontiers_padding_mask.bool()