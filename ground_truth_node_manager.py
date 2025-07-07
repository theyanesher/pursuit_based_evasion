import time
import torch

import numpy as np
from utils import *
from parameter import *
import quads
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors


class GroundTruthNodeManager:
    def __init__(self, node_manager, ground_truth_map_info, sensor_range, robot_height, robot_height_cell, max_height, device='cpu', plot=False):
        self.nodes_dict = quads.QuadTree((0, 0), 1000, 1000)
        self.node_manager = node_manager
        self.ground_truth_map_info = ground_truth_map_info
        self.ground_truth_node_coords = None
        self.ground_truth_node_utility = None
        self.explored_sign = None
        self.utility = None
        self.utility_ground = None
        self.utility_safe = None
        self.utility_safe_uncovered = None
        self.safe_indicator = None
        self.robot_height = robot_height
        self.robot_height_cell = robot_height_cell
        self.max_height = max_height
        self.sensor_range = sensor_range
        self.device = device
        self.plot = plot

        if self.plot:
            self.x = []
            self.y = []

        self.initialize_graph()
        
        # map = np.copy(self.ground_truth_map_info.map)
        # print(self.robot_height_cell)
        # map[map <  self.robot_height_cell] = 0
        # plt.imshow(map, cmap='gray')
        # all_node_coords = []
        # for node in self.nodes_dict.__iter__():
        #     if node.data.explored == 0:
        #         all_node_coords.append(node.data.coords)
        # all_node_cells = get_cell_position_from_coords(np.array(all_node_coords), self.ground_truth_map_info)
        # plt.scatter(all_node_cells[:, 0], all_node_cells[:, 1], c='r', zorder=5)
        # for i in range(len(self.x)):
        #       plt.plot((self.x[i] - self.ground_truth_map_info.map_origin_x) / CELL_SIZE,
        #                (self.y[i] - self.ground_truth_map_info.map_origin_y) / CELL_SIZE, 'tan')
        # plt.show()
    def check_node_exist_in_dict(self, coords):
        key = (coords[0], coords[1])
        exist = self.nodes_dict.find(key)
        return exist

    def get_ground_truth_observation(self, robot_location, robot_locations):
        self.update_graph()

        all_node_coords = []
        # Ensure node order is the same as node manager
        for node in self.node_manager.nodes_dict.__iter__():
            all_node_coords.append(node.data.coords)
        for node in self.nodes_dict.__iter__():
            if node.data.explored == 0:
                all_node_coords.append(node.data.coords)
        all_node_coords = np.array(all_node_coords).reshape(-1, 2)

        explored_sign = []
        utility = []
        utility_ground = []
        utility_safe = []
        utility_safe_uncovered = []
        safe_indicator = []
        occluded_indicator = []

        n_nodes = all_node_coords.shape[0]
        # print(n_nodes)
        adjacent_matrix = np.ones((n_nodes, n_nodes)).astype(int)
        node_coords_to_check = all_node_coords[:, 0] + all_node_coords[:, 1] * 1j
        for i, coords in enumerate(all_node_coords):
            node = self.nodes_dict.find((coords[0], coords[1])).data
            utility.append(node.utility)
            utility_ground.append(node.utility_ground)
            utility_safe.append(node.utility_safe)
            utility_safe_uncovered.append(node.utility_safe_uncovered)
            safe_indicator.append(node.safe_indicator)
            occluded_indicator.append(node.occluded_indicator)
            explored_sign.append(node.explored)

            for neighbor in node.neighbor_list:
                index = np.argwhere(node_coords_to_check == neighbor[0] + neighbor[1] * 1j)
                index = index[0][0]
                adjacent_matrix[i, index] = 0

        utility = np.array(utility)
        utility_ground = np.array(utility_ground)
        utility_safe = np.array(utility_safe)
        utility_safe_uncovered = np.array(utility_safe_uncovered)
        safe_indicator = np.array(safe_indicator)
        occluded_indicator = np.array(occluded_indicator)
        explored_sign = np.array(explored_sign)

        current_index = np.argwhere(node_coords_to_check == robot_location[0] + robot_location[1] * 1j)[0][0]
        
        # neighbor_indices = np.argwhere(adjacent_matrix[current_index] == 0).reshape(-1)
        # neighbor_indices = []
        # current_node_in_belief = self.node_manager.nodes_dict.find(robot_location.tolist()).data
        # for neighbor in current_node_in_belief.neighbor_list:
        #     index = np.argwhere(node_coords_to_check == neighbor[0] + neighbor[1] * 1j)[0][0]
        #     neighbor_indices.append(index)
        # neighbor_indices = np.sort(np.array(neighbor_indices))

        indices = np.argwhere(utility > 0).reshape(-1)
        utility_node_coords = all_node_coords[indices]
        non_utility_node_coords = all_node_coords[np.argwhere(utility == 0).reshape(-1)]
        dist_dict, prev_dict = self.Dijkstra(robot_location)
        nearest_utility_coords = robot_location
        nearest_dist = 1e8
        for coords in utility_node_coords:
            dist = dist_dict[(coords[0], coords[1])]
            if 0 < dist < nearest_dist:
                nearest_dist = dist
                nearest_utility_coords = coords
                # print(nearest_dist, coords, nearest_utility_coords, robot_location)
        path_coords, dist = self.a_star(robot_location, nearest_utility_coords)
        guidepost = np.zeros_like(utility)
        for coords in path_coords:
            if coords[0] != robot_location[0] and coords[1] != robot_location[1]:
                index = np.argwhere(all_node_coords[:, 0] + all_node_coords[:, 1] * 1j == coords[0] + coords[1] * 1j)[0]
                guidepost[index] = 1

        dist_guidepost = np.zeros_like(utility)
        neigh = NearestNeighbors(n_neighbors=1)
        if len(utility_node_coords) > 0:
            neigh.fit(utility_node_coords)
            for coords in non_utility_node_coords:
                dist, _ = neigh.kneighbors([coords])
                index = np.argwhere(all_node_coords[:, 0] + all_node_coords[:, 1] * 1j == coords[0] + coords[1] * 1j)[0]
                dist_guidepost[index] = dist[0][0]

        # Guide post for ground belief       
        indices_ground = np.argwhere(utility_ground > 0).reshape(-1)
        utility_node_coords_ground = all_node_coords[indices_ground]
        non_utility_node_coords_ground = all_node_coords[np.argwhere(utility_ground == 0).reshape(-1)]
        # dist_dict, prev_dict = self.Dijkstra(robot_location)
        nearest_utility_coords_ground = robot_location
        nearest_dist_ground = 1e8
        for coords in utility_node_coords_ground:
            dist = dist_dict[(coords[0], coords[1])]
            if dist < nearest_dist_ground and dist != 0:
                nearest_dist_ground = dist
                nearest_utility_coords_ground = coords
        path_coords_ground, dist = self.a_star(robot_location, nearest_utility_coords_ground)
        # last_path_coords = path_coords_ground[-1]
        # last_index = np.argwhere(all_node_coords[:, 0] + all_node_coords[:, 1] * 1j == last_path_coords[0] + last_path_coords[1] * 1j)[0]
        # print(utility_ground[last_index])
        # assert utility_ground[last_index] > 0
        guidepost_ground = np.zeros_like(utility_ground)
        for coords in path_coords_ground:
            if coords[0] != robot_location[0] and coords[1] != robot_location[1]:
                index = np.argwhere(all_node_coords[:, 0] + all_node_coords[:, 1] * 1j == coords[0] + coords[1] * 1j)[0]
                guidepost_ground[index] = 1
                
        dist_guidepost_ground = np.zeros_like(utility)
        neigh_ground = NearestNeighbors(n_neighbors=1)
        if len(utility_node_coords_ground) > 0:
            neigh_ground.fit(utility_node_coords_ground)
            for coords in non_utility_node_coords_ground:
                dist, _ = neigh_ground.kneighbors([coords])
                index = np.argwhere(all_node_coords[:, 0] + all_node_coords[:, 1] * 1j == coords[0] + coords[1] * 1j)[0]
                dist_guidepost_ground[index] = dist[0][0]

        # Guide post for safe belief
        indices_safe = np.argwhere(utility_safe > 0).reshape(-1)
        utility_node_coords_safe = all_node_coords[indices_safe]
        # non_utility_node_coords_safe = all_node_coords[np.argwhere(utility_safe == 0).reshape(-1)]
        nearest_utility_coords_safe = robot_location
        nearest_dist_safe = 1e8
        for coords in utility_node_coords_safe:
            dist = dist_dict[(coords[0], coords[1])]
            if dist < nearest_dist_safe and dist != 0:
                nearest_dist_safe = dist
                nearest_utility_coords_safe = coords
        path_coords_safe, dist = self.a_star(robot_location, nearest_utility_coords_safe)
        guidepost_safe = np.zeros_like(utility_safe)
        for coords in path_coords_safe:
            if coords[0] != robot_location[0] and coords[1] != robot_location[1]:
                index = np.argwhere(all_node_coords[:, 0] + all_node_coords[:, 1] * 1j == coords[0] + coords[1] * 1j)[0]
                guidepost_safe[index] = 1

        # Guide post for unsafe belief
        indices_uncovered_safe = np.argwhere(utility_safe_uncovered > 0).reshape(-1)
        utility_node_coords_uncovered_safe  = all_node_coords[indices_uncovered_safe]
        # non_utility_node_coords_uncovered_safe  = all_node_coords[np.argwhere(utility_safe_uncovered == 0).reshape(-1)]
        nearest_utility_coords_uncovered_safe  = robot_location
        nearest_dist_uncovered_safe  = 1e8
        for coords in utility_node_coords_uncovered_safe :
            dist = dist_dict[(coords[0], coords[1])]
            if dist < nearest_dist_uncovered_safe  and dist != 0:
                nearest_dist_uncovered_safe  = dist
                nearest_utility_coords_uncovered_safe  = coords
        path_coords_uncovered_safe , dist = self.a_star(robot_location, nearest_utility_coords_uncovered_safe)
        guidepost_uncovered_safe  = np.zeros_like(utility_safe_uncovered)
        for coords in path_coords_uncovered_safe :
            if coords[0] != robot_location[0] and coords[1] != robot_location[1]:
                index = np.argwhere(all_node_coords[:, 0] + all_node_coords[:, 1] * 1j == coords[0] + coords[1] * 1j)[0]
                guidepost_uncovered_safe[index] = 1

        robot_in_graph = self.nodes_dict.nearest_neighbors(robot_location.tolist(), 1)[0].data.coords
        current_index = np.argwhere(node_coords_to_check == robot_in_graph[0] + robot_in_graph[1] * 1j)[0][0]
        neighbor_indices = np.argwhere(adjacent_matrix[current_index] == 0).reshape(-1)

        occupancy = np.zeros((n_nodes, 1))
        for location in robot_locations:
            location_in_graph = self.nodes_dict.find((location[0], location[1])).data.coords
            index = np.argwhere(node_coords_to_check == location_in_graph[0] + location_in_graph[1] * 1j)[0][0]
            if index == current_index:
                occupancy[index] = -1
            else:
                occupancy[index] = 1

        node_coords = all_node_coords
        node_utility = utility.reshape(-1, 1)
        node_utility_ground = utility_ground.reshape(-1, 1)
        node_utility_safe = utility_safe.reshape(-1, 1)
        node_utility_safe_uncovered = utility_safe_uncovered.reshape(-1, 1)
        node_safe_indicator = safe_indicator.reshape(-1, 1)
        node_occluded_indicator = occluded_indicator.reshape(-1, 1)
        node_explored_sign = explored_sign.reshape(-1, 1)
        node_guidepost = guidepost.reshape(-1, 1)
        node_guidepost_ground = guidepost_ground.reshape(-1, 1)
        node_dist_guidepost = dist_guidepost.reshape(-1, 1)
        node_dist_guidepost_ground = dist_guidepost_ground.reshape(-1, 1)
        node_guidepost_safe = guidepost_safe.reshape(-1, 1)
        node_guidepost_uncovered_safe = guidepost_uncovered_safe.reshape(-1, 1)
        node_occupancy = occupancy.reshape(-1, 1)

        # neighboring_robot_indices = np.where(node_occupancy==1)[0]
        current_index = current_index
        edge_mask = adjacent_matrix
        current_edge = neighbor_indices
        n_node = node_coords.shape[0]

        current_node_coords = node_coords[current_index]
        node_coords = np.concatenate((node_coords[:, 0].reshape(-1, 1) - current_node_coords[0],
                                      node_coords[:, 1].reshape(-1, 1) - current_node_coords[1]),
                                      axis=-1) / UPDATING_MAP_SIZE / 2
        
        # border_coords = get_coords_from_cell_position(self.border_points, self.ground_truth_map_info)
        # border_node_coords = np.concatenate((border_coords[:, 0].reshape(-1, 1) - current_node_coords[0],            # Scale to [-1, 1]
        #                                     border_coords[:, 1].reshape(-1, 1) - current_node_coords[1]),
        #                                    axis=-1) / UPDATING_MAP_SIZE / 2
        # border_node_coords = border_node_coords.flatten().reshape(1, -1)
        # border_node_coords = torch.FloatTensor(border_node_coords).unsqueeze(0).to(self.device)

        node_utility = node_utility / (2 * self.sensor_range * 3.14 // FRONTIER_CELL_SIZE)
        node_utility_ground = node_utility_ground / (2 * self.sensor_range * 3.14 // FRONTIER_CELL_SIZE)
        node_utility_safe = node_utility_safe / (2 * self.sensor_range * 3.14 // FRONTIER_CELL_SIZE)
        node_utility_safe_uncovered = node_utility_safe_uncovered / (2 * self.sensor_range * 3.14 // FRONTIER_CELL_SIZE)
        node_dist_guidepost = node_dist_guidepost / UPDATING_MAP_SIZE
        node_dist_guidepost_ground = node_dist_guidepost_ground / UPDATING_MAP_SIZE
        # node_border_indicator = np.clip(node_border_indicator, None, 4)

        node_inputs = np.concatenate((node_coords, node_occupancy, node_utility, node_utility_ground, node_guidepost, node_guidepost_ground,
                                      node_dist_guidepost, node_dist_guidepost_ground, node_utility_safe, node_utility_safe_uncovered,
                                     node_safe_indicator, node_occluded_indicator, node_guidepost_safe, node_guidepost_uncovered_safe, node_explored_sign), axis=1)
        # assert np.all((node_inputs >= -1) & (node_inputs <= 1)), f"Values not between -1 and 1: {np.where((node_inputs < -1) | (node_inputs > 1))}"
        node_inputs = torch.FloatTensor(node_inputs).unsqueeze(0).to(self.device)
        # assert node_coords.shape[0] < NODE_PADDING_SIZE, print(node_coords.shape[0], NODE_PADDING_SIZE)
        padding = torch.nn.ZeroPad2d((0, 0, 0, NODE_PADDING_SIZE - n_node))
        node_inputs = padding(node_inputs)

        node_padding_mask = torch.zeros((1, 1, n_node), dtype=torch.int16).to(self.device)
        node_padding = torch.ones((1, 1, NODE_PADDING_SIZE - n_node), dtype=torch.int16).to(
            self.device)
        node_padding_mask = torch.cat((node_padding_mask, node_padding), dim=-1)

        edge_mask = torch.tensor(edge_mask).unsqueeze(0).to(self.device)

        padding = torch.nn.ConstantPad2d(
            (0, NODE_PADDING_SIZE - n_node, 0, NODE_PADDING_SIZE - n_node), 1)
        edge_mask = padding(edge_mask)

        current_in_edge = np.argwhere(current_edge == current_index)[0][0]
        current_edge = torch.tensor(current_edge).unsqueeze(0)
        k_size = current_edge.size()[-1]
        padding = torch.nn.ConstantPad1d((0, K_SIZE - k_size), 0)
        current_edge = padding(current_edge)
        current_edge = current_edge.unsqueeze(-1)

        edge_padding_mask = torch.zeros((1, 1, k_size), dtype=torch.int16).to(self.device)
        edge_padding_mask[0, 0, current_in_edge] = 1
        padding = torch.nn.ConstantPad1d((0, K_SIZE - k_size), 1)
        edge_padding_mask = padding(edge_padding_mask)

        current_index = torch.tensor([current_index]).reshape(1, 1, 1).to(self.device)

        # neighboring_robot_indices = torch.tensor(neighboring_robot_indices).reshape(1, -1, 1).to(self.device)

        return [node_inputs, node_padding_mask, edge_mask, current_index, current_edge, edge_padding_mask]
    
    def Dijkstra(self, start):
        q = set()
        dist_dict = {}
        prev_dict = {}

        for node in self.nodes_dict.__iter__():
            coords = node.data.coords
            key = (coords[0], coords[1])
            dist_dict[key] = 1e8
            prev_dict[key] = None
            q.add(key)

        # assert (start[0], start[1]) in dist_dict.keys()
        dist_dict[(start[0], start[1])] = 0

        while len(q) > 0:
            u = None
            for coords in q:
                if u is None:
                    u = coords
                elif dist_dict[coords] < dist_dict[u]:
                    u = coords

            q.remove(u)

            if self.nodes_dict.find(u) is None:
                print(u)
                for node in self.nodes_dict.__iter__():
                    print(node.data.coords)

            node = self.nodes_dict.find(u).data
            for neighbor_node_coords in node.neighbor_list:
                v = (neighbor_node_coords[0], neighbor_node_coords[1])
                if v in q:
                    cost = ((neighbor_node_coords[0] - u[0]) ** 2 + (
                            neighbor_node_coords[1] - u[1]) ** 2) ** (1 / 2)
                    cost = np.round(cost, 2)
                    alt = dist_dict[u] + cost
                    if alt < dist_dict[v]:
                        dist_dict[v] = alt
                        prev_dict[v] = u

        return dist_dict, prev_dict
    
    def h(self, coords_1, coords_2):
        # h = abs(coords_1[0] - coords_2[0]) + abs(coords_1[1] - coords_2[1])
        h = ((coords_1[0] - coords_2[0]) ** 2 + (coords_1[1] - coords_2[1]) ** 2) ** (1 / 2)
        h = np.round(h, 2)
        return h
    
    def a_star(self, start, destination, boundary=None, max_dist=None):
        # the path does not include the start
        if not self.check_node_exist_in_dict(start):
            print(start)
            Warning("start position is not in node dict")
            return [], 1e8
        if not self.check_node_exist_in_dict(destination):
            Warning("end position is not in node dict")
            return [], 1e8

        if start[0] == destination[0] and start[1] == destination[1]:
            return [destination], 0

        open_list = {(start[0], start[1])}
        closed_list = set()
        g = {(start[0], start[1]): 0}
        parents = {(start[0], start[1]): (start[0], start[1])}

        while len(open_list) > 0:
            n = None
            h_n = 1e8

            for v in open_list:
                h_v = self.h(v, destination)
                if n is not None:
                    node = self.nodes_dict.find(n).data
                    n_coords = node.coords
                    h_n = self.h(n_coords, destination)
                if n is None or g[v] + h_v < g[n] + h_n:
                    n = v
                    node = self.nodes_dict.find(n).data
                    n_coords = node.coords

            if max_dist is not None:
                if g[n] > max_dist:
                    return [], 1e8

            if n_coords[0] == destination[0] and n_coords[1] == destination[1]:
                path = []
                length = g[n]
                while parents[n] != n:
                    path.append(n)
                    n = parents[n]
                path.reverse()
                return path, np.round(length, 2)

            for neighbor_node_coords in node.neighbor_list:
                if self.nodes_dict.find(neighbor_node_coords) is None:
                    continue
                if boundary is not None:
                    if not (boundary[0] < neighbor_node_coords[0] < boundary[2] and boundary[1] < neighbor_node_coords[1] < boundary[3]):
                        continue
                cost = ((neighbor_node_coords[0] - n_coords[0]) ** 2 + (
                            neighbor_node_coords[1] - n_coords[1]) ** 2) ** (1 / 2)
                cost = np.round(cost, 2)
                m = (neighbor_node_coords[0], neighbor_node_coords[1])
                if m not in open_list and m not in closed_list:
                    open_list.add(m)
                    parents[m] = n
                    g[m] = g[n] + cost
                else:
                    if g[m] > g[n] + cost:
                        g[m] = g[n] + cost
                        parents[m] = n

                        if m in closed_list:
                            closed_list.remove(m)
                            open_list.add(m)
            open_list.remove(n)
            closed_list.add(n)

        print('Path does not exist!')

        return [], 1e8

    def add_node_to_dict(self, coords):
        key = (coords[0], coords[1])
        node = Node(coords, self.sensor_range, self.robot_height, self.max_height)
        self.nodes_dict.insert(point=key, data=node)
        return node
    
    def remove_node_from_dict(self, node):
        for neighbor_coords in node.data.neighbor_list[1:]:
            neighbor_node = self.nodes_dict.find(neighbor_coords)
            neighbor_node.data.neighbor_list.remove(tuple(node.data.coords))
        self.nodes_dict.remove(tuple(node.data.coords))

    def initialize_graph(self):
        node_coords = self.get_ground_truth_node_coords(self.ground_truth_map_info)

        # plt.imshow(self.ground_truth_map_info.map, cmap='gray')
        # nodes_cell = get_cell_position_from_coords(node_coords, self.ground_truth_map_info)
        # plt.scatter(nodes_cell[:, 0], nodes_cell[:, 1], zorder=2, s=25)
        # for i, txt in enumerate(node_coords):
        #     plt.annotate(txt, (nodes_cell[i, 0], nodes_cell[i,1]), size=6, c='white', zorder =10)
        # for i, txt in enumerate(nodes_cell):
        #     plt.annotate(txt, (nodes_cell[i, 0] + 2, nodes_cell[i,1] + 2), size=6, c='blue', zorder =10)

        # plt.show()
        
        for coords in node_coords:
            self.add_node_to_dict(coords)

        plot_x = self.x if self.plot else None
        plot_y = self.y if self.plot else None  

        for node in self.nodes_dict.__iter__():
            node.data.get_neighbor_nodes(self.ground_truth_map_info, self.nodes_dict, plot_x, plot_y)
        self.remove_small_components()

        all_coords = []
        for node in self.nodes_dict.__iter__():
           all_coords.append(node.data.coords)
        all_coords = np.array(all_coords)
        self.ground_truth_node_coords = all_coords
    def find_connected_components(self):
        # Create adjacency list representation
        adj_list = {}
        for node in self.nodes_dict.__iter__():
            coords = tuple(node.data.coords)
            adj_list[coords] = [tuple(neighbor) for neighbor in node.data.neighbor_list[1:]]
        
        # DFS to find components
        visited = set()
        components = []
        
        def dfs(vertex, component):
            visited.add(vertex)
            component.append(vertex)
            for neighbor in adj_list[vertex]:
                if neighbor not in visited:
                    dfs(neighbor, component)

        # Find all components
        for vertex in adj_list:
            if vertex not in visited:
                component = []
                dfs(vertex, component)
                components.append(component)
        return components
    
    def remove_small_components(self, min_size_ratio=1):
        components = self.find_connected_components()
        
        # Find largest component size
        sizes = [len(c) for c in components]
        largest_size = max(sizes)
        min_size = largest_size * min_size_ratio  # Only keep largest size
        
        # Remove nodes from small components
        for component in components:
            if len(component) < min_size:
                for coords in component:
                    node = self.nodes_dict.find(coords)
                    if node is not None:
                        self.remove_node_from_dict(node)
                        # print('Remove {} node from GT graph (DFS)'.format(node.data.coords))

    def update_graph(self):
        for node in self.node_manager.nodes_dict.__iter__():
            coords = node.data.coords
            ground_truth_node = self.nodes_dict.find(coords.tolist())
            # assert ground_truth_node is not None, print(coords)
            ground_truth_node.data.utility = node.data.utility
            ground_truth_node.data.utility_ground = node.data.utility_ground
            ground_truth_node.data.utility_safe = node.data.utility_safe
            ground_truth_node.data.utility_safe_uncovered = node.data.utility_safe_uncovered
            ground_truth_node.data.safe_indicator = node.data.safe
            ground_truth_node.data.occluded_indicator = node.data.occluded
            ground_truth_node.data.explored = 1

    def get_ground_truth_node_coords(self, ground_truth_map_info):
        x_min = ground_truth_map_info.map_origin_x
        y_min = ground_truth_map_info.map_origin_y
        x_max = ground_truth_map_info.map_origin_x + (ground_truth_map_info.map.shape[1] - 1) * CELL_SIZE
        y_max = ground_truth_map_info.map_origin_y + (ground_truth_map_info.map.shape[0] - 1) * CELL_SIZE

        if x_min % NODE_RESOLUTION != 0:
            x_min = (x_min // NODE_RESOLUTION + 1) * NODE_RESOLUTION
        if x_max % NODE_RESOLUTION != 0:
            x_max = x_max // NODE_RESOLUTION * NODE_RESOLUTION
        if y_min % NODE_RESOLUTION != 0:
            y_min = (y_min // NODE_RESOLUTION + 1) * NODE_RESOLUTION
        if y_max % NODE_RESOLUTION != 0:
            y_max = y_max // NODE_RESOLUTION * NODE_RESOLUTION

        x_coords = np.arange(x_min, x_max + 0.1, NODE_RESOLUTION)
        y_coords = np.arange(y_min, y_max + 0.1, NODE_RESOLUTION)
        t1, t2 = np.meshgrid(x_coords, y_coords)
        nodes = np.vstack([t1.T.ravel(), t2.T.ravel()]).T
        nodes = np.around(nodes, 1)

        indices = []
        nodes_cells = get_cell_position_from_coords(nodes, ground_truth_map_info).reshape(-1, 2)
        for i, cell in enumerate(nodes_cells):
            # assert 0 <= cell[1] < ground_truth_map_info.map.shape[0] and 0 <= cell[0] < ground_truth_map_info.map.shape[1]
            if (ground_truth_map_info.map[cell[1], cell[0]] >= 0) and (ground_truth_map_info.map[cell[1], cell[0]] < self.robot_height_cell):
                indices.append(i)
        indices = np.array(indices)
        nodes = nodes[indices].reshape(-1, 2)

        return nodes

    def plot_ground_truth_env(self, robot_location):
        plt.subplot(1, 3, 3)
        plt.imshow(self.ground_truth_map_info.map, cmap='gray')
        plt.axis('off')
        robot = get_cell_position_from_coords(robot_location, self.ground_truth_map_info)
        nodes = get_cell_position_from_coords(self.ground_truth_node_coords, self.ground_truth_map_info)
        plt.imshow(self.ground_truth_map_info.map, cmap='gray')
        plt.scatter(nodes[:, 0], nodes[:, 1], c=self.explored_sign, zorder=2)
        plt.plot(robot[0], robot[1], 'mo', markersize=16, zorder=5)
        plt.close()

class Node:
    def __init__(self, coords, sensor_range, height, max_height_cell):
        self.coords = coords
        self.robot_height = height     # in meters
        self.max_height_cell = max_height_cell
        self.utility = -(2 * sensor_range * 3.14 // FRONTIER_CELL_SIZE)
        self.utility_ground = -(2 * sensor_range * 3.14 // FRONTIER_CELL_SIZE)
        self.utility_safe = -(2 * sensor_range * 3.14 // FRONTIER_CELL_SIZE)
        self.utility_safe_uncovered = -(2 * sensor_range * 3.14 // FRONTIER_CELL_SIZE)
        self.safe_indicator = -1
        self.occluded_indicator = -1
        self.explored = 0

        self.neighbor_matrix = -np.ones((NUM_NODE_NEIGHBORS, NUM_NODE_NEIGHBORS))
        self.neighbor_list = []
        self.neighbor_list.append((self.coords[0], self.coords[1]))
        
    def get_neighbor_nodes(self, ground_truth_map_info, nodes_dict, plot_x=None, plot_y=None):
        for i in range(self.neighbor_matrix.shape[0]):
            for j in range(self.neighbor_matrix.shape[1]):
                if self.neighbor_matrix[i, j] != -1:
                    continue
                else:
                    center_index = self.neighbor_matrix.shape[0] // 2
                    if i == center_index and j == center_index:
                        self.neighbor_matrix[i, j] = 1
                        continue

                    neighbor_coords = np.around(np.array([self.coords[0] + (i - center_index) * NODE_RESOLUTION,
                                                self.coords[1] + (j - center_index) * NODE_RESOLUTION]), 1)
                    neighbor_node = nodes_dict.find((neighbor_coords[0], neighbor_coords[1]))
                    if neighbor_node is None:
                        continue
                    elif (neighbor_node.data.coords[0] <= ground_truth_map_info.map_origin_x
                          or neighbor_node.data.coords[1] <= ground_truth_map_info.map_origin_y
                          or neighbor_node.data.coords[0] >= ground_truth_map_info.map_origin_x +
                          ground_truth_map_info.map.shape[1] * ground_truth_map_info.cell_size
                          or neighbor_node.data.coords[1] >= ground_truth_map_info.map_origin_y +
                          ground_truth_map_info.map.shape[0] * ground_truth_map_info.cell_size):
                        continue
                    else:

                        collision = check_collision(self.coords, neighbor_coords, self.robot_height, ground_truth_map_info, self.max_height_cell)

                        if not collision:
                            self.neighbor_matrix[i, j] = 1
                            self.neighbor_list.append((neighbor_coords[0], neighbor_coords[1]))
                            if plot_x != None and plot_y != None:
                                plot_x.append([self.coords[0], neighbor_coords[0]])
                                plot_y.append([self.coords[1], neighbor_coords[1]])
                            neighbor_matrix_x = center_index + (center_index - i)
                            neighbor_matrix_y = center_index + (center_index - j)
                            neighbor_node = neighbor_node.data
                            neighbor_node.neighbor_matrix[neighbor_matrix_x, neighbor_matrix_y] = 1
                            neighbor_node.neighbor_list.append((self.coords[0], self.coords[1]))
