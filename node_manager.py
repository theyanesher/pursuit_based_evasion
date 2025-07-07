import time

import numpy as np
from utils import *
from parameter import *
import quads
# import copy
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from scipy.spatial import KDTree
from collections import deque

class Node_manager:
    def __init__(self, robot_height, robot_height_cell, max_height_cell, quad_tree_size=None, sensor_range = None, plot=False):
        self.nodes_dict = quads.QuadTree((0, 0), quad_tree_size, quad_tree_size)
        self.plot = plot
        self.robot_height = robot_height   # in metres
        self.robot_height_cell = robot_height_cell
        self.max_height_cell = max_height_cell
        self.sensor_range = sensor_range
        self.invalid_node_list = []
        self.all_nodes = np.array([]).reshape(-1, 2)

        if self.plot:
            self.x = []
            self.y = []

    def check_node_exist_in_dict(self, coords):
        key = (coords[0], coords[1])
        exist = self.nodes_dict.find(key)
        return exist

    def add_node_to_dict(self, coords, local_frontiers, local_frontiers_ground, safe_frontiers, uncovered_safe_frontiers, updating_flight_info, safe, occluded):
        key = (coords[0], coords[1])
        node = Node(coords, local_frontiers, local_frontiers_ground, safe_frontiers, uncovered_safe_frontiers, updating_flight_info, self.robot_height, self.sensor_range, self.max_height_cell, safe, occluded)   
        self.nodes_dict.insert(point=key, data=node)
        return node
    
    def remove_node_from_dict(self, node):
        for neighbor_coords in node.neighbor_list[1:]:
            neighbor_node = self.nodes_dict.find(neighbor_coords)
            neighbor_node.data.neighbor_list.remove(tuple(node.coords))
        self.nodes_dict.remove(tuple(node.coords))

    def check_invalid_node(self, check_coord):
        if len(self.invalid_node_list) == 0:
            return False
        else:
            is_in_list = any(np.array_equal(check_coord, coord) for coord in self.invalid_node_list)
        return is_in_list

    def fast_bfs_split(self, points, max_size, radius):
        points = np.array(points)
        tree = KDTree(points)
        N = len(points)
        visited = np.zeros(N, dtype=bool)
        subclusters = []

        # Default: farthest from centroid (peripheral points)
        centroid = np.mean(points, axis=0)
        dists = np.linalg.norm(points - centroid, axis=1)
        sorted_indices = np.argsort(-dists)  # start with farthest points

        for i in sorted_indices:
            if visited[i]:
                continue
            queue = deque([i])
            current_cluster = []

            while queue and len(current_cluster) < max_size:
                idx = queue.popleft()
                if visited[idx]:
                    continue
                visited[idx] = True
                current_cluster.append(tuple(points[idx]))

                if len(current_cluster) >= max_size:
                    break

                neighbors = tree.query_ball_point(points[idx], radius)
                for n_idx in neighbors:
                    if not visited[n_idx]:
                        queue.append(n_idx)

            subclusters.append(current_cluster)

        return subclusters

    def update_graph(self, robot_location, local_frontiers, local_frontiers_ground, local_frontiers_safe, uncovered_safe_frontiers, updating_flight_info, 
                     updating_ground_info, updating_safe_info, robot_locations=None, k=None):
        node_coords, _ = get_updating_node_coords(robot_location, updating_flight_info, self.robot_height_cell, check_connectivity=False)
        # node_cells = get_cell_position_from_coords(node_coords, updating_flight_info)
        skeleton_coords = get_nodes(updating_flight_info)

        nbrs_sk = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(skeleton_coords)
        sk_distances, _ = nbrs_sk.kneighbors(skeleton_coords)
        mask_sk = np.sum(sk_distances, axis=1, keepdims=True).flatten() >= 10
        skeleton_coords = skeleton_coords[mask_sk]

        # Calculate distances between node_cells and skeleton_cells
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(skeleton_coords)
        distances, _ = nbrs.kneighbors(node_coords)
        
        # Create mask for points that are far enough from skeleton
        mask = distances.flatten() >= 8
        
        # Filter node_cells and node_coords
        node_coords_filtered = node_coords[mask]

        node_coords= np.concatenate((node_coords_filtered, skeleton_coords), axis=0)
        
        # skeleton_cells = get_cell_position_from_coords(skeleton_coords, updating_flight_info)
        # node_cells_new = get_cell_position_from_coords(node_coords_new, updating_flight_info)
        # node_cells = get_cell_position_from_coords(node_coords, updating_flight_info)
        # plt.subplot(1,2,1)
        # plt.imshow(updating_flight_info.map, cmap='gray')
        # plt.scatter(node_cells[:, 0], node_cells[:, 1], c='r')
        # plt.subplot(1,2,2)
        # plt.imshow(updating_flight_info.map , cmap='gray')
        # plt.scatter(skeleton_cells[:, 0], skeleton_cells[:, 1], c='b')
        # plt.scatter(node_cells_new[:, 0], node_cells_new[:, 1], c='r')
        # plt.show()

        dists = np.linalg.norm(node_coords[:, None, :] - self.all_nodes[None, :, :], axis=2) 
        # Check if any point in a is within 5 units of any point in b
        mask = mask = ~np.any(dists < 5, axis=1)

        node_coords = node_coords[mask]

        set_filtered_node_coords = set(map(tuple, node_coords))
        set_all_nodes = set(map(tuple, self.all_nodes))

        # Find elements in b that are NOT in a
        new_elements = np.array(list(set_all_nodes - set_filtered_node_coords))

        # Append only if there are new elements
        if new_elements.shape[0] > 0:
            node_coords = np.vstack((node_coords, new_elements))

        if robot_locations is not None:
            node_coords = np.concatenate((node_coords, robot_locations), axis=0)
        
        node_cells = get_cell_position_from_coords(node_coords, updating_flight_info)

        all_node_list = []
        for i, coords in enumerate(node_coords):
            node = self.check_node_exist_in_dict(coords)  # Return Point object
            invalid = self.check_invalid_node(coords)
 
            if node is None and not invalid:
                safe = 0
                occluded = 0
                if updating_safe_info.map[node_cells[i][1], node_cells[i][0]] == 255:
                    safe = 1
                if updating_ground_info.map[node_cells[i][1], node_cells[i][0]] == -1:
                    occluded = 1
                if updating_ground_info.map[node_cells[i][1], node_cells[i][0]] > 0:    
                    occluded = 0.5
                # elif updating_flight_info.map[node_cells[i][1], node_cells[i][0]] > 0:
                #     safe = 0.5

                node = self.add_node_to_dict(coords, local_frontiers, local_frontiers_ground, local_frontiers_safe, uncovered_safe_frontiers, updating_flight_info, safe, occluded)

            elif node is not None and not invalid:
                node = node.data 
                node.visited = 0
                # Update flight data
                if node.utility == 0 or np.linalg.norm(node.coords - robot_location) > 2 * self.sensor_range:
                    pass
                else:
                    node.update_node_observable_frontiers(local_frontiers, updating_flight_info)

                # Update ground data
                # Assume that occlusion is not considered for ground data
                if updating_ground_info.map[node_cells[i][1], node_cells[i][0]] == -1:
                    node.set_occluded()
                elif updating_ground_info.map[node_cells[i][1], node_cells[i][0]] > 0:
                    node.set_building_ground()
                else:
                    if np.linalg.norm(node.coords - robot_location) > 2 * self.sensor_range:
                        pass
                    else:
                        node.set_unoccluded()
                        node.update_node_observable_frontiers_ground(local_frontiers_ground, updating_flight_info)

                # Update safe data
                if updating_safe_info.map[node_cells[i][1], node_cells[i][0]] == 255:
                    # plt.imshow(updating_flight_info.map, cmap='gray')
                    # alpha_mask = updating_safe_info.map / 255 / 3
                    # plt.imshow(updating_safe_info.map, cmap='Greens', alpha=alpha_mask)
                    # plt.scatter(node_cells[i][0], node_cells[i][1], c='r')
                    # plt.show()
                    node.set_safe()
                    if np.linalg.norm(node.coords - robot_location) > 2 * self.sensor_range:
                        pass
                    else:
                        node.update_observable_safe_frontiers(local_frontiers_safe, uncovered_safe_frontiers, updating_flight_info)
                elif updating_ground_info.map[node_cells[i][1], node_cells[i][0]] > 0:
                    node.set_safe_building()
                else:
                    # plt.imshow(updating_flight_info.map, cmap='gray')
                    # alpha_mask = updating_safe_info.map / 255 / 3
                    # plt.imshow(updating_safe_info.map, cmap='Greens', alpha=alpha_mask)
                    # plt.scatter(node_cells[i][0], node_cells[i][1], c='r')
                    # plt.show()
                    node.set_unsafe()
            else:
                pass

            if not invalid:
                all_node_list.append(node)

        plot_x = self.x if self.plot else None
        plot_y = self.y if self.plot else None

        for node in all_node_list:
            # print(len(node.neighbor_list))
            # if len(node.neighbor_list) == 1:
            #     print(node.neighbor_list)
            #     self.remove_node_from_dict(node)
            # node_check = ground_truth_manager.nodes_dict.find((node.coords[0], node.coords[1]))
            # if node_check is None:
            #     continue
                # self.remove_node_from_dict(node)
                # self.invalid_node_list.append(node.coords)
                # print('Node {} not found in ground truth graph'.format(node.coords))
            # else:
                if node.need_update_neighbor and np.linalg.norm(node.coords - robot_location) < (
                        self.sensor_range + NODE_RESOLUTION * 3):
                    node.update_neighbor_nodes_2(updating_flight_info, self.nodes_dict, plot_x, plot_y)

        for node in all_node_list:
            if len(node.neighbor_list) == 1:
                self.remove_node_from_dict(node)
                
        # Update frontier clusters
        if k == 0 and len(local_frontiers_safe) > 0:
            # Apply DBSCAN to cluster the points based on distance
            safe_frontiers = np.array(list(local_frontiers_safe)).reshape(-1, 2)
            dist = (2 * ((FRONTIER_CELL_SIZE)**2))**0.5 
            frontier_clusters = DBSCAN(eps=dist, min_samples=1).fit(safe_frontiers)
            labels = frontier_clusters.labels_

            # Group points by their labels
            clusters = {}
            old_clusters = {}
            new_label = 0
            for label in np.unique(labels):
                if label == -1:  # -1 is the label for noise points
                    print('Got noises!')
                    continue
                cluster_points = safe_frontiers[labels == label]
                old_clusters[label] = list(map(tuple, cluster_points))

            for label in np.unique(labels):
                if label == -1:
                    continue
                cluster_points = safe_frontiers[labels == label]
                if len(cluster_points) <= MAX_CLUSTER_SIZE:
                    clusters[new_label] = list(map(tuple, cluster_points))
                    new_label += 1
                else:
                    # Efficient splitting using KDTree-based BFS
                    subclusters = self.fast_bfs_split(cluster_points, MAX_CLUSTER_SIZE, dist)
                    for sub in subclusters:
                        clusters[new_label] = sub
                        new_label += 1

            centroid_info = []
            size_info = []
            spread_info = []
            # overall_spread_info = []
            uncovered_safe_utility_info = []
            for label, cluster in clusters.items():
                uncovered_safe_utility = 0
                points_arr = np.array(cluster)
                # Compute centroid as the mean of the points
                centroid = np.mean(points_arr, axis=0)
                
                # Compute spread as the standard deviation along each axis
                spread = np.std(points_arr, axis=0)
                
                # Optionally, compute an overall spread (Euclidean norm of the spread)
                # overall_spread = np.linalg.norm(spread)
                # for point in points_arr:
                #     if (point[0], point[1]) in uncovered_safe_frontiers:
                #         uncovered_safe_utility += 1

                centroid_info.append(centroid)
                size_info.append(len(cluster)/MAX_CLUSTER_SIZE)
                spread_info.append(spread/MAX_CLUSTER_SIZE*dist)
                # overall_spread_info.append(overall_spread/MAX_CLUSTER_SIZE*dist)
                uncovered_safe_utility_info.append(uncovered_safe_utility/len(cluster))
            
            centroid_info = np.array(centroid_info)
            size_info = np.array(size_info).reshape(-1, 1)
            spread_info = np.array(spread_info)
            # overall_spread_info = np.array(overall_spread_info)
            uncovered_safe_utility_info = np.array(uncovered_safe_utility_info).reshape(-1, 1)
            self.frontier_centroids = centroid_info
            self.frontier_clusters = np.concatenate((size_info, spread_info, uncovered_safe_utility_info), axis=1)

            # Plot the clusters for safe belief
            # import matplotlib.pyplot as plt
            # plt.figure()
            # plt.subplot(1, 2, 1)
            # plt.imshow(updating_safe_info.map, cmap='gray')
            # # Generate a color map with as many colors as clusters
            # cmap = plt.cm.get_cmap("viridis", len(clusters))
            # for i, cluster in enumerate(clusters.values()):
            #     points_cell = get_cell_position_from_coords(np.array(cluster), updating_flight_info).reshape(-1, 2)
            #     plt.scatter(points_cell[:, 0], points_cell[:, 1], color=cmap(i))
            #     for j, point in enumerate(points_cell):
            #         plt.annotate(f"{i}", point, textcoords="offset points", xytext=(5,5), ha='center', color=cmap(i))
            # plt.title("Safe Belief Frontier Clusters")
            # plt.xlabel("X Coordinate")  
            # plt.ylabel("Y Coordinate")
            # plt.subplot(1, 2, 2)
            # plt.imshow(updating_safe_info.map, cmap='gray')
            # # Generate a color map with as many colors as clusters
            # cmap = plt.cm.get_cmap("viridis", len(old_clusters))
            # for i, cluster in enumerate(old_clusters.values()):
            #     points_cell = get_cell_position_from_coords(np.array(cluster).reshape(-1, 2), updating_flight_info).reshape(-1, 2)
            #     plt.scatter(points_cell[:, 0], points_cell[:, 1], color=cmap(i), label=f"Cluster {i}")
            # plt.title("Safe Belief Frontier Clusters")
            # plt.xlabel("X Coordinate")
            # plt.ylabel("Y Coordinate")
            # plt.legend()
            # plt.show()

    def update_past_locations(self, past_locations):
        length = len(past_locations)
        for i, past_location in enumerate(past_locations):

            nei = self.nodes_dict.nearest_neighbors((past_location[0], past_location[1]), 1)
            if len(nei) == 0:
                continue
            node_in_graph = self.nodes_dict.nearest_neighbors((past_location[0], past_location[1]), 1)[0].data
            current_value = (i + 1)/length
            if node_in_graph.visited < current_value:
                node_in_graph.visited = current_value
            
    def get_all_node_graph(self, robot_location, robot_locations, past_locations):
        all_node_coords = []
        for node in self.nodes_dict.__iter__():
            all_node_coords.append(node.data.coords)
            
        all_node_coords = np.array(all_node_coords).reshape(-1, 2)
        utility = []
        utility_ground = []
        utility_safe = []
        utility_uncovered_safe = []
        safe_indicator = []
        occluded_indicator = []
        visitations = []
        
        # Correct robot location
        nei = self.nodes_dict.nearest_neighbors((robot_location[0], robot_location[1]), 1)
        if len(nei) > 0:
            robot_location = self.nodes_dict.nearest_neighbors((robot_location[0], robot_location[1]), 1)[0].data.coords
        for i, location in enumerate(robot_locations):
            nei = self.nodes_dict.nearest_neighbors((robot_location[0], robot_location[1]), 1)
            if len(nei) > 0:
                location_in_graph = self.nodes_dict.nearest_neighbors((location[0], location[1]), 1)[0].data.coords
                robot_locations[i] = location_in_graph

        n_nodes = all_node_coords.shape[0]
        adjacent_matrix = np.ones((n_nodes, n_nodes)).astype(int)
        node_coords_to_check = all_node_coords[:, 0] + all_node_coords[:, 1] * 1j
        for i, coords in enumerate(all_node_coords):
            node = self.nodes_dict.find((coords[0], coords[1])).data
            utility.append(node.utility)
            utility_ground.append(node.utility_ground)
            utility_safe.append(node.utility_safe)
            utility_uncovered_safe.append(node.utility_safe_uncovered)
            safe_indicator.append(node.safe)
            occluded_indicator.append(node.occluded)
            visitations.append(node.visited)

            for neighbor in node.neighbor_list:
                index = np.argwhere(node_coords_to_check == neighbor[0] + neighbor[1] * 1j)
                if index or index == [[0]]:
                    index = index[0][0]
                    adjacent_matrix[i, index] = 0

        utility = np.array(utility)
        utility_ground = np.array(utility_ground)
        utility_safe = np.array(utility_safe)
        utility_uncovered_safe = np.array(utility_uncovered_safe)
        safe_indicator = np.array(safe_indicator)
        occluded_indicator = np.array(occluded_indicator)
        visitations = np.array(visitations)
        past_locations_index = []

        # for past_location in past_locations:
        #     index = np.argwhere(node_coords_to_check == past_location[0] + past_location[1] * 1j)[0][0]  
        #     past_locations_index.append(index)
        #     visitations[index] *= -1  

        for past_location in past_locations:
            if len(np.argwhere(node_coords_to_check == past_location[0] + past_location[1] * 1j)) > 0:
                index = np.argwhere(node_coords_to_check == past_location[0] + past_location[1] * 1j)[0][0]
                past_locations_index.append(index)

        # if len(past_locations_index) < TRAJECTORY_LEN:
        past_locations_index = np.pad(past_locations_index, (0, TRAJECTORY_LEN - len(past_locations_index)), 'constant', constant_values=-1)

        # Guidepost for flight belief
        indices = np.argwhere(utility > 0).reshape(-1)
        utility_node_coords = all_node_coords[indices]
        non_utility_node_coords = all_node_coords[np.argwhere(utility == 0).reshape(-1)]
        dist_dict, _ = self.Dijkstra(robot_location)
        nearest_utility_coords = robot_location
        nearest_dist = 1e8
        for coords in utility_node_coords:
            dist = dist_dict[(coords[0], coords[1])]
            if dist < nearest_dist and dist != 0:
                nearest_dist = dist
                nearest_utility_coords = coords
        path_coords, dist = self.a_star(robot_location, nearest_utility_coords)
        # last_path_coords = path_coords[-1]
        # last_index = np.argwhere(all_node_coords[:, 0] + all_node_coords[:, 1] * 1j == last_path_coords[0] + last_path_coords[1] * 1j)[0]
        # print(utility[last_index], nearest_utility_coords, path_coords)
        # assert utility[last_index] > 0
        guidepost = np.zeros_like(utility)
        for coords in path_coords:
            if coords[0] != robot_location[0] and coords[1] != robot_location[1]:
                index = np.argwhere(all_node_coords[:, 0] + all_node_coords[:, 1] * 1j == coords[0] + coords[1] * 1j)[0]
                guidepost[index] = 1
        
        # dist_guidepost = np.zeros_like(utility)
        # neigh = NearestNeighbors(n_neighbors=1)
        # if len(utility_node_coords) > 0:
        #     neigh.fit(utility_node_coords)
        #     for coords in non_utility_node_coords:
        #         dist, _ = neigh.kneighbors([coords])
        #         index = np.argwhere(all_node_coords[:, 0] + all_node_coords[:, 1] * 1j == coords[0] + coords[1] * 1j)[0]
        #         dist_guidepost[index] = dist[0][0]
        
        # Guide post for ground belief       
        indices_ground = np.argwhere(utility_ground > 0).reshape(-1)
        utility_node_coords_ground = all_node_coords[indices_ground]
        non_utility_node_coords_ground = all_node_coords[np.argwhere(utility_ground == 0).reshape(-1)]
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
                
        # dist_guidepost_ground = np.zeros_like(utility)
        # neigh_ground = NearestNeighbors(n_neighbors=1)
        # if len(utility_node_coords_ground) > 0:
        #     neigh_ground.fit(utility_node_coords_ground)
        #     for coords in non_utility_node_coords_ground:
        #         dist, _ = neigh_ground.kneighbors([coords])
        #         index = np.argwhere(all_node_coords[:, 0] + all_node_coords[:, 1] * 1j == coords[0] + coords[1] * 1j)[0]
        #         dist_guidepost_ground[index] = dist[0][0]

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

        # Guide post for uncovered safe belief
        indices_uncovered_safe = np.argwhere(utility_uncovered_safe > 0).reshape(-1)
        utility_node_coords_uncovered_safe = all_node_coords[indices_uncovered_safe]
        # non_utility_node_coords_uncovered_safe = all_node_coords[np.argwhere(utility_uncovered_safe == 0).reshape(-1)]
        nearest_utility_coords_uncovered_safe = robot_location
        nearest_dist_uncovered_safe = 1e8
        for coords in utility_node_coords_uncovered_safe:
            dist = dist_dict[(coords[0], coords[1])]
            if dist < nearest_dist_uncovered_safe and dist != 0:
                nearest_dist_uncovered_safe = dist
                nearest_utility_coords_uncovered_safe = coords
        path_coords_uncovered_safe, dist = self.a_star(robot_location, nearest_utility_coords_uncovered_safe)

        guidepost_uncovered_safe = np.zeros_like(utility_uncovered_safe)
        for coords in path_coords_uncovered_safe:
            if coords[0] != robot_location[0] and coords[1] != robot_location[1]:
                index = np.argwhere(all_node_coords[:, 0] + all_node_coords[:, 1] * 1j == coords[0] + coords[1] * 1j)[0]
                guidepost_uncovered_safe[index] = 1
                
        # robot_in_graph = self.nodes_dict.nearest_neighbors(robot_location.tolist(), 1)[0].data.coords
        current_index = np.argwhere(node_coords_to_check == robot_location[0] + robot_location[1] * 1j)[0][0]       # Identify current robot node index
        neighbor_indices = np.argwhere(adjacent_matrix[current_index] == 0).reshape(-1)                             # Identify neighbor nodes

        occupancy = np.zeros((n_nodes, 1))
        for location in robot_locations:
            # location_in_graph = self.nodes_dict.nearest_neighbors((location[0], location[1]), 1)[0].data.coords
            index = np.argwhere(node_coords_to_check == location[0] + location[1] * 1j)[0][0]
            if index == current_index:
                occupancy[index] = -1
            else:
                occupancy[index] = 1

        #assert sum(occupancy) == 2, print(robot_locations)  # only for 4 robots
        path_coords = np.array(path_coords).reshape(-1, 2)
        path_coords_ground = np.array(path_coords_ground).reshape(-1, 2)

        self.all_nodes = all_node_coords
        return all_node_coords, utility, guidepost, occupancy, adjacent_matrix, current_index, neighbor_indices, utility_ground, guidepost_ground, path_coords_ground, utility_safe, utility_uncovered_safe, safe_indicator, occluded_indicator, guidepost_safe, guidepost_uncovered_safe, visitations, past_locations_index, self.frontier_centroids, self.frontier_clusters
    
    def h(self, coords_1, coords_2):
        # h = abs(coords_1[0] - coords_2[0]) + abs(coords_1[1] - coords_2[1])
        h = ((coords_1[0] - coords_2[0]) ** 2 + (coords_1[1] - coords_2[1]) ** 2) ** (1 / 2)
        h = np.round(h, 2)
        return h

    def a_star(self, start, destination, max_dist=1e8):
        # the path does not include the start
        if not self.check_node_exist_in_dict(start):
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

            # if g[n] > max_dist:
            #     return [], 1e8

            if n_coords[0] == destination[0] and n_coords[1] == destination[1]:
                path = []
                length = g[n]
                while parents[n] != n:
                    path.append(n)
                    n = parents[n]
                path.reverse()
                return path, np.round(length, 2)

            for neighbor_node_coords in node.neighbor_list:
                cost = ((neighbor_node_coords[0] - n_coords[0]) ** 2 + (
                            neighbor_node_coords[1] - n_coords[1]) ** 2) ** (1 / 2)
                cost = np.round(cost, 2)
                m = (neighbor_node_coords[0], neighbor_node_coords[1])
                if g[n] + cost > max_dist:
                    continue
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

        dist_dict[(start[0], start[1])] = 0

        while len(q) > 0:
            u = None
            for coords in q:
                if u is None:
                    u = coords
                elif dist_dict[coords] < dist_dict[u]:
                    u = coords

            q.remove(u)

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

    def get_Dijkstra_path_and_dist(self, dist_dict, prev_dict, end):
        dist = dist_dict[(end[0], end[1])]

        path = [(end[0], end[1])]
        prev_node = prev_dict[(end[0], end[1])]
        while prev_node is not None:
            path.append(prev_node)
            temp = prev_node
            prev_node = prev_dict[temp]

        path.reverse()
        return path[1:], np.round(dist, 2)


class Node:
    def __init__(self, coords, local_frontiers, local_frontiers_ground, safe_frontiers, uncovered_safe_frontiers, updating_flight_info, height, sensor_range, max_height_cell, safe, occluded):
        self.coords = coords
        self.robot_height = height     # in meters
        self.position = np.hstack((coords, self.robot_height))
        self.max_height_cell = max_height_cell
        # Safe info
        self.safe = safe
        self.occluded = occluded

        self.utility_range = UTILITY_RANGE
        self.utility_range_grd = round(np.sqrt((self.utility_range**2) + (self.robot_height**2))) 
        self.observable_frontiers = self.initialize_observable_frontiers(local_frontiers, updating_flight_info)
        self.observable_frontiers_ground = self.initialize_observable_frontiers_ground(local_frontiers_ground, updating_flight_info)
        self.observable_frontiers_safe, self.observable_frontiers_safe_uncovered = self.initialize_observable_frontiers_safe(safe_frontiers, uncovered_safe_frontiers, updating_flight_info)

        self.neighbor_matrix = -np.ones((NUM_NODE_NEIGHBORS, NUM_NODE_NEIGHBORS))
        self.neighbor_list = []
        self.neighbor_list.append((self.coords[0], self.coords[1]))
        self.need_update_neighbor = True

        self.visited = 0

    def initialize_observable_frontiers(self, frontiers, updating_map_info):
        if len(frontiers) == 0:
            self.utility = 0
            return set()
        else:
            observable_frontiers = set()
            frontiers = np.array(list(frontiers)).reshape(-1, 2)
            dist_list = np.linalg.norm(frontiers - self.coords, axis=-1)
            new_frontiers_in_range = frontiers[dist_list < self.utility_range]
            for point in new_frontiers_in_range:
                collision = check_collision(self.coords, point, self.robot_height, updating_map_info, self.max_height_cell)
                if not collision:
                    observable_frontiers.add(tuple(point))
            self.utility = len(observable_frontiers)

            if self.utility <= MIN_UTILITY:
                self.utility = 0
                observable_frontiers = set()
            return observable_frontiers
        
    def initialize_observable_frontiers_ground(self, frontiers_ground, updating_map_info):  # Thrid column is height in metre

        if len(frontiers_ground) == 0:
            self.utility_ground = 0
            return set()
        elif self.occluded == 1 or self.occluded == 0.5:
            self.utility_ground = 0
            return set()
        else:
            observable_frontiers_ground = set()
            frontiers_ground = np.array(list(frontiers_ground)).reshape(-1, 2)
            # dist_list = np.linalg.norm(frontiers_ground - self.position, axis=-1)
            # new_frontiers_in_range = frontiers_ground[dist_list < self.utility_range_grd]
            dist_list = np.linalg.norm(frontiers_ground - self.coords, axis=-1)
            new_frontiers_in_range = frontiers_ground[dist_list < self.utility_range]
            for point in new_frontiers_in_range:
                collision = check_collision_3D(self.position, point, self.robot_height, updating_map_info, self.max_height_cell, use_cell=False, show_map=False)
                if not collision:
                    observable_frontiers_ground.add(tuple(point))
            self.utility_ground = len(observable_frontiers_ground)

            if self.utility_ground <= MIN_UTILITY:
                self.utility_ground = 0
                observable_frontiers_ground = set()
            return observable_frontiers_ground
        
    def initialize_observable_frontiers_safe(self, frontiers_safe, uncovered_safe_frontiers, updating_map_info):  # Thrid column is height in metre
        if len(frontiers_safe) == 0:
            self.utility_safe = 0
            self.utility_safe_uncovered = 0
            return set(), set()
        elif self.safe == 0 or self.safe == 0.5:
            self.utility_safe = 0
            self.utility_safe_uncovered = 0
            return set(), set()
        else:
            observable_frontiers_safe = set()
            observable_uncovered_frontiers_safe = set()
            frontiers_safe = np.array(list(frontiers_safe)).reshape(-1, 2)
            dist_list = np.linalg.norm(frontiers_safe - self.coords, axis=-1)
            frontiers_in_range = frontiers_safe[dist_list < self.utility_range]
            for point in frontiers_in_range:
                collision = check_collision_3D(self.position, point, self.robot_height, updating_map_info, self.max_height_cell, use_cell=False, show_map=False)
                if not collision: 
                    observable_frontiers_safe.add(tuple(point))
                    if len(uncovered_safe_frontiers) > 0:
                        if tuple(point) in uncovered_safe_frontiers:
                            observable_uncovered_frontiers_safe.add(tuple(point))

            self.utility_safe = len(observable_frontiers_safe)
            self.utility_safe_uncovered = len(observable_uncovered_frontiers_safe)

            # if self.utility_safe <= MIN_UTILITY:
            #     self.utility_safe = 0
            #     observable_frontiers_safe = set()

            return observable_frontiers_safe, observable_uncovered_frontiers_safe
    
    def update_neighbor_nodes(self, updating_map_info, nodes_dict, plot_x, plot_y):
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
                    elif (neighbor_node.data.coords[0] <= updating_map_info.map_origin_x
                          or neighbor_node.data.coords[1] <= updating_map_info.map_origin_y
                          or neighbor_node.data.coords[0] >= updating_map_info.map_origin_x +
                          updating_map_info.map.shape[1] * updating_map_info.cell_size
                          or neighbor_node.data.coords[1] >= updating_map_info.map_origin_y +
                          updating_map_info.map.shape[0] * updating_map_info.cell_size):
                        continue
                    else:
                        collision = check_collision(self.coords, neighbor_coords, self.robot_height, updating_map_info, self.max_height_cell)
                        
                        if not collision:
                            self.neighbor_matrix[i, j] = 1
                            self.neighbor_list.append((neighbor_coords[0], neighbor_coords[1]))
                            if plot_x != None and plot_y != None:
                                plot_x.append([self.coords[0], neighbor_coords[0]])
                                plot_y.append([self.coords[1], neighbor_coords[1]])
                            neighbor_node = neighbor_node.data
                            neighbor_matrix_x = center_index + (center_index - i)
                            neighbor_matrix_y = center_index + (center_index - j)
                            neighbor_node.neighbor_matrix[neighbor_matrix_x, neighbor_matrix_y] = 1
                            neighbor_node.neighbor_list.append((self.coords[0], self.coords[1]))
                                         
        if self.utility == 0:
            self.need_update_neighbor = False

    def update_neighbor_nodes_2(self, updating_map_info, nodes_dict, plot_x, plot_y):
        x = 1
        neigbor_nodes = nodes_dict.nearest_neighbors((self.coords[0], self.coords[1]), 25)[1:]

        for neighbor_node in neigbor_nodes:
            if tuple(self.coords) in neighbor_node.data.neighbor_list:
                continue
            else:
                dist = np.linalg.norm(neighbor_node.data.coords - self.coords)
                if dist <= 20:
                    collision = check_collision(self.coords, neighbor_node.data.coords, self.robot_height, updating_map_info, self.max_height_cell)
                    if not collision:
                        self.neighbor_list.append((neighbor_node.data.coords[0], neighbor_node.data.coords[1]))
                        if plot_x != None and plot_y != None:
                            plot_x.append([self.coords[0], neighbor_node.data.coords[0]])
                            plot_y.append([self.coords[1], neighbor_node.data.coords[1]])
                        neighbor_node.data.neighbor_list.append((self.coords[0], self.coords[1]))

        if self.utility == 0:
            self.need_update_neighbor = False

    def update_node_observable_frontiers(self, frontiers, updating_map_info):
        frontiers_observed = []
        for frontier in self.observable_frontiers:
            if frontier not in frontiers:
                frontiers_observed.append(frontier)
        for frontier_observed in frontiers_observed:
            self.observable_frontiers.remove(frontier_observed)

        # add new frontiers in the observable frontiers
        new_frontiers = frontiers - self.observable_frontiers
        new_frontiers = np.array(list(new_frontiers)).reshape(-1, 2)
        dist_list = np.linalg.norm(new_frontiers - self.coords, axis=-1)
        new_frontiers_in_range = new_frontiers[dist_list < self.utility_range]
        for point in new_frontiers_in_range:
            collision = check_collision(self.coords, point, self.robot_height, updating_map_info, self.max_height_cell)
            if not collision:
                self.observable_frontiers.add(tuple(point))

        self.utility = len(self.observable_frontiers)
        if self.utility <= MIN_UTILITY:
            self.utility = 0
            self.observable_frontiers = set()
            self.need_update_neighbor = False
        
    def update_node_observable_frontiers_ground(self, frontiers_ground, updating_map_info):  # Flight updating, global ground
        frontiers_ground_observed = []
        for frontier in self.observable_frontiers_ground:
            if frontier not in frontiers_ground:
                frontiers_ground_observed.append(frontier)
        for frontier_observed in frontiers_ground_observed:
            self.observable_frontiers_ground.remove(frontier_observed)

        # add new frontiers in the observable frontiers
        new_frontiers = frontiers_ground - self.observable_frontiers_ground
        new_frontiers = np.array(list(new_frontiers)).reshape(-1, 2)
        dist_list = np.linalg.norm(new_frontiers - self.coords, axis=-1)
        new_frontiers_in_range = new_frontiers[dist_list < self.utility_range]
        for point in new_frontiers_in_range:
            collision = check_collision_3D(self.position, point, self.robot_height, updating_map_info, self.max_height_cell, use_cell=False)
            if not collision:
                self.observable_frontiers_ground.add(tuple(point))

        self.utility_ground = len(self.observable_frontiers_ground)
        if self.utility_ground <= MIN_UTILITY:
            self.utility_ground = 0
            self.observable_frontiers_ground = set()
        
    def update_observable_safe_frontiers(self, safe_frontiers, uncovered_safe_frontiers, updating_map_info):
        if self.safe == 0 or len(safe_frontiers) == 0:       # If the node is not safe, then no need to update the safe frontiers
            self.utility_safe = 0
            self.utility_safe_uncovered = 0
            return
        
        frontiers_safe_observed = []
        frontiers_safe_uncovered_observed = []
        for frontier in self.observable_frontiers_safe:
            if frontier not in safe_frontiers:
                frontiers_safe_observed.append(frontier)
        for frontier in frontiers_safe_observed:
            self.observable_frontiers_safe.remove(frontier)

        for frontier_uncovered in self.observable_frontiers_safe_uncovered:
            if frontier_uncovered not in uncovered_safe_frontiers:
                frontiers_safe_uncovered_observed.append(frontier_uncovered)
        for frontier_uncovered in frontiers_safe_uncovered_observed:
            self.observable_frontiers_safe_uncovered.remove(frontier_uncovered)

        # add new frontiers in the observable frontiers
        new_frontiers = safe_frontiers - self.observable_frontiers_safe
        new_frontiers = np.array(list(new_frontiers)).reshape(-1, 2)
        dist_list = np.linalg.norm(new_frontiers - self.coords, axis=-1)
        new_frontiers_in_range = new_frontiers[dist_list < self.utility_range]
        for point in new_frontiers_in_range:
            collision = check_collision_3D(self.position, point, self.robot_height, updating_map_info, self.max_height_cell, use_cell=False)
            if not collision:
                self.observable_frontiers_safe.add(tuple(point))
                if len(uncovered_safe_frontiers) > 0:
                        if tuple(point) in uncovered_safe_frontiers:
                            self.observable_frontiers_safe_uncovered.add(tuple(point))

        self.utility_safe = len(self.observable_frontiers_safe)
        self.utility_safe_uncovered = len(self.observable_frontiers_safe_uncovered)
        # if self.utility_safe <= MIN_UTILITY:
        #     self.utility_safe = 0
        #     self.utility_safe_uncovered = 0
        #     self.observable_frontiers_safe = set()
        #     self.observable_frontiers_safe_uncovered = set()

    def set_visited(self):
        self.visited = 1
        self.observable_frontiers = set()
        self.utility = 0

    def set_safe(self):
        self.safe = 1
    def set_safe_building(self):
        self.safe = 0.5
        self.observable_frontiers_safe = set()
        self.observable_frontiers_safe_uncovered = set()
        self.utility_safe = 0
        self.utility_safe_uncovered = 0

    def set_unsafe(self):
        self.safe = 0
        self.observable_frontiers_safe = set()
        self.observable_frontiers_safe_uncovered = set()
        self.utility_safe = 0
        self.utility_safe_uncovered = 0

    def set_occluded(self):
        self.occluded = 1
        self.observable_frontiers_ground = set()
        self.utility_ground = 0
    def set_unoccluded(self):
        self.occluded = 0
    def set_building_ground(self):
        self.occluded = 0.5

