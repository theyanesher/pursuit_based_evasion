import time

import numpy as np
import imageio
import os
from copy import deepcopy, copy
from skimage.measure import block_reduce
from scipy.signal import decimate
from functools import partial
import matplotlib.pyplot as plt
import ffmpeg
import math
from sklearn.cluster import DBSCAN
from skimage.morphology import skeletonize, medial_axis
from scipy.ndimage import convolve
from skimage.measure import label
from parameter import *
from skimage.draw import line as sk_line


def get_cell_position_from_coords(coords, map_info):
    coords = coords.reshape(-1, 2)
    coords_x = coords[:, 0]
    coords_y = coords[:, 1]
    cell_x = ((coords_x - map_info.map_origin_x) / map_info.cell_size)
    cell_y = ((coords_y - map_info.map_origin_y) / map_info.cell_size)

    cell_position = np.around(np.stack((cell_x, cell_y), axis=-1)).astype(int)
    # assert False not in (cell_position.flatten() >= 0)
    if cell_position.shape[0] == 1:
        return cell_position[0]
    else:
        return cell_position


def get_coords_from_cell_position(cell_position, map_info):
    cell_position = cell_position.reshape(-1, 2)
    cell_x = cell_position[:, 0]
    cell_y = cell_position[:, 1]
    coords_x = cell_x * map_info.cell_size + map_info.map_origin_x
    coords_y = cell_y * map_info.cell_size + map_info.map_origin_y
    coords = np.stack((coords_x, coords_y), axis=-1)
    coords = np.around(coords, 1)
    if coords.shape[0] == 1:
        return coords[0]
    else:
        return coords


def get_hgt_scaling(max_cell_height):
    hgt_scaling = (max_cell_height - MIN_BUILDING_HGT_CELL) / (MAX_FLIGHT_HGT - MIN_FLIGHT_HGT)  # cell/metre
    return hgt_scaling


def get_cell_hgt(height, max_cell_height):
    hgt_scaling = get_hgt_scaling(max_cell_height)
    height_cell = math.ceil((height) * hgt_scaling)
    return height_cell


def get_metre_hgt(cell_height, max_cell_height):
    scale_factor = get_hgt_scaling(max_cell_height)
    height_metre = (cell_height) / scale_factor
    return height_metre


def get_free_area_coords(map_info):
    free_indices = np.where(map_info.map == 255)
    free_cells = np.asarray([free_indices[1], free_indices[0]]).T
    free_coords = get_coords_from_cell_position(free_cells, map_info)
    return free_coords


def get_ground_truth_node_coords(global_map_info):
    x_min = (global_map_info.map_origin_x // NODE_RESOLUTION) * NODE_RESOLUTION
    y_min = (global_map_info.map_origin_y // NODE_RESOLUTION) * NODE_RESOLUTION
    x_max = ((global_map_info.map_origin_x + global_map_info.map.shape[
        1] * CELL_SIZE) // NODE_RESOLUTION) * NODE_RESOLUTION
    y_max = ((global_map_info.map_origin_y + global_map_info.map.shape[
        0] * CELL_SIZE) // NODE_RESOLUTION) * NODE_RESOLUTION

    x_coords = np.arange(x_min, x_max, NODE_RESOLUTION)
    y_coords = np.arange(y_min, y_max, NODE_RESOLUTION)
    t1, t2 = np.meshgrid(x_coords, y_coords)
    nodes = np.vstack([t1.T.ravel(), t2.T.ravel()]).T
    nodes = np.around(nodes, 1)

    indices = []
    nodes_cells = get_cell_position_from_coords(nodes, global_map_info)
    for i, cell in enumerate(nodes_cells):
        if global_map_info.map[cell[1], cell[0]] == 255:
            indices.append(i)
    indices = np.array(indices)
    nodes = nodes[indices]

    return nodes


def get_free_and_connected_map(location, map_info, robot_height, show=False):
    # a binary map for free and connected areas
    # free = (map_info.map == 255).astype(float)
    # mask = (map_info.map >= 0) & (map_info.map  < robot_height - FLT_SAFETY_BUFFER*get_hgt_scaling())  
    mask = (map_info.map >= 0) & (map_info.map < robot_height)
    free = (map_info.map).astype(float)
    free[mask] = 1
    free[~mask] = 0
    labeled_free = label(free, connectivity=2)  # Only got 0 and 1
    if show:
        plt.subplot(1, 2, 1)
        plt.imshow(map_info.map, cmap='gray')
        plt.subplot(1, 2, 2)
        plt.imshow(labeled_free)
        plt.colorbar()
        plt.title("Labeled Free Map")
        plt.show()
    cell = get_cell_position_from_coords(location, map_info)
    label_number = labeled_free[cell[1], cell[0]]
    connected_free_map = (labeled_free == label_number)

    return connected_free_map


def is_frontier(location, map_info, type):
    # Flight info
    cell = get_cell_position_from_coords(location, map_info)
    if type == 'flight':
        free_value = 0
        unknown_value = -127
    if type == 'ground':
        free_value = 0
        unknown_value = [-127, -1]

    if map_info.map[cell[1], cell[0]] != free_value:
        return False
    else:
        y_min = max(0, cell[1] - 1)
        y_max = min(map_info.map.shape[0] - 1, cell[1] + 2)
        x_min = max(0, cell[0] - 1)
        x_max = min(map_info.map.shape[1] - 1, cell[0] + 2)
        # assert (
        #     cell[1] - 1 >= 0 and
        #     cell[0] - 1 >= 0 and
        #     cell[1] + 2 <= map_info.map.shape[0] and
        #     cell[0] + 2 <= map_info.map.shape[1]
        # ), "Index out of bounds for slicing."
        subarray = map_info.map[y_min:y_max, x_min:x_max]
        unknown = np.isin(subarray, unknown_value)
        n = np.sum(unknown)
        if 1 < n < 8:
            return True
        else:
            return False


def get_updating_node_coords(location, updating_map_info, robot_height, show=False, check_connectivity=True):
    x_min = updating_map_info.map_origin_x
    y_min = updating_map_info.map_origin_y
    x_max = updating_map_info.map_origin_x + (updating_map_info.map.shape[1] - 1) * CELL_SIZE
    y_max = updating_map_info.map_origin_y + (updating_map_info.map.shape[0] - 1) * CELL_SIZE

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

    free_connected_map = None

    if not check_connectivity:
        indices = []
        nodes_cells = get_cell_position_from_coords(nodes, updating_map_info).reshape(-1, 2)
        for i, cell in enumerate(nodes_cells):
            # assert 0 <= cell[1] < updating_map_info.map.shape[0] and 0 <= cell[0] < updating_map_info.map.shape[1]
            if updating_map_info.map[cell[1], cell[0]] >= 0 and updating_map_info.map[cell[1], cell[0]] <= robot_height:
                indices.append(i)
        indices = np.array(indices)
        nodes = nodes[indices].reshape(-1, 2)
    else:
        free_connected_map = get_free_and_connected_map(location, updating_map_info, robot_height, show=show)
        free_connected_map = np.array(free_connected_map)

        indices = []
        nodes_cells = get_cell_position_from_coords(nodes, updating_map_info).reshape(-1, 2)
        for i, cell in enumerate(nodes_cells):
            # assert 0 <= cell[1] < free_connected_map.shape[0] and 0 <= cell[0] < free_connected_map.shape[1]
            if free_connected_map[cell[1], cell[0]] == 1:
                indices.append(i)
        indices = np.array(indices)
        nodes = nodes[indices].reshape(-1, 2)

    return nodes, free_connected_map


def get_nodes(updating_map_info):
    ground_area_mask = (updating_map_info.map == 0)
    skeleton = skeletonize(ground_area_mask)

    labeled_skeleton = label(skeleton)  # Label connected components
    component_sizes = np.bincount(labeled_skeleton.ravel())
    largest_label = np.argmax(component_sizes[1:]) + 1
    cleaned_skeleton = (labeled_skeleton == largest_label)

    # """
    # Identify endpoints of the skeleton using a 3x3 convolution.
    # An endpoint is a pixel with exactly one neighbor.
    # """
    kernel = np.array([[1, 1, 1],
                       [1, 10, 1],
                       [1, 1, 1]])

    neighbor_count = convolve(cleaned_skeleton.astype(int), kernel, mode='constant', cval=0)
    keypoints_mask = ((neighbor_count == 11) | (
                neighbor_count >= 13)) & cleaned_skeleton  # 13 or more means 3+ neighbors

    # Cluster intersection points and keep one per connected region
    labeled, num_labels = label(keypoints_mask, return_num=True)
    representative_points = []
    for i in range(1, num_labels + 1):
        pts = np.argwhere(labeled == i)
        if pts.size:
            # Choose the median point (or compute the mean and round)
            rep = np.median(pts, axis=0).astype(int)
            representative_points.append(rep)
    intersection_points = np.array(representative_points)

    # Get all skeleton points
    skeleton_points = np.argwhere(cleaned_skeleton)
    # # Identify corner points using convolution
    selected_points = []
    route_points = []
    selected_points.extend(intersection_points.tolist())
    for point in skeleton_points:
        # Check distance to all previously selected points
        distances = np.linalg.norm(np.array(selected_points) - point, axis=1)
        if np.all(distances >= 10):
            selected_points.append(point)
            route_points.append(point)

    skeleton_points = np.array(selected_points)[:, ::-1]
    route_points = np.array(route_points)

    # # Visualize the results
    # overlay = np.copy(updating_map_info.map)
    # overlay[cleaned_skeleton] = 128  # Set skeleton pixels to a gray value
    # plt.imshow(overlay, cmap='gray')
    # plt.scatter(intersection_points[:, 1], intersection_points[:, 0], c='r', marker='x')
    # plt.scatter(skeleton_points[:, 0], skeleton_points[:, 1], c='b', marker='o')
    # plt.show()  
    skeleton_coords = get_coords_from_cell_position(skeleton_points, updating_map_info)
    return skeleton_coords


def radial_grid_sampling(center, r_min=10, growth_factor=2, max_radius=50):
    """
    Generates points in concentric circles with increasing separation.
    """
    cx, cy = center
    points = [(cx, cy)]  # Start with the center node
    r = r_min
    while r < max_radius:
        num_points = max(6, int(2 * np.pi * r / r_min))  # Ensure a minimum number of points per ring
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        for theta in angles:
            x, y = cx + r * np.cos(theta), cy + r * np.sin(theta)
            points.append((x, y))
        r *= growth_factor  # Increase radius for the next layer

    return np.array(points)


def poisson_disk_sampling(center, min_dist=1, max_radius=20, samples=30):
    from scipy.spatial import distance
    """
    Implements Poisson Disk Sampling with increasing separation distance.
    """
    cx, cy = center
    active_list = [(cx, cy)]
    points = [(cx, cy)]

    while active_list:
        idx = np.random.randint(len(active_list))
        x, y = active_list.pop(idx)
        min_d = min_dist * (1 + np.linalg.norm([x - cx, y - cy]) / max_radius)  # Increasing radius

        for _ in range(samples):
            r = np.random.uniform(min_d, 2 * min_d)
            theta = np.random.uniform(0, 2 * np.pi)
            new_x, new_y = x + r * np.cos(theta), y + r * np.sin(theta)

            if np.linalg.norm([new_x - cx, new_y - cy]) > max_radius:
                continue  # Ignore points outside max radius

            if all(distance.euclidean((new_x, new_y), p) >= min_d for p in points):
                points.append((new_x, new_y))
                active_list.append((new_x, new_y))

    return np.array(points)

    # all_points = []
    # for points in selected_points:
    #     all_points.extend(radial_grid_sampling(points, max_radius=20))
    #     # all_points.extend(poisson_disk_sampling(points, max_radius=20))

    # all_points = np.array([all_points]).reshape(-1, 2)

    # plt.subplot(1,3,1)
    # plt.imshow(ground_truth, cmap='gray')
    # plt.subplot(1, 3, 2)
    # plt.imshow(skeleton, cmap='gray')
    # plt.subplot(1, 3, 3)
    # plt.imshow(cleaned_skeleton, cmap='gray')
    # plt.show()

    # Overlay the skeleton on the ground truth image
    overlay = np.copy(ground_truth)
    overlay[cleaned_skeleton] = 128  # Set skeleton pixels to a gray value

    plt.subplot(1, 3, 1)
    plt.imshow(overlay, cmap='gray')
    plt.title('Ground Truth with Skeleton Overlay')
    plt.subplot(1, 3, 2)
    plt.imshow(cleaned_skeleton, cmap='gray')
    # plt.scatter(points[:,1], points[:,0], color='red', label="Endpoints", s=50)
    plt.scatter(intersection_points[:, 1], intersection_points[:, 0], color='blue', label="Intersections", s=50)
    plt.title('Skeleton')
    plt.subplot(1, 3, 3)
    plt.imshow(cleaned_skeleton, cmap='gray')
    plt.scatter(skeleton_points[:, 1], skeleton_points[:, 0], color='red', label="Endpoints", s=50)
    # plt.scatter(final_nodes[:,1], final_nodes[:,0], color='blue', label="Route Points", s=50)
    plt.show()

    return route_points


def get_frontier_in_map(map_info_to_copy, robot_height):
    map_info = copy(map_info_to_copy)
    x_len = map_info.map.shape[1]
    y_len = map_info.map.shape[0]
    unknown = (map_info.map == -127) * 1
    unknown = np.lib.pad(unknown, ((1, 1), (1, 1)), 'constant', constant_values=0)
    unknown_neighbor = unknown[2:][:, 1:x_len + 1] + unknown[:y_len][:, 1:x_len + 1] + unknown[1:y_len + 1][:, 2:] \
                       + unknown[1:y_len + 1][:, :x_len] + unknown[:y_len][:, 2:] + unknown[2:][:, :x_len] + \
                       unknown[2:][:, 2:] + unknown[:y_len][:, :x_len]
    free_cell_indices_1 = np.where(map_info.map.ravel(order='F') >= 0)[0]
    free_cell_indices_2 = np.where(map_info.map.ravel(order='F') < robot_height)[0]
    free_cell_indices = np.intersect1d(free_cell_indices_1, free_cell_indices_2)
    frontier_cell_1 = np.where(1 < unknown_neighbor.ravel(order='F'))[0]
    frontier_cell_2 = np.where(unknown_neighbor.ravel(order='F') < 8)[0]
    frontier_cell_indices = np.intersect1d(frontier_cell_1, frontier_cell_2)
    frontier_cell_indices = np.intersect1d(free_cell_indices, frontier_cell_indices)

    x = np.linspace(0, x_len - 1, x_len)
    y = np.linspace(0, y_len - 1, y_len)
    t1, t2 = np.meshgrid(x, y)
    cells = np.vstack([t1.T.ravel(), t2.T.ravel()]).T
    frontier_cell = cells[frontier_cell_indices].astype(int)

    frontier_coords = get_coords_from_cell_position(frontier_cell, map_info).reshape(-1, 2)
    if frontier_cell.shape[0] > 0 and FRONTIER_CELL_SIZE != CELL_SIZE:
        frontier_coords = frontier_coords.reshape(-1, 2)
        frontier_coords = frontier_down_sample(frontier_coords)
    frontier_coords = set(map(tuple, frontier_coords))

    if len(frontier_coords) == 0:
        frontier_coords = set()

    return frontier_coords


def get_frontier_in_map_ground(map_info_to_copy, max_cell_height):
    map_info = copy(map_info_to_copy)
    x_len = map_info.map.shape[1]
    y_len = map_info.map.shape[0]
    current_map = map_info.map.copy()

    current_map[current_map == -1] = -127  # To transform occluded area into unknown area
    unknown = (current_map == -127) * 1
    unknown = np.lib.pad(unknown, ((1, 1), (1, 1)), 'constant', constant_values=0)
    unknown_neighbor = unknown[2:][:, 1:x_len + 1] + unknown[:y_len][:, 1:x_len + 1] + unknown[1:y_len + 1][:, 2:] \
                       + unknown[1:y_len + 1][:, :x_len] + unknown[:y_len][:, 2:] + unknown[2:][:, :x_len] + \
                       unknown[2:][:, 2:] + unknown[:y_len][:, :x_len]
    free_cell_indices = np.where(current_map.ravel(order='F') == 0)[0]
    frontier_cell_1 = np.where(1 < unknown_neighbor.ravel(order='F'))[0]
    frontier_cell_2 = np.where(unknown_neighbor.ravel(order='F') < 8)[0]
    frontier_cell_indices = np.intersect1d(frontier_cell_1, frontier_cell_2)
    frontier_cell_indices = np.intersect1d(free_cell_indices, frontier_cell_indices)

    x = np.linspace(0, x_len - 1, x_len)
    y = np.linspace(0, y_len - 1, y_len)
    t1, t2 = np.meshgrid(x, y)
    cells = np.vstack([t1.T.ravel(), t2.T.ravel()]).T
    frontier_cell = cells[frontier_cell_indices].astype(int)

    if frontier_cell.shape[0] > 0:
        # frontier_cell = filter_coordinates(frontier_cell, map_info.map, 0)
        frontier_coords = get_coords_from_cell_position(frontier_cell, map_info)
        if frontier_coords.shape[0] > 0 and FRONTIER_CELL_SIZE != CELL_SIZE:
            frontier_coords = frontier_coords.reshape(-1, 2)
            frontier_coords = frontier_down_sample(frontier_coords)
            # heights = np.zeros((frontier_coords.shape[0], 1))
            # frontier_coords = np.hstack((frontier_coords, heights))
        frontier_coords = set(map(tuple, frontier_coords))

        if len(frontier_coords) == 0:
            frontier_coords = set()
    else:
        frontier_coords = set()

    return frontier_coords


# Compute safe zone frontiers in 2D (x,y), assume height = 0
def get_safe_zone_frontier(safe_info, ground_info, show=False):
    safe_info = deepcopy(safe_info)

    x_len = safe_info.map.shape[1]
    y_len = safe_info.map.shape[0]

    safe_free_cell_indices = np.where(safe_info.map.ravel(order='F') == 255)[0]
    safe_zone = (safe_info.map == 255) * 1
    safe_zone = np.lib.pad(safe_zone, ((1, 1), (1, 1)), 'constant', constant_values=0)
    safe_neighbor = safe_zone[2:][:, 1:x_len + 1] + safe_zone[:y_len][:, 1:x_len + 1] + safe_zone[1:y_len + 1][:, 2:] \
                    + safe_zone[1:y_len + 1][:, :x_len] + safe_zone[:y_len][:, 2:] + safe_zone[2:][:, :x_len] + \
                    safe_zone[2:][:, 2:] + safe_zone[:y_len][:, :x_len]

    safe_frontier_cell_1 = np.where(1 < safe_neighbor.ravel(order='F'))[0]
    safe_frontier_cell_2 = np.where(safe_neighbor.ravel(order='F') < 8)[0]
    safe_frontier_cell_indices = np.intersect1d(safe_frontier_cell_1, safe_frontier_cell_2)
    safe_frontier_cell_indices = np.intersect1d(safe_free_cell_indices,
                                                safe_frontier_cell_indices)  # border of safe zone

    x_len = ground_info.map.shape[1]
    y_len = ground_info.map.shape[0]
    obstacle = (ground_info.map > 0) * 1
    obstacle = np.lib.pad(obstacle, ((1, 1), (1, 1)), 'constant', constant_values=1)  # add obstacle around extended map
    obstacle_neighbor = obstacle[2:][:, 1:x_len + 1] + obstacle[:y_len][:, 1:x_len + 1] + obstacle[1:y_len + 1][:, 2:] \
                        + obstacle[1:y_len + 1][:, :x_len] + obstacle[:y_len][:, 2:] + obstacle[2:][:, :x_len] + \
                        obstacle[2:][:, 2:] + obstacle[:y_len][:, :x_len]
    obstacle_frontier_cell_1 = np.where(1 <= obstacle_neighbor.ravel(order='F'))[0]  # border of map
    obstacle_frontier_cell_2 = np.where(obstacle_neighbor.ravel(order='F') < 8)[0]
    obstacle_frontier_cell = np.intersect1d(obstacle_frontier_cell_1, obstacle_frontier_cell_2)
    valid_safe_frontier_cell_indices = np.setdiff1d(safe_frontier_cell_indices,
                                                    obstacle_frontier_cell)  # border of safe zone in free area

    x = np.linspace(0, x_len - 1, x_len)
    y = np.linspace(0, y_len - 1, y_len)
    t1, t2 = np.meshgrid(x, y)
    cells = np.vstack([t1.T.ravel(), t2.T.ravel()]).T
    frontier_cell = cells[valid_safe_frontier_cell_indices].astype(int)

    # safe_frontier_cells = cells[safe_frontier_cell_indices].astype(int)
    # pixel_values_safe_2 = safe_info.map[frontier_cell[:, 0], frontier_cell[:, 1]]
    # print(pixel_values_safe_2)
    # assert -127 not in pixel_values_safe_2, "Array contains -127"

    if frontier_cell.shape[0] > 0:
        # frontier_cell = filter_coordinates(frontier_cell, map_info.map, 0)
        frontier_coords = get_coords_from_cell_position(frontier_cell, safe_info)
        if frontier_coords.shape[0] > 0 and FRONTIER_CELL_SIZE != CELL_SIZE:
            frontier_coords = frontier_coords.reshape(-1, 2)
            frontier_coords = frontier_down_sample(frontier_coords)
            # heights = np.zeros((frontier_coords.shape[0], 1))
            # frontier_coords = np.hstack((frontier_coords, heights))
        frontier_coords = set(map(tuple, frontier_coords))

        if len(frontier_coords) == 0:
            frontier_coords = set()
    else:
        frontier_coords = set()

    ###########################################
    # Plotting
    ###########################################
    if show:
        plt.subplot(2, 2, 1)
        plt.title('Final frontiers')
        plt.imshow(safe_info.map, cmap='gray')
        safe_frontiers = cells[safe_frontier_cell_indices]
        plt.scatter(safe_frontiers[:, 0], safe_frontiers[:, 1], c='g', s=5)
        row_indices, col_indices = np.where(ground_info.map == -1)
        if len(row_indices) > 0:
            plt.scatter(col_indices, row_indices, c='c', s=1, zorder=2)
        plt.subplot(2, 2, 2)
        plt.title('Ground map')
        plt.imshow(ground_info.map, cmap='gray')
        obstacle_frontier_cell = cells[obstacle_frontier_cell]
        plt.scatter(obstacle_frontier_cell[:, 0], obstacle_frontier_cell[:, 1], c='r', s=2)
        if len(row_indices) > 0:
            plt.scatter(col_indices, row_indices, c='c', s=1, zorder=2)
        plt.subplot(2, 2, 3)
        plt.title('Obstacle map')
        plt.imshow(obstacle, cmap='gray')
        if len(row_indices) > 0:
            plt.scatter(col_indices, row_indices, c='c', s=1, zorder=2)
        plt.scatter(frontier_cell[:, 0], frontier_cell[:, 1], c='g', s=5)
        plt.subplot(2, 2, 4)
        plt.title('Ground map')
        plt.imshow(ground_info.map, cmap='gray')
        if len(row_indices) > 0:
            plt.scatter(col_indices, row_indices, c='c', s=1, zorder=2)
        plt.show()

    # frontier_coords = get_coords_from_cell_position(frontier_cell, safe_info)
    # if frontier_cell.shape[0] > 0 and FRONTIER_CELL_SIZE != CELL_SIZE:
    #     # pixel_values = map_info.map[frontier_cell[:, 1], frontier_cell[:, 0]]
    #     frontier_coords = np.array(list(frontier_down_sample(frontier_coords, ground_info=ground_info)))

    #     frontier_heights = np.full((frontier_coords.shape[0], 1), 0)
    #     if frontier_coords.shape[0] > 0:
    #         frontier_heights = np.full((frontier_coords.shape[0], 1), 0)
    #         frontier_coords = set(map(tuple, np.hstack((frontier_coords, frontier_heights)).reshape(-1,3))) 
    #     else:
    #         frontier_coords = set()
    #     # frontier_coords_to_check = frontier_coords[:,0] + frontier_coords[:,1] * 1j
    #     # frontier_coords_new_to_check = frontier_coords_new[:,0] + frontier_coords_new[:,1] * 1j
    #     # indices = np.where(np.isin(frontier_coords_to_check, frontier_coords_new_to_check))
    #     # # new_pixel_hgts = np.where(pixel_values[indices] == 1, 0, pixel_values[indices])     # Convert pixel value 1 free space to 0 metres
    #     # hgt_metres = np.vectorize(get_metre_hgt)(pixel_values[indices])
    #     # frontier_coords = np.column_stack((frontier_coords_new, hgt_metres)) 
    # else:
    #     frontier_heights = np.full((frontier_coords.shape[0], 1), 0)
    #     frontier_coords = set(map(tuple, np.hstack((frontier_coords, frontier_heights)).reshape(-1,3))) 

    return frontier_coords


def filter_coordinates(coordinates, pixel_array, threshold):
    """
    Filter coordinates based on pixel values in a pixel array.
    
    Args:
    - coordinates (numpy.ndarray): Array of coordinates with shape (n, 2).
    - pixel_array (numpy.ndarray): Pixel array with shape (p, p).
    - threshold (int): Threshold value for pixel intensity.
    
    Returns:
    - filtered_coordinates (list): List of coordinates with pixel values less than threshold.
    """
    # Extract pixel values at the given coordinates
    coordinates = coordinates.astype(int)
    pixel_values = pixel_array[coordinates[:, 1], coordinates[:, 0]]

    # Find indices where pixel values are more than the threshold
    filtered_indices = np.where(pixel_values >= threshold)[0]

    # Use the filtered indices to get corresponding coordinates
    filtered_coordinates = coordinates[filtered_indices]

    # plt.imshow(pixel_array, cmap='gray')
    # plt.show()

    return filtered_coordinates


def get_partial_map_from_center(original_map_info, center_coords, partial_map_size):
    partial_map_origin_x = (center_coords[
                                0] - partial_map_size / 2) // NODE_RESOLUTION * NODE_RESOLUTION
    partial_map_origin_y = (center_coords[
                                1] - partial_map_size / 2) // NODE_RESOLUTION * NODE_RESOLUTION
    partial_map_top_x = partial_map_origin_x + partial_map_size + NODE_RESOLUTION
    partial_map_top_y = partial_map_origin_y + partial_map_size + NODE_RESOLUTION

    min_x = original_map_info.map_origin_x
    min_y = original_map_info.map_origin_y
    max_x = original_map_info.map_origin_x + original_map_info.cell_size * original_map_info.map.shape[1]
    max_y = original_map_info.map_origin_y + original_map_info.cell_size * original_map_info.map.shape[0]

    if partial_map_origin_x < min_x:
        partial_map_origin_x = min_x
    if partial_map_origin_y < min_y:
        partial_map_origin_y = min_y
    if partial_map_top_x > max_x:
        partial_map_top_x = max_x
    if partial_map_top_y > max_y:
        partial_map_top_y = max_y

    partial_map_origin_x = np.around(partial_map_origin_x, 1)
    partial_map_origin_y = np.around(partial_map_origin_y, 1)
    partial_map_top_x = np.around(partial_map_top_x, 1)
    partial_map_top_y = np.around(partial_map_top_y, 1)

    partial_map_origin = np.array([partial_map_origin_x, partial_map_origin_y])
    partial_map_origin_in_global_map = get_cell_position_from_coords(partial_map_origin, original_map_info)

    partial_map_top = np.array([partial_map_top_x, partial_map_top_y])
    partial_map_top_in_global_map = get_cell_position_from_coords(partial_map_top, original_map_info)

    partial_map = original_map_info.map[
                  partial_map_origin_in_global_map[1]:partial_map_top_in_global_map[1],
                  partial_map_origin_in_global_map[0]:partial_map_top_in_global_map[0]]

    partial_map_info = Map_info(partial_map, partial_map_origin_x, partial_map_origin_y, original_map_info.cell_size)

    return partial_map_info


def frontier_down_sample(data, ground_info=None, voxel_size=FRONTIER_CELL_SIZE):
    if data.ndim == 1:
        data = data.reshape(1, 2)
    data = data[:, :2]
    voxel_indices = np.array(data / voxel_size, dtype=int).reshape(-1, 2)

    voxel_dict = {}
    for i, point in enumerate(data):
        voxel_index = tuple(voxel_indices[i])

        if voxel_index not in voxel_dict:
            voxel_dict[voxel_index] = point
        else:
            current_point = voxel_dict[voxel_index]
            if np.linalg.norm(point - np.array(voxel_index) * voxel_size) < np.linalg.norm(
                    current_point - np.array(voxel_index) * voxel_size):
                voxel_dict[voxel_index] = point
    downsampled_data = np.array(list(voxel_dict.values()))
    # if ground_info is not None:
    #     downsampled_data = remove_isolate_frontiers(downsampled_data, voxel_size, ground_info)
    return downsampled_data


def check_collision(start, end, height, map_info, max_cell_height):
    # Bresenham line algorithm checking
    collision = False
    start_cell = get_cell_position_from_coords(np.array(start), map_info)
    end_cell = get_cell_position_from_coords(np.array(end), map_info)
    cell_height = get_cell_hgt(height, max_cell_height)
    map = map_info.map

    x0 = start_cell[0]
    y0 = start_cell[1]
    x1 = end_cell[0]
    y1 = end_cell[1]
    x0 = x0.round()
    y0 = y0.round()
    x1 = x1.round()
    y1 = y1.round()
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    x, y = x0, y0
    error = dx - dy
    x_inc = 1 if x1 > x0 else -1
    y_inc = 1 if y1 > y0 else -1
    dx *= 2
    dy *= 2

    collision_flag = 0
    max_collision = 3

    while 0 <= x < map.shape[1] and 0 <= y < map.shape[0]:
        k = map.item(int(y), int(x))
        if x == x1 and y == y1:
            break
        # Add a safety buffer for building that are too close the robot
        if k >= cell_height:  # - FLT_SAFETY_BUFFER*hgt_scaling:
            collision = True
            break

        if k >= cell_height and collision_flag < max_collision:
            collision_flag += 1
            if collision_flag >= max_collision:
                break

        if k == -127:
            collision = True
            break
        if error > 0:
            x += x_inc
            error -= dy
        else:
            y += y_inc
            error += dx
    return collision


def check_collision_3D(start, end, height, map_info, max_height_cell, use_cell=False, show_map=False, occlussion=False):
    # Only for ground info map
    # Bresenham line algorithm checking
    collision = False

    if use_cell:
        start_cell = start
        end_cell = end
    else:
        start_cell = get_cell_position_from_coords(start[:2], map_info)
        end_cell = get_cell_position_from_coords(end[:2], map_info)
        height = get_cell_hgt(height, max_height_cell)  # Convert height to cell height
        # end_height = get_cell_hgt(end[2], max_height_cell) 

    end_height = 0

    map = map_info.map

    x0 = start_cell[0]
    y0 = start_cell[1]
    x1 = end_cell[0]
    y1 = end_cell[1]
    x0 = x0.round()
    y0 = y0.round()
    x1 = x1.round()
    y1 = y1.round()

    rr, cc = sk_line(y0, x0, y1, x1)

    # Compute total distance for scaling current_height
    total_dist = np.hypot(x1 - x0, y1 - y0)
    collision_count = 0
    max_collision = 2

    # Precompute the distances and corresponding heights for all points along the line
    if total_dist != 0:
        distances = np.hypot(x1 - cc, y1 - rr)
        heights = (distances / total_dist) * height
    else:
        heights = np.full_like(rr, height, dtype=float)

    for i in range(len(rr)):
        # Stop if we've reached the end point.
        if cc[i] == x1 and rr[i] == y1:
            break

        # Compute the euclidean distance from the current point to the end point.
        current_height = heights[i]
        k = map[rr[i], cc[i]]
        # try:
        #     k = map[rr[i], cc[i]]
        # except:
        #     plt.figure()
        #     plt.imshow(map, cmap='gray', interpolation='nearest')
        #     plt.plot(x0, y0, 'ro')
        #     plt.plot(x1, y1, 'bo')
        #     colors = np.linspace(0, 1, len(rr))
        #     scatter = plt.scatter(cc, rr, c=colors, cmap='jet')
        #     plt.colorbar(scatter, label='Gradient along the line')
        #     plt.title("Line from (%d, %d) to (%d, %d)" % (x0, y0, x1, y1))
        #     print(x0,y0,x1,y1)
        #     print(cc,rr)
        #     plt.show()

        if k >= current_height:
            collision_count += 1
            if collision_count > max_collision:
                collision = True
                break

        if k == -127:
            collision = True
            break

        if not occlussion:
            if k == -1:
                collision = True
                break

    # dx, dy = abs(x1 - x0), abs(y1 - y0)
    # x, y = x0, y0
    # error = dx - dy
    # x_inc = 1 if x1 > x0 else -1
    # y_inc = 1 if y1 > y0 else -1
    # dx *= 2
    # dy *= 2

    # total_dist = np.linalg.norm(start_cell-end_cell)   # Total distance between start and end point in cell

    # if show_map:
    #     plt.clf()
    #     plt.imshow(map, cmap='gray')
    #     plt.scatter(x0,y0,c='g',s=10, zorder=10)
    #     plt.scatter(x1,y1,c='r',s=10, zorder=8)

    # collision_count = 0
    # max_collision = 3

    # while 0 <= x < map.shape[1] and 0 <= y < map.shape[0]:
    #     # Compute the current voxel height
    #     next_point = np.array([x,y])
    #     if show_map:
    #         plt.scatter(x,y,c='b',s=10)
    #     current_dist = np.linalg.norm(next_point-end_cell)

    #     if total_dist == 0:
    #         current_height = height
    #     else:
    #         current_height = (current_dist/total_dist)*(height - end_height) + end_height   # in cell

    #     k = map.item(int(y), int(x))

    #     if x == x1 and y == y1:

    #         break

    #     if k >= current_height:
    #         collision_count += 1
    #         if collision_count > max_collision:
    #             collision = True
    #             break

    #     if k == -127:
    #         collision = True
    #         break

    #     if error > 0:
    #         x += x_inc
    #         error -= dy
    #     else:
    #         y += y_inc
    #         error += dx

    # if show_map and collision:
    #     print(end_height)
    #     print(k, current_height)
    #     plt.show()
    return collision


def check_collision_3D_safe(start, end, map_info, max_height_cell, use_cell=False):
    # Only for ground info map
    # Bresenham line algorithm checking
    collision = False

    if use_cell:
        start_cell = start
        end_cell = end
        end_height = map_info.map[end_cell[1], end_cell[0]]
        if end_height == -127:
            end_height = 0
    else:
        start_cell = get_cell_position_from_coords(start[:2], map_info)
        end_cell = get_cell_position_from_coords(end[:2], map_info)
        height = get_cell_hgt(start[2], max_height_cell)  # Convert height to cell height
        end_height = end[2]
    # start_cell = get_cell_position_from_coords(start, map_info)
    # end_cell = get_cell_position_from_coords(end, map_info)

    if end_height != 0:
        print('end', end_height)

    map = map_info.map

    x0 = start_cell[0]
    y0 = start_cell[1]
    x1 = end_cell[0]
    y1 = end_cell[1]
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    x, y = x0, y0
    error = dx - dy
    x_inc = 1 if x1 > x0 else -1
    y_inc = 1 if y1 > y0 else -1
    dx *= 2
    dy *= 2

    total_dist = np.linalg.norm(start_cell - end_cell)  # Total distance between start and end point in cell

    collision_count = 0
    max_collision = 2

    while 0 <= x < map.shape[1] and 0 <= y < map.shape[0]:
        # Compute the current voxel height
        next_point = np.array([x, y])
        current_dist = np.linalg.norm(next_point - end_cell)

        if total_dist == 0:
            current_height = height
        else:
            current_height = (current_dist / total_dist) * (height - end_height) + end_height  # in cell

        k = map.item(int(y), int(x))

        if x == x1 and y == y1:
            break
        if k >= current_height or k == -127:
            collision_count += 1
            if collision_count > max_collision:
                collision = True
                break
        # if k == -127:
        #     collision = True
        #     break

        if error > 0:
            x += x_inc
            error -= dy
        else:
            y += y_inc
            error += dx
    return collision


def check_collision_safe(start, end, map_info, max_collision=1):
    # Bresenham line algorithm checking
    collision = False

    start_cell = get_cell_position_from_coords(start, map_info)
    end_cell = get_cell_position_from_coords(end, map_info)
    map = map_info.map

    x0 = start_cell[0]
    y0 = start_cell[1]
    x1 = end_cell[0]
    y1 = end_cell[1]
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    x, y = x0, y0
    error = dx - dy
    x_inc = 1 if x1 > x0 else -1
    y_inc = 1 if y1 > y0 else -1
    dx *= 2
    dy *= 2

    collision_flag = 0

    while 0 <= x < map.shape[1] and 0 <= y < map.shape[0]:
        k = map.item(int(y), int(x))
        if x == x1 and y == y1:
            break
        if k in [0, 1, 127]:
            collision_flag += 1
            if collision_flag >= max_collision:
                collision = True
                break
        if error > 0:
            x += x_inc
            error -= dy
        else:
            y += y_inc
            error += dx
    return collision


def check_cumulative_collision(start, end, map_info, max_collision=1):
    collision = False

    start_cell = get_cell_position_from_coords(start, map_info)
    end_cell = get_cell_position_from_coords(end, map_info)
    map = map_info.map

    x0 = start_cell[0]
    y0 = start_cell[1]
    x1 = end_cell[0]
    y1 = end_cell[1]
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    x, y = x0, y0
    error = dx - dy
    x_inc = 1 if x1 > x0 else -1
    y_inc = 1 if y1 > y0 else -1
    dx *= 2
    dy *= 2

    collision_flag = 0
    last_k = -1

    while 0 <= x < map.shape[1] and 0 <= y < map.shape[0]:
        k = map.item(int(y), int(x))
        if x == x1 and y == y1:
            break
        if k in [0, 1, 127]:
            collision_flag += 1
            if collision_flag >= max_collision:
                collision = True
                break
        if error > 0:
            x += x_inc
            error -= dy
        else:
            y += y_inc
            error += dx
        if last_k != k and collision_flag > 0:
            collision_flag -= 1
        last_k = k
    return collision


def euclidean_distance(p1, p2):
    p1 = np.array(p1)
    p2 = np.array(p2)
    return np.sqrt(np.sum((p1 - p2) ** 2))


def find_common_elements(arrays):
    # Concatenate all arrays together along axis 0
    concatenated_array = np.concatenate(arrays, axis=0)

    # Find unique rows across the concatenated array
    unique_rows, counts = np.unique(concatenated_array, axis=0, return_counts=True)

    # Filter out rows that appear in all arrays
    common_rows = unique_rows[counts == len(arrays)]

    return common_rows


def remove_isolate_frontiers(data, voxel_size, ground_info, min_cluster_size=3, plot=False):
    """
    Clusters frontiers and removes noise and small clusters.

    Parameters:
        data (np.ndarray): Input frontier points (N x 2).
        voxel_size (float): Size of the voxel grid.
        min_cluster_size (int): Minimum number of points required for a cluster to be retained.

    Returns:
        np.ndarray: Filtered frontier points with noise and small clusters removed.
    """
    if ground_info is not None and plot:
        plt.subplot(1, 2, 1)
        plt.axis('off')
        plt.title('Ground zone', y=-0.12, fontweight='bold')
        plt.imshow(ground_info.map, cmap='gray_r')
        plt.imshow(ground_info.map == -127, cmap=plt.cm.colors.ListedColormap(['none', 'lightblue']), alpha=0.8)
        data_cell = get_cell_position_from_coords(data, ground_info)
        plt.scatter(data_cell[:, 0], data_cell[:, 1], c='r', s=5)
    if len(data) <= 1:
        return np.array([])

    # Reshape the data to ensure it's 2D
    data = data.reshape(-1, 2)

    # Perform clustering using DBSCAN
    clustering = DBSCAN(eps=voxel_size * 2, min_samples=3).fit(data)
    labels = clustering.labels_

    # Filter out noise (label == -1) and small clusters
    filtered_data = []
    for label in set(labels):
        if label == -1:  # Skip noise points
            continue
        cluster_points = data[labels == label]
        if len(cluster_points) >= min_cluster_size:  # Retain only sufficiently large clusters
            filtered_data.append(cluster_points)

    # Concatenate all remaining clusters into a single array
    if filtered_data:
        if ground_info is not None and plot:
            data = np.vstack(filtered_data)
            plt.subplot(1, 2, 2)
            plt.axis('off')
            plt.title('Ground zone', y=-0.12, fontweight='bold')
            plt.imshow(ground_info.map, cmap='gray_r')
            plt.imshow(ground_info.map == -127, cmap=plt.cm.colors.ListedColormap(['none', 'lightblue']), alpha=0.8)
            data_cell = get_cell_position_from_coords(data, ground_info)
            plt.scatter(data_cell[:, 0], data_cell[:, 1], c='r', s=5)
            plt.show()
        return np.vstack(filtered_data)
    else:
        return np.array([])


def make_gif(path, n, frame_files, rate):
    with imageio.get_writer('{}/{}_explored_rate_{:.4g}.gif'.format(path, n, rate), mode='I', duration=128) as writer:
        for frame in frame_files:
            image = imageio.imread(frame)
            writer.append_data(image)
    print('Generation of gif complete!')

    # Remove files
    for filename in frame_files[:-1]:
        os.remove(filename)

    # stream = ffmpeg.input('{}/{}_explored_rate_{:.4g}.gif'.format(path, n, rate))
    # stream = ffmpeg.filter(stream, 'fps', fps=25)  # Set the desired frame rate
    # stream = ffmpeg.output(stream, '{}/{}_explored_rate_{:.4g}.mp4'.format(path, n, rate), vcodec='libx264', pix_fmt='yuv420p', movflags='faststart')

    # ffmpeg.run(stream, quiet=True)
    # print('Generation of mp4 complete!\n')


class Map_info:
    def __init__(self, map, map_origin_x, map_origin_y, cell_size):
        self.map = map
        self.map_origin_x = map_origin_x
        self.map_origin_y = map_origin_y
        self.cell_size = cell_size

    def update_map_info(self, map, map_origin_x, map_origin_y):
        self.map = map
        self.map_origin_x = map_origin_x
        self.map_origin_y = map_origin_y
