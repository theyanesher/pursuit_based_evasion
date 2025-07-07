import os
import matplotlib.pyplot as plt
from skimage import io
# from skimage.measure import block_reduce
from copy import deepcopy
import numpy as np
import random
from scipy import ndimage
from skimage.measure import label

from sensor import exploration_sensor, coverage_sensor, decrease_safety_by_frontier, identify_occlussion
from parameter import *
from utils import *
from matplotlib.collections import LineCollection

from matplotlib.patches import Circle


class Env:
    def __init__(self, episode_index, n_agents, plot=False, save_path=None):
        self.episode_index = episode_index
        self.plot = plot
        self.n_agents = n_agents
        self.opening_size = SOUTH_OPENING_SIZE
        self.cell_size = CELL_SIZE  # meter
        self.sensor_range = SENSOR_RANGE  # meter
        self.sensor_range_cell = round(self.sensor_range / self.cell_size)
        self.safety_range = SAFETY_RANGE  # meter
        self.safety_range_cell = round(self.safety_range / self.cell_size)
        self.building_percentile = PERCENTAGE_BUILDING_HEIGHT  # percentile of building below the drone
        self.min_building_size = MIN_BUILDING_SIZE
        self.num_border_points = NUM_BORDER_POINTS
        self.save_path = save_path

        # 0: ground, 1-254 building, 255: border
        self.ground_truth, self.colour_map = self.import_ground_truth(plot=self.plot)

        self.ground_sensor_range = round(np.sqrt((self.sensor_range ** 2) + (self.robot_height ** 2)))
        self.ground_sensor_range_cell = round(self.ground_sensor_range / self.cell_size)

        self.ground_truth_size = np.shape(self.ground_truth)  # cell
        self.belief_flight = np.ones(self.ground_truth_size) * -127
        self.belief_ground = np.ones(self.ground_truth_size) * -127
        self.safe_zone = np.zeros(self.ground_truth_size)

        # Initial starting region
        self.belief_flight[-20:, :] = self.ground_truth[-20:, :]
        self.belief_ground[-20:, :] = self.ground_truth[-20:, :]

        # Compute the origin
        free_cells = np.where(self.belief_flight == 0)
        random_location_index = random.randint(0, free_cells[0].shape[0] - 1)
        initial_cell = np.array([free_cells[1][random_location_index], free_cells[0][random_location_index]])

        self.belief_origin_x = -np.round(initial_cell[0] * self.cell_size, 1)  # meter
        self.belief_origin_y = -np.round(initial_cell[1] * self.cell_size, 1)  # meter

        self.flight_info = Map_info(self.belief_flight, self.belief_origin_x, self.belief_origin_y, self.cell_size)
        self.ground_info = Map_info(self.belief_ground, self.belief_origin_x, self.belief_origin_y, self.cell_size)
        self.safe_info = Map_info(self.safe_zone, self.belief_origin_x, self.belief_origin_y, self.cell_size)
        self.ground_truth_info = Map_info(self.ground_truth, self.belief_origin_x, self.belief_origin_y, self.cell_size)

        # Episode variables
        self.explored_rate = 0
        self.explored_rate_ground = 0
        self.safe_rate = 0

        if self.plot:
            self.frame_files = []

    def unique_labelling(self, buildings_mask):
        """
        Labels each region with unique values separately, ensuring that disconnected 
        regions with the same pixel value are treated as different objects.
        """
        unique_values = np.unique(buildings_mask)  # Get all unique pixel values

        labeled_mask = np.zeros_like(buildings_mask, dtype=np.int32)  # Final labeled output
        current_label = 1  # Start labeling from 1

        for value in unique_values:
            if value == 0:
                continue  # Ignore background (assuming 0 is background)

            # Create a binary mask for this specific unique value
            binary_mask = (buildings_mask == value)

            # Label each connected region separately
            labeled_submask, num_features = label(binary_mask, return_num=True)

            # Assign unique labels to each detected region
            labeled_submask[labeled_submask > 0] += current_label - 1
            labeled_mask += labeled_submask  # Merge into final labeled mask

            current_label += num_features  # Increment label count for next objects

        return labeled_mask

    def import_ground_truth(self, plot=False):
        map_dir = f'new_map_oct'
        map_list = os.listdir(map_dir)
        map_index = self.episode_index % np.size(map_list)
        self.map_name = map_list[map_index]
        ground_truth = (io.imread(map_dir + '/' + map_list[map_index], 1) * 255).astype(int)

        map_length = ground_truth.shape[1]
        self.border_length = int((map_length - round(self.opening_size / self.cell_size)) / 2)

        env_padding = np.zeros((25, ground_truth.shape[1]))  # For new free starting areas
        ground_truth = np.concatenate((ground_truth, env_padding), axis=0)
        ground_truth[(ground_truth > 0) & (ground_truth <= 30)] = 0

        # Implement border
        ground_truth[-3:, :self.border_length] = 255
        ground_truth[-3:, -self.border_length:] = 255
        ground_truth[:, :10] = 255
        ground_truth[:, -10:] = 255

        # Generate points at fixed intervals between border points      
        # border_points = np.array([[self.border_length, ground_truth.shape[0]-1], [ground_truth.shape[1] - self.border_length, ground_truth.shape[0]-1]])
        # x_coords = np.linspace(border_points[0][0], border_points[1][0], self.num_border_points)
        # y_coords = np.full_like(x_coords, border_points[0][1])  # All points have same y-coordinate
        # self.border_points = np.column_stack((x_coords, y_coords))

        # Clean up ground truth
        ground_truth_no_border = np.copy(ground_truth)
        ground_truth_no_border[ground_truth_no_border == 255] = 0
        buildings_mask = (ground_truth_no_border != 0)

        # labelled_maslab = self.unique_labelling(buildings_mask)
        labeled_mask, num_features = label(buildings_mask, return_num=True)

        if num_features > 1:
            for label_idx in range(1, num_features + 1):
                component = (labeled_mask == label_idx)
                if component.sum() < self.min_building_size:  # Identify small free areas
                    ground_truth[component] = 0

        # plt.subplot(1,2,1)
        # plt.title('Ground Truth')
        # plt.imshow(ground_truth, cmap='gray')

        # plt.subplot(1,2,2)
        # plt.imshow(labelled_maslab, cmap='nipy_spectral')
        # plt.title('Labeled Components')
        # plt.colorbar()

        # plt.show()
        # print('here')
        # Identify robot height
        ground_truth_filtered = ground_truth[
            (ground_truth < 255) & (ground_truth > 0)]  # Filter the array to include only values less than 255 (border)
        self.max_height_cell = np.max(ground_truth_filtered)

        self.robot_height_cell = np.round(np.percentile(ground_truth_filtered, self.building_percentile))
        self.robot_height = get_metre_hgt(self.robot_height_cell, self.max_height_cell)

        if plot:
            map_dir_colour = f'new_map_colour_oct'
            colour_ground_truth = (io.imread(map_dir_colour + '/' + map_list[map_index]))
        else:
            colour_ground_truth = None

        return ground_truth, colour_ground_truth

    def generate_robot_starts(self):

        free, _ = get_updating_node_coords(np.array([0.0, 0.0]), self.flight_info, robot_height=self.robot_height_cell,
                                           show=False, check_connectivity=False)
        min_value = np.min(free[:, 0])
        max_value = np.max(free[:, 0])
        interval = abs(max_value - min_value) / (self.n_agents - 1)
        robots_positions = np.zeros((self.n_agents, 2))

        for i in range(self.n_agents):
            target_x = min_value + interval * i
            x_value = free[np.argmin(np.abs(free[:, 0] - target_x)), 0]
            robots_positions[i] = free[free[:, 0] == x_value][0]

        return robots_positions

    def init_episode(self, node_manager):
        robot_positions = self.generate_robot_starts()
        self.initial_locations = robot_positions
        self.robot_locations = robot_positions
        self.node_manager = node_manager

        # Update initial flight map for all robots
        self.robot_cells = get_cell_position_from_coords(self.robot_locations, self.flight_info)
        for robot_cell in self.robot_cells:
            self.update_robot_belief(robot_cell)

        # Update new free area in unknown area
        self.ground_info.map[(self.flight_info.map >= 0) & (self.ground_info.map == -127)] = self.flight_info.map[
            (self.flight_info.map >= 0) & (self.ground_info.map == -127)]

        for cell in self.robot_cells:
            self.update_ground(cell)

        for cell in self.robot_cells:
            self.update_visible_occlusion(cell)

        # Find connected components of zeros
        labeled_array = label(self.belief_ground == 0)

        # Get sizes of each region
        region_sizes = np.bincount(labeled_array.ravel())[1:]

        # Find label of largest region (excluding background label 0)
        largest_region_label = np.argmax(region_sizes) + 1

        # Set safe zone to 255 only for largest region
        self.safe_zone[labeled_array == largest_region_label] = 255
        ############################################
        # Function to visualise robot belief
        ############################################
        # plt.subplot(2,2,1)
        # plt.title('Safe zone')
        # plt.imshow(self.safe_info.map, cmap='gray')
        # plt.subplot(2,2,2)
        # plt.title('Flight map')
        # robot_belief = np.copy(self.flight_info.map)
        # plt.scatter(self.robot_cells[:, 0], self.robot_cells[:, 1], c='b', s=15, zorder=6)
        # # robot_belief[(robot_belief < self.robot_height_cell - FLT_SAFETY_BUFFER*get_hgt_scaling()) & (robot_belief > 0)] = 0
        # plt.imshow(robot_belief, cmap='gray')
        # plt.subplot(2,2,3)
        # plt.title('Ground map')
        # plt.imshow(self.ground_info.map, cmap='gray')
        # plt.scatter(self.robot_cells[:, 0], self.robot_cells[:, 1], c='b', s=15, zorder=6)
        # row_indices, col_indices = np.where(self.ground_info.map == -1)
        # if len(row_indices) > 0:
        #     plt.scatter(col_indices, row_indices, c='r', s=1, zorder=5)
        # plt.subplot(2,2,4)
        # plt.title('Ground map')
        # plt.imshow(self.ground_info.map, cmap='gray')
        # plt.show()

        # Copy belief
        self.old_belief_flight = np.copy(self.belief_flight)
        self.old_belief_ground = np.copy(self.belief_ground)
        self.old_safe_zone = np.copy(self.safe_zone)

        # Compute frontiers
        self.global_frontiers = get_frontier_in_map(self.flight_info, self.robot_height_cell)
        self.global_frontiers_ground = get_frontier_in_map_ground(self.ground_info, self.max_height_cell)
        self.global_safe_zone_frontiers = get_safe_zone_frontier(self.safe_info, self.ground_info)
        self.classify_safe_frontier(self.robot_locations)
        # self.covered_safe_frontiers = deepcopy(self.global_safe_zone_frontiers)
        # self.uncovered_safe_frontiers = [] 
        return self.robot_locations, self.safe_info.map, self.robot_cells

    def update_robot_belief(self, robot_cell):
        exploration_sensor(robot_cell, self.sensor_range_cell, self.belief_flight,
                           self.ground_truth, self.robot_height_cell)

    def update_ground(self, robot_cell):
        # self.belief_ground, self.safe_zone = ground_sensor(robot_cell, self.sensor_range_cell, self.ground_truth,
        #                                                     self.belief_ground, self.safe_zone, self.robot_height_cell)    
        # self.belief_ground, self.safe_zone = ground_sensor_radial(robot_cell, self.sensor_range_cell, self.ground_truth,
        #                                                     self.belief_ground, self.safe_zone, self.robot_height_cell)   
        # Update new free area in unknown area
        # self.ground_info.map[(self.flight_info.map >= 0) & (self.ground_info.map == -127)] =  self.flight_info.map[(self.flight_info.map >= 0) & (self.ground_info.map == -127)] 

        occluded_pixels, valid_occlusion = identify_occlussion(self.belief_flight, robot_cell, self.sensor_range_cell,
                                                               self.robot_height_cell)

        if valid_occlusion:
            mask = (self.belief_ground[occluded_pixels[:, 1], occluded_pixels[:, 0]] == 0) & (
                        self.safe_zone[occluded_pixels[:, 1], occluded_pixels[:, 0]] == 0)
            self.ground_info.map[occluded_pixels[mask][:, 1], occluded_pixels[mask][:, 0]] = -1

    def update_visible_occlusion(self, robot_cell):
        occluded_pixels = np.where(self.ground_info.map == -1)
        # Convert to array of coordinates
        occluded_coords = np.column_stack((occluded_pixels[1], occluded_pixels[0]))  # x,y format
        # Calculate distances from robot_cell to all occluded pixels
        distances = np.sqrt(np.sum((occluded_coords - np.array(robot_cell)) ** 2, axis=1))
        # Filter points within sensor range
        valid_indices = distances <= self.sensor_range_cell
        occluded_pixels = (occluded_pixels[0][valid_indices], occluded_pixels[1][valid_indices])
        for row_index, col_index in zip(*occluded_pixels):
            if not check_collision_3D(robot_cell, (col_index, row_index), self.robot_height_cell, self.flight_info,
                                      self.max_height_cell, use_cell=True, show_map=False, occlussion=True):
                self.ground_info.map[row_index, col_index] = 0

    def update_safe_zone(self, robot_cell):
        self.safe_zone = coverage_sensor(robot_cell, self.sensor_range_cell, self.belief_ground,
                                         self.safe_zone, self.robot_height_cell)

    def get_intersect_area(self, locations_togo):
        robot_cells = get_cell_position_from_coords(self.robot_locations, self.flight_info)
        robot_cells_togo = get_cell_position_from_coords(locations_togo, self.flight_info)
        curr_coverage = np.zeros_like(self.safe_zone)
        next_coverage = np.zeros_like(self.safe_zone)
        for robot_cell in robot_cells:
            curr_coverage = coverage_sensor(robot_cell, self.sensor_range_cell, self.belief_ground,
                                            curr_coverage, self.robot_height_cell)
        for robot_cell in robot_cells_togo:
            next_coverage = coverage_sensor(robot_cell, self.sensor_range_cell, self.belief_ground,
                                            next_coverage, self.robot_height_cell)
        intersection = curr_coverage * next_coverage
        intersection[intersection > curr_coverage.max()] = curr_coverage.max()
        return intersection

    def decrease_safety(self, locations_togo):

        # print('Decrease safety')
        # frontiers_cells = get_cell_position_from_coords(np.array(list(self.global_safe_zone_frontiers)).reshape(-1,3)[:,:2], self.safe_info).reshape(-1, 2)
        # intersect_area = self.get_intersect_area(locations_togo[:,:2])

        # Precompute the array of global safe zone frontiers once.
        global_frontiers = np.array(list(self.global_safe_zone_frontiers)).reshape(-1, 2)
        frontiers_cells = get_cell_position_from_coords(global_frontiers, self.safe_info).reshape(-1, 2)
        # start_time = time.time()
        intersect_area = self.get_intersect_area(locations_togo[:, :2])
        # print(f"Time taken to compute global safe zone frontiers: {time.time() - start_time} seconds")
        # print('Step 1')  
        for frontier_loc, frontier_cell in zip(global_frontiers, frontiers_cells):
            diff = frontier_loc[:2] - locations_togo[:, :2]
            distances_sq = np.sum(diff ** 2, axis=1)
            nearby_agent_indices = np.where(distances_sq <= self.sensor_range)[0]
            nearby_agent_locations = locations_togo[nearby_agent_indices]
            uncovered = True
            # print('Step 2')
            for loc in nearby_agent_locations.reshape(-1, 3):
                if not check_collision_3D(loc, frontier_loc, self.robot_height, self.flight_info, self.max_height_cell):
                    uncovered = False
                    break
            # print('Step 3')
            if uncovered:
                cell_center = [self.safety_range_cell, self.safety_range_cell]
                x_lower, x_upper = frontier_cell[0] - self.safety_range_cell, frontier_cell[
                    0] + self.safety_range_cell + 1  # Due to python indexing
                y_lower, y_upper = frontier_cell[1] - self.safety_range_cell, frontier_cell[
                    1] + self.safety_range_cell + 1

                if x_lower < 0:
                    cell_center[0] += x_lower
                    x_lower = 0
                if x_upper > self.safe_zone.shape[1]:
                    x_upper = self.safe_zone.shape[1]
                if y_lower < 0:
                    cell_center[1] += y_lower
                    y_lower = 0
                if y_upper > self.safe_zone.shape[0]:
                    y_upper = self.safe_zone.shape[0]
                # print('Step 4')
                sub_belief_ground = self.belief_ground[y_lower: y_upper, x_lower: x_upper]
                sub_safe_zone = self.safe_zone[y_lower: y_upper, x_lower: x_upper]
                sub_intersection = intersect_area[y_lower: y_upper, x_lower: x_upper]
                decrease_safety_by_frontier(cell_center, self.safety_range_cell, sub_safe_zone, sub_belief_ground,
                                            sub_intersection)
        # print(f"Time taken for decrease_safety: {time.time() - start_time} seconds")

    def classify_safe_frontier(self, robot_locations):
        self.uncovered_safe_frontiers, self.covered_safe_frontiers = set(), set()
        safe_zone_frontiers = np.array(list(self.global_safe_zone_frontiers)).reshape(-1, 2)

        for frontier_loc in safe_zone_frontiers:
            nearby_agent_indices = np.argwhere(
                np.linalg.norm(frontier_loc - robot_locations[:, :2], axis=1) <= self.sensor_range).reshape(-1)
            nearby_agent_locations = robot_locations[nearby_agent_indices][:, :2]
            uncovered = True

            for location in nearby_agent_locations:
                if not check_collision_3D(location, frontier_loc, self.robot_height, self.flight_info,
                                          self.max_height_cell, use_cell=False, show_map=False):
                    uncovered = False
            if uncovered:
                self.uncovered_safe_frontiers.add(tuple(frontier_loc))
            else:
                self.covered_safe_frontiers.add(tuple(frontier_loc))

    def calculate_team_reward(self, dist_list):
        safety_increase = np.sum(self.safe_zone == 255) - np.sum(self.old_safe_zone == 255)
        # Apply a non-linear reward/penalty scaling: log1p growth for increases, negative log1p penalty for decreases
        # if safety_increase >= 0:
        #     modified_increase = np.log1p(safety_increase)
        # else:
        #     modified_increase = -np.log1p(-safety_increase)

        team_reward = np.ones(self.n_agents) * safety_increase / 350
        self.old_safe_zone = deepcopy(self.safe_zone)

        return team_reward, safety_increase

    def check_done(self):
        border_safe_indcators = self.safe_info.map[-1, self.border_length:-self.border_length]
        safe = False if np.any(border_safe_indcators == 0) else True
        return safe

    def evaluate_exploration_rate(self):
        self.explored_rate = np.sum(self.belief_flight == 0) / np.sum(self.ground_truth == 0)
        self.explored_rate_ground = np.sum(self.belief_ground == 0) / np.sum(self.ground_truth == 0)
        self.safe_rate = np.sum(self.safe_zone == 255) / np.sum(self.ground_truth == 0)

    def step(self, step, next_waypoints, robots, ground_truth_node_manager):

        # print('Start here!')
        intermediate_waypoints = np.linspace(self.robot_locations, next_waypoints, SIM_STEP + 1)[1:]
        intermediate_cells = get_cell_position_from_coords(intermediate_waypoints.reshape(-1, 2),
                                                           self.flight_info).reshape(intermediate_waypoints.shape)
        robot_heights = np.full((intermediate_waypoints.shape[1], 1), self.robot_height)

        next_3D_waypoints = np.hstack((next_waypoints, robot_heights))
        self.decrease_safety(next_3D_waypoints)

        for sim_step, (waypoints, next_cells) in enumerate(zip(intermediate_waypoints, intermediate_cells)):
            # print('1')
            # print((f"Step {step} / {'128'}") + (f" | Sim step {sim_step} / {SIM_STEP}"))
            # start_time = time.time()
            # next_3D_waypoints = np.hstack((waypoints, robot_heights))
            # print('2')
            # Decrease safe zone
            # self.decrease_safety(next_3D_waypoints)
            # print(f"\nDecrease safe zone time: {time.time() - start_time:.4f} seconds")
            self.robot_locations = waypoints
            # print('3')

            # start_time = time.time()
            for cell in next_cells:
                self.update_robot_belief(cell)
            # update_belief_time = time.time() - start_time
            # print(f"Update robot belief time: {update_belief_time:.4f} seconds")
            # print('4')

            # Update new free area in unknown area
            self.ground_info.map[(self.flight_info.map >= 0) & (self.ground_info.map == -127)] = self.flight_info.map[
                (self.flight_info.map >= 0) & (self.ground_info.map == -127)]

            # start_time = time.time()
            for cell in next_cells:
                self.update_ground(cell)
            # update_ground_time = time.time() - start_time
            # print(f"Update ground time: {update_ground_time:.4f} seconds")

            # start_time = time.time()
            for cell in next_cells:
                self.update_visible_occlusion(cell)
            # update_occlusion_time = time.time() - start_time
            # print(f"Update visible occlusion time: {update_occlusion_time:.4f} seconds")

            # start_time = time.time()
            for cell in next_cells:
                self.update_safe_zone(cell)
            # update_safe_time = time.time() - start_time
            # print(f"Update safe zone time: {update_safe_time:.4f} seconds")            # print('5')

            # plt.imshow(self.safe_zone, cmap='gray') 
            # plt.show()

            # start_time = time.time()
            self.global_safe_zone_frontiers = get_safe_zone_frontier(self.safe_info, self.ground_info)
            self.classify_safe_frontier(self.robot_locations)
            # update_safe_time = time.time() - start_time
            # print(f"Classify safe frontier time: {update_safe_time:.4f} seconds")
            # print('6')
            if self.plot:
                # start_time = time.time()
                self.evaluate_exploration_rate()

                self.global_frontiers = get_frontier_in_map(self.flight_info, self.robot_height_cell)
                self.global_frontiers_ground = get_frontier_in_map_ground(self.ground_info, self.max_height_cell)
                # start_time = time.time()
                self.plot_env(step, sim_step, robots, ground_truth_node_manager, next_waypoints)
                # print((f"Plot time: {time.time() - start_time:.4f} seconds"))
                for robot in robots:
                    robot.trajectory_x.append(waypoints[robot.id][0])
                    robot.trajectory_y.append(waypoints[robot.id][1])
            # print('7 \n')
        if not self.plot:
            self.evaluate_exploration_rate()

            self.global_frontiers = get_frontier_in_map(self.flight_info, self.robot_height_cell)
            self.global_frontiers_ground = get_frontier_in_map_ground(self.ground_info, self.max_height_cell)
        return self.safe_rate
        # self.classify_safe_frontier(next_3D_waypoints)
        # frontiers_cell = get_cell_position_from_coords(np.array(list(self.global_frontiers)), self.flight_info)
        # frontiers_ground__cell = get_cell_position_from_coords(np.array(list(self.global_frontiers_ground)).reshape(-1,3)[:,:2], self.ground_info)
        # frontiers_safe_cell = get_cell_position_from_coords(np.array(list(self.global_safe_zone_frontiers)).reshape(-1,3)[:,:2], self.safe_info)

        # print(self.flight_info.map[frontiers_cell[:, 1], frontiers_cell[:, 0]])
        # print(self.ground_info.map[frontiers_ground__cell[:, 1], frontiers_ground__cell[:, 0]])
        # print(self.safe_info.map[frontiers_safe_cell[:, 1], frontiers_safe_cell[:, 0]])

        # plt.subplot(1,3,1)
        # plt.imshow(self.flight_info.map, cmap='gray')
        # plt.scatter(frontiers_cell[:, 0], frontiers_cell[:, 1], c='r', marker='o')
        # plt.subplot(1,3,2)
        # plt.imshow(self.ground_info.map, cmap='gray')
        # plt.scatter(frontiers_ground__cell[:, 0], frontiers_ground__cell[:, 1], c='b', marker='o')
        # plt.subplot(1,3,3)
        # plt.imshow(self.safe_info.map, cmap='gray')
        # plt.scatter(frontiers_safe_cell[:, 0], frontiers_safe_cell[:, 1], c='r', marker='o')
        # plt.show()

        # Check if frontiers have value 0 in respective maps
        # assert np.all(self.flight_info.map[frontiers_cell[:, 1], frontiers_cell[:, 0]] >= 0)
        # assert np.all(self.ground_info.map[frontiers_ground__cell[:, 1], frontiers_ground__cell[:, 0]] == 0)
        # assert np.all(self.safe_info.map[frontiers_safe_cell[:, 1], frontiers_safe_cell[:, 0]] == 255)

    def plot_env(self, step, sim_step, robots, ground_truth_node_manager, assigned_frontiers, node_manager=None):
        plt.figure(figsize=(8, 6))
        color_list = ['r', 'b', 'g', 'y', 'c', 'm', 'k']
        cmap_list = ['Reds', 'Blues', 'Greens', 'YlOrBr', 'ocean_r', 'Purples', 'Greys']
        cell_height = get_cell_hgt(self.robot_height, self.max_height_cell)
        hgt_scaling = get_hgt_scaling(self.max_height_cell)

        plt.subplot(2, 3, 1)
        plt.axis('off')
        plt.title('Actual map', y=-0.12, fontweight='bold')
        plt.imshow(np.where(self.colour_map == [255, 255, 255], [0, 0, 0], self.colour_map), alpha=1)
        for robot in robots:
            c = color_list[robot.id % 7]
            robot_cell = get_cell_position_from_coords(self.robot_locations[robot.id], robot.global_flight_info)
            plt.plot(robot_cell[0], robot_cell[1], c=c, marker='o', markersize=10, zorder=5)

        # plt.subplot(2, 3, 4)
        # plt.axis('off')
        # plt.title('Ground truth', y=-0.08, fontweight='bold')
        # plt.imshow(self.ground_truth, cmap='gray_r')
        # ground_truth_nodes = get_cell_position_from_coords(ground_truth_node_manager.ground_truth_node_coords, ground_truth_node_manager.ground_truth_map_info)
        # plt.scatter(ground_truth_nodes[:, 0], ground_truth_nodes[:, 1], c='g', s=5)

        plt.subplot(2, 3, 2)
        plt.axis('off')
        plt.title('Flight belief', y=-0.08, fontweight='bold')
        robot_belief_flight = self.belief_flight.copy()
        robot_belief_flight[
            (robot_belief_flight < cell_height - FLT_SAFETY_BUFFER * hgt_scaling) & (robot_belief_flight > 0)] = 0
        plt.imshow(robot_belief_flight, cmap='gray_r')
        plt.imshow(robot_belief_flight == -127, cmap=plt.cm.colors.ListedColormap(['none', 'lightblue']), alpha=0.8)

        # Plot flight frontiers
        flight_frontiers = get_cell_position_from_coords(np.array(list(self.global_frontiers)),
                                                         self.flight_info).reshape(-1, 2)
        plt.scatter(flight_frontiers[:, 0], flight_frontiers[:, 1], c='r', s=1)
        for robot in robots:
            c = color_list[robot.id % 7]
            if robot.id == 0:
                nodes = get_cell_position_from_coords(robot.node_coords, robot.global_flight_info)
                plt.scatter(nodes[:, 0], nodes[:, 1], c=robot.utility, zorder=2, s=5)
                # for i in range(len(nodes)):
                #     plt.annotate(f'{robot.utility[i]}', (nodes[i,0], nodes[i,1]), xytext=(2,2), textcoords='offset points', fontsize=5)
                if node_manager:
                    for i in range(len(node_manager.x)):
                        plt.plot((node_manager.x[i] - self.flight_info.map_origin_x) / self.cell_size,
                                 (node_manager.y[i] - self.flight_info.map_origin_y) / self.cell_size, 'tan', zorder=1,
                                 linewidth=1)
            robot_cell = get_cell_position_from_coords(self.robot_locations[robot.id], robot.global_flight_info)
            plt.plot(robot_cell[0], robot_cell[1], c=c, marker='o', markersize=10, zorder=5)

            x = (np.array(robot.trajectory_x) - robot.global_flight_info.map_origin_x) / robot.cell_size
            y = (np.array(robot.trajectory_y) - robot.global_flight_info.map_origin_y) / robot.cell_size

            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            n_segments = len(segments)
            t = np.linspace(0.35, 1.0, n_segments)
            lc = LineCollection(segments, cmap=cmap_list[robot.id % 7], norm=plt.Normalize(0, 1), linewidth=2)
            lc.set_array(t)
            lc.set_alpha(1.0)
            plt.gca().add_collection(lc)

        plt.subplot(2, 3, 3)
        plt.axis('off')
        plt.title('Ground belief', y=-0.08, fontweight='bold')
        robot_belief = self.belief_ground.copy()
        plt.imshow(robot_belief, cmap='gray_r')
        plt.imshow(robot_belief == -127, cmap=plt.cm.colors.ListedColormap(['none', 'lightblue']), alpha=0.8)

        # Plot ground frontiers
        frontiers_ground = get_cell_position_from_coords(np.array(list(self.global_frontiers_ground)),
                                                         self.ground_info).reshape(-1, 2)
        plt.scatter(frontiers_ground[:, 0], frontiers_ground[:, 1], c='b', s=2, zorder=6)

        # Plot occluded areas
        row_indices, col_indices = np.where(self.ground_info.map == -1)
        if len(row_indices) > 0:
            plt.scatter(col_indices, row_indices, c='tab:orange', s=1, zorder=2)

        # Plot robot location
        for robot in robots:
            # nodes = get_cell_position_from_coords(robot.node_coords, robot.global_flight_info)
            # plt.scatter(nodes[:, 0], nodes[:, 1], c=robot.utility_ground, zorder=2, s=5)
            # for i in range(len(nodes)):
            #         plt.annotate(f'{robot.utility_ground[i]}', (nodes[i,0], nodes[i,1]), xytext=(2,2), textcoords='offset points', fontsize=5) 
            c = color_list[robot.id % 7]
            robot_cell = get_cell_position_from_coords(self.robot_locations[robot.id], robot.global_flight_info)
            plt.plot(robot_cell[0], robot_cell[1], c=c, marker='o', markersize=10, zorder=5)

            # plt.plot((np.array(robot.trajectory_x) - robot.global_flight_info.map_origin_x) / robot.cell_size,
            #         (np.array(robot.trajectory_y) - robot.global_flight_info.map_origin_y) / robot.cell_size, c,
            #         linewidth=2, zorder=1)

        # Plot robot location and trajectory
        # for robot in self.robot_list:
        #     c = color_list[robot.id%7]
        #     robot_cell = get_cell_position_from_coords(robot.location, robot.global_flight_info)
        #     plt.plot(robot_cell[0], robot_cell[1], c+'o', markersize=10, zorder=5)
        #     plt.plot((np.array(robot.trajectory_x) - robot.global_flight_info.map_origin_x) / robot.cell_size,
        #              (np.array(robot.trajectory_y) - robot.global_flight_info.map_origin_y) / robot.cell_size, c,
        #              linewidth=2, zorder=1)
        #     # if robot.id == 0:
        #     nodes = get_cell_position_from_coords(robot.node_coords, robot.global_flight_info)
        #     plt.scatter(nodes[:, 0], nodes[:, 1], c=robot.border_indicators, zorder=2)
        # for i in range(len(self.local_node_manager.x)):
        #   plt.plot((self.local_node_manager.x[i] - self.env.flight_info.map_origin_x) / robot.cell_size,
        #            (self.local_node_manager.y[i] - self.env.flight_info.map_origin_y) / robot.cell_size, 'tan', zorder=1)

        # Plot selected frontiers in a different color (magenta)
        if len(assigned_frontiers) > 0:
            selected_frontiers_array = np.array(assigned_frontiers)
            # Convert the selected frontiers to cell coordinates using the same flight_info
            selected_frontiers_cells = get_cell_position_from_coords(selected_frontiers_array, self.flight_info)
            plt.scatter(selected_frontiers_cells[:, 0], selected_frontiers_cells[:, 1],
                        c='magenta', s=15, edgecolors='k', zorder=10, label='Selected Frontiers')
            # Optional: add a legend if you want to identify the selected frontiers
            plt.legend(loc='upper right')
        # -------------------------------

        plt.subplot(2, 3, 4)
        plt.axis('off')
        plt.title('Ground truth', y=-0.08, fontweight='bold')
        plt.imshow(self.ground_truth, cmap='gray_r')
        plt.scatter(nodes[:, 0], nodes[:, 1], c='c', zorder=2, s=5)
        for robot in robots:
            c = color_list[robot.id % 7]
            robot_cell = get_cell_position_from_coords(self.robot_locations[robot.id], robot.global_flight_info)
            plt.plot(robot_cell[0], robot_cell[1], c=c, marker='o', markersize=10, zorder=5)

        for i in range(len(self.node_manager.x)):
            plt.plot((self.node_manager.x[i] - self.flight_info.map_origin_x) / robot.cell_size,
                     (self.node_manager.y[i] - self.flight_info.map_origin_y) / robot.cell_size, 'cyan', linewidth=0.5,
                     zorder=1)

        plt.subplot(2, 3, 5)
        plt.axis('off')
        plt.title('Nodes (Uncovered safe Utility)', y=-0.08, fontweight='bold')
        plt.imshow(robot_belief, cmap='gray_r')
        plt.imshow(robot_belief == -127, cmap=plt.cm.colors.ListedColormap(['none', 'lightblue']), alpha=0.8)
        # if len(self.env.global_safe_zone_frontiers) != 0:
        #     covered_safe_frontier_cells = get_cell_position_from_coords(np.array(list(self.env.global_safe_zone_frontiers)).reshape(-1,3)[:,:2], self.env.safe_info).reshape(-1, 2)
        #     plt.scatter(covered_safe_frontier_cells[:, 0], covered_safe_frontier_cells[:, 1], c='b', s=1, zorder=6)
        # nodes = get_cell_position_from_coords(self.robot_list[0].node_coords, self.robot_list[0].global_flight_info)
        # plt.scatter(nodes[:, 0], nodes[:, 1], c=robots[0].occluded_indicator, zorder=2, s=5)
        # print(robots[0].occluded_indicator)
        # for i in range(len(nodes)):
        #             plt.annotate(f'{robot.utility_uncovered_safe[i]}', (nodes[i,0], nodes[i,1]), xytext=(2,2), textcoords='offset points', fontsize=5) 
        global_safe_zone_frontiers = get_cell_position_from_coords(np.array(list(self.global_safe_zone_frontiers)),
                                                                   self.safe_info).reshape(-1, 2)
        plt.scatter(global_safe_zone_frontiers[:, 0], global_safe_zone_frontiers[:, 1], c='y', s=1, zorder=6)
        for robot in robots:
            if robot.id == 0:
                plt.imshow(robot.global_safe_info.map, cmap='Greens', alpha=0.2)

        for i in range(len(self.node_manager.x)):
            plt.plot((self.node_manager.x[i] - self.flight_info.map_origin_x) / robot.cell_size,
                     (self.node_manager.y[i] - self.flight_info.map_origin_y) / robot.cell_size, 'cyan', linewidth=0.5,
                     zorder=1)

        plt.subplot(2, 3, 6)
        plt.axis('off')
        plt.title('Safe zone', y=-0.08, fontweight='bold')
        plt.imshow(robot_belief, cmap='gray_r')
        plt.imshow(robot_belief == -127, cmap=plt.cm.colors.ListedColormap(['none', 'lightblue']), alpha=0.8)

        # Plot safe frontiers
        # robot_heights = np.full((self.robot_locations.shape[0], 1), self.robot_height)
        # next_3D_waypoints = np.hstack((self.robot_locations, robot_heights))
        if len(self.covered_safe_frontiers) != 0:
            covered_safe_frontier_cells = get_cell_position_from_coords(np.array(list(self.covered_safe_frontiers)),
                                                                        self.safe_info).reshape(-1, 2)
            plt.scatter(covered_safe_frontier_cells[:, 0], covered_safe_frontier_cells[:, 1], c='g', s=1, zorder=6)
            # from sklearn.cluster import DBSCAN
            # clustering = DBSCAN(eps=8, min_samples=1).fit(covered_safe_frontier_cells)
            # unique_labels = np.unique(clustering.labels_[clustering.labels_ >= 0])
            # for label in unique_labels:
            #     cluster_points = covered_safe_frontier_cells[clustering.labels_ == label]
            #     cluster_center = np.mean(cluster_points, axis=0)
            #     plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c='g', s=3, zorder=6)
            #     plt.annotate(f'{len(cluster_points)}', (cluster_center[0], cluster_center[1]), 
            #                xytext=(3,5), textcoords='offset points', fontsize=8, color='g')

        if len(self.uncovered_safe_frontiers) != 0:
            uncovered_safe_frontier_cells = get_cell_position_from_coords(np.array(list(self.uncovered_safe_frontiers)),
                                                                          self.safe_info).reshape(-1, 2)
            plt.scatter(uncovered_safe_frontier_cells[:, 0], uncovered_safe_frontier_cells[:, 1], c='r', s=1, zorder=6)
            # clustering = DBSCAN(eps=8, min_samples=1).fit(uncovered_safe_frontier_cells)
            # unique_labels = np.unique(clustering.labels_[clustering.labels_ >= 0])
            # for label in unique_labels:
            #     cluster_points = uncovered_safe_frontier_cells[clustering.labels_ == label]
            #     cluster_center = np.mean(cluster_points, axis=0)
            #     plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c='r', s=3, zorder=6)
            #     plt.annotate(f'{len(cluster_points)}', (cluster_center[0], cluster_center[1]), 
            #                xytext=(3,5), textcoords='offset points', fontsize=8, color='r')        
        # global_safe_zone_frontiers = get_cell_position_from_coords(np.array(list(self.global_safe_zone_frontiers)).reshape(-1,3)[:,:2], self.safe_info).reshape(-1, 2)
        # plt.scatter(global_safe_zone_frontiers[:, 0], global_safe_zone_frontiers[:, 1], c='y', s=1, zorder=6)

        # Plot occluded areas
        # if len(row_indices) > 0:
        #     plt.scatter(col_indices, row_indices, c='tab:orange', s=1, zorder=2)

        # Plot robot location and nodes 
        for robot in robots:
            c = color_list[robot.id % 7]
            if robot.id == 5:
                # nodes = get_cell_position_from_coords(robot.node_coords, robot.global_flight_info)
                # for i in range(len(nodes)):
                #     plt.annotate(f'{robot.utility_safe[i]}', (nodes[i,0], nodes[i,1]), xytext=(2,2), textcoords='offset points', fontsize=5) 
                # plt.scatter(nodes[:, 0], nodes[:, 1], c=robot.utility_safe, zorder=2, s=5)
                # alpha_mask = robot.global_safe_info.map / 255 / 3
                plt.imshow(robot.global_safe_info.map, cmap='Greens', alpha=0.2)
                # plt.scatter(nodes[:, 0], nodes[:, 1], c=robot.utility_ground, zorder=6)
                # plt.scatter(nodes[:, 0], nodes[:, 1], c=robot.border_indicators, s=8, zorder=2)
                # for i, txt in enumerate(robot.border_indicators):
                #     if txt > 0:
                #         plt.annotate(txt, (nodes[i, 0], nodes[i,1]), zorder =10)

            robot_cell = get_cell_position_from_coords(self.robot_locations[robot.id], robot.global_flight_info)
            plt.plot(robot_cell[0], robot_cell[1], c=c, marker='o', markersize=10, zorder=5)
            circle = plt.Circle((robot_cell[0], robot_cell[1]), robot.sensor_range_cell, color=c, fill=False, alpha=0.3,
                                zorder=1)
            ax = plt.gca()
            ax.add_patch(circle)
            ax.set_xlim(0, robot.global_flight_info.map.shape[0] - 15)
            ax.set_ylim(robot.global_flight_info.map.shape[1] + 15, 0)

        plt.suptitle(
            'Explored ratio: {:.4g}   |    Explored ratio ground: {:4g}    \n    Safe ratio: {:4g}    |    Travel distance: {:.4g}'.format(
                self.explored_rate,
                self.explored_rate_ground,
                self.safe_rate,
                max([robot.travel_dist for robot in
                     robots])), fontweight='bold')
        plt.tight_layout()
        # plt.gcf().patch.set_facecolor('green')
        # plt.show()
        # if step == -1:
        #     plt.show()
        plt.savefig('{}/{}_{}_{}_samples.png'.format(self.save_path, self.episode_index, step, sim_step), dpi=150)
        plt.close()
        frame = '{}/{}_{}_{}_samples.png'.format(self.save_path, self.episode_index, step, sim_step)
        self.frame_files.append(frame)
