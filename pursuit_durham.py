import copy
import random
import time

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import ray
import csv
from copy import deepcopy
import copy
from old_env import Env
from old_agent import Agent
from ground_truth_node_manager import GroundTruthNodeManager
from utils import *
from node_manager import Node_manager
from sensor import coverage_sensor

TEST_N_AGENTS = 6
EXPLORATION = True
# GROUP_START: change in test_parameter.py
# MIN_UTILITY = 0  # !! change in parameter.py
RAY_META_AGENT = 8

NUM_TEST = 1
MAX_EPISODE_STEP = 30
# SENSOR_RANGE = 20  # 7.9812
SENSOR_RANGE = 30  # 7.9812
CELL_SIZE = 1
SAVE_IMG = True
SAVE_CSV = False
gifs_path = 'results/gifs'

if not os.path.exists(gifs_path):
    os.makedirs(gifs_path)


class DurhamWorker:
    def __init__(self, meta_agent_id, global_step, save_image=False):
        self.device = 'cpu'
        self.meta_agent_id = meta_agent_id
        self.global_step = global_step
        self.save_image = save_image
        np.random.seed(123)
        self.selected_frontiers = []
        self.env = Env(self.global_step, n_agents=TEST_N_AGENTS, plot=True, save_path=gifs_path)
        self.robot_height = self.env.robot_height
        self.node_manager = Node_manager(self.robot_height, self.env.robot_height_cell, self.env.max_height_cell,
                                         quad_tree_size=QUAD_TREE_SIZE,
                                         sensor_range=self.env.sensor_range, plot=self.save_image)
        self.robot_list = [Agent(i, None, self.node_manager, self.robot_height, self.env.robot_height_cell,
                                 self.env.max_height_cell, self.device, self.save_image) for i in
                           range(self.env.n_agents)]
        self.ground_truth_node_manager = GroundTruthNodeManager(self.node_manager, self.env.ground_truth_info,
                                                                self.env.sensor_range, self.robot_height,
                                                                self.env.robot_height_cell, self.env.max_height_cell,
                                                                device=self.device, plot=self.save_image)

        # self.env.init_robot_start()
        self.env.init_episode(self.node_manager)
        self.utility = None
        self.agent_status = [0] * self.env.n_agents  # 0: follower, 1: frontier guard
        self.agent_next_location = [None] * self.env.n_agents
        self.agent_guard_post = [None] * self.env.n_agents  # Track assigned guard positions
        self.guard_verified = [False] * self.env.n_agents   # Track safeguard verification

    def assign_frontier_guard(self, best_locations):
        frontier_guards = []
        for location in best_locations:
            min_dist = float('inf')
            frontier_guard = None
            best_path = None
            
            for robot in self.robot_list:
                # Skip if already guarding and not verified
                if (self.agent_status[robot.id] == 1 
                    and not self.guard_verified[robot.id]):
                    continue
                    
                # Get path and handle failures
                path_coords, dist = self.node_manager.a_star(robot.location, location)
                
                # Skip if no path found (critical fix)
                if path_coords is None or len(path_coords) == 0:
                    print(f"No path for robot {robot.id} to location {location}")
                    continue
                    
                if dist < min_dist:
                    min_dist = dist
                    frontier_guard = robot.id
                    best_path = path_coords
            
            # Handle cases where no robot/path was found
            if frontier_guard is not None:
                frontier_guards.append(frontier_guard)
                self.agent_status[frontier_guard] = 1
                self.agent_guard_post[frontier_guard] = location  # Track guard post
                self.guard_verified[frontier_guard] = False       # Reset verification
                self.agent_next_location[frontier_guard] = best_path[0] if best_path else robot.location

    # def assign_frontier_guard(self, best_locations):
    #     # Maintain current guards while adding new ones
    #     current_guards = [i for i, status in enumerate(self.agent_status) if status == 1]
    #     new_guard_positions = []
        
    #     # Preserve current guard positions
    #     for idx in current_guards:
    #         new_guard_positions.append(self.robot_list[idx].location.tolist())
        
    #     # Add new guard positions
    #     for location in best_locations:
    #         if len(new_guard_positions) >= self.env.n_agents:
    #             break  # No more robots available
    #         new_guard_positions.append(location)
        
    #     # Assign robots to guard positions
    #     frontier_guards = []
    #     for location in new_guard_positions:
    #         min_dist = 1e6
    #         frontier_guard = None
            
    #         for robot in self.robot_list:
    #             if robot.id in frontier_guards:
    #                 continue
                    
    #             path_coords, dist = self.node_manager.a_star(robot.location, location)
    #             if dist < min_dist:
    #                 min_dist = dist
    #                 frontier_guard = robot.id
    #                 best_path = path_coords
            
    #         if frontier_guard is not None:
    #             frontier_guards.append(frontier_guard)
    #             self.agent_status[frontier_guard] = 1
    #             # Only move if not already at guard position
    #             if not np.allclose(robot.location, location, atol=1e-5):
    #                 self.agent_next_location[frontier_guard] = best_path[0] if best_path else location

    def verify_guard_coverage(self):
        for robot_id in range(self.env.n_agents):
            if self.agent_status[robot_id] == 1:  # Only check guards
                guard_loc = self.robot_list[robot_id].location
                # Check if assigned frontier is covered
                if self.agent_guard_post[robot_id] is not None:
                    dist = np.linalg.norm(guard_loc - self.agent_guard_post[robot_id])
                    self.guard_verified[robot_id] = (dist < 1e-5)  # Verified if at post


    def assign_follower(self):
        # Followers only replace guards when in position
        for robot in self.robot_list:
            if self.agent_status[robot.id] == 0:  # Follower
                min_dist = 1e6
                best_path = None
                
                # Find nearest guard that needs replacement
                for guard_id, status in enumerate(self.agent_status):
                    if status == 1:
                        guard_robot = self.robot_list[guard_id]
                        # Only approach guards that are stationary
                        if np.allclose(guard_robot.location, self.agent_next_location[guard_id], atol=1e-5):
                            path_coords, dist = self.node_manager.a_star(robot.location, guard_robot.location)
                            if dist < min_dist:
                                min_dist = dist
                                best_path = path_coords
                                target_guard = guard_id
                
                if best_path:
                    self.agent_next_location[robot.id] = best_path[0]
                    # Guard can be released when follower is close
                    if min_dist < SENSOR_RANGE/2:
                        self.agent_status[target_guard] = 0  # Release guard
                else:
                    # Fallback to nearest frontier
                    if self.env.global_safe_zone_frontiers:
                        frontiers = np.array(list(self.env.global_safe_zone_frontiers))
                        dists = np.linalg.norm(frontiers - robot.location, axis=1)
                        nearest_idx = np.argmin(dists)
                        path_coords, _ = self.node_manager.a_star(robot.location, frontiers[nearest_idx])
                        self.agent_next_location[robot.id] = path_coords[0] if path_coords else robot.location



    def check_frontier_coverage(self, selected_locations):
        selected_locations = np.array(selected_locations).reshape(-1, 2)
        n_selected = selected_locations.shape[0]
        
        # Initialize coverage tracking
        selected_frontier_uncoverage = np.zeros(n_selected)
        frontier_coverage_status = [False] * len(self.env.global_safe_zone_frontiers)
        
        # Check coverage for each frontier point
        for f_idx, frontier_loc in enumerate(self.env.global_safe_zone_frontiers):
            covered = False
            for s_idx, loc in enumerate(selected_locations):
                distance = np.linalg.norm(loc - frontier_loc)
                if distance < SENSOR_RANGE:
                    if not check_collision(frontier_loc, loc, self.robot_height,
                                        map_info=self.env.flight_info,
                                        max_cell_height=self.env.max_height_cell):
                        covered = True
                        break
            if not covered:
                frontier_coverage_status[f_idx] = False
                # Find nearest selected location to this uncovered frontier
                distances = np.linalg.norm(selected_locations - frontier_loc, axis=1)
                nearest_idx = np.argmin(distances)
                if nearest_idx < n_selected:
                    selected_frontier_uncoverage[nearest_idx] += 1
            else:
                frontier_coverage_status[f_idx] = True

        # Adjust problematic viewpoints
        for i in range(n_selected):
            if selected_frontier_uncoverage[i] > 0:  # Changed threshold to >0
                # Find closest robot to this viewpoint
                robot_distances = np.linalg.norm(self.env.robot_locations - selected_locations[i], axis=1)
                closest_robot_idx = np.argmin(robot_distances)
                
                # Get new viewpoint near the robot (Durham's local frontier approach)
                new_viewpoint = self.get_viewpoint_near_robot(
                    self.env.robot_locations[closest_robot_idx],
                    self.env.global_safe_zone_frontiers
                )
                selected_locations[i] = new_viewpoint

        return selected_locations, frontier_coverage_status

    def get_viewpoint_near_robot(self, robot_position, frontiers):
        """Durham-style viewpoint selection near robot position"""
        # Convert set to list for indexing
        frontiers_list = list(frontiers) if isinstance(frontiers, set) else frontiers
        
        # Find nearest frontier to robot
        frontier_dists = [np.linalg.norm(np.array(f) - np.array(robot_position)) 
                        for f in frontiers_list]
        nearest_frontier = frontiers_list[np.argmin(frontier_dists)]
        
        # Move toward frontier while maintaining min distance (d_min)
        direction = np.array(nearest_frontier) - np.array(robot_position)
        distance = np.linalg.norm(direction)
        unit_vector = direction / distance if distance > 0 else np.array([1, 0])
        
        # Calculate optimal viewpoint (Durham's d_min parameter)
        move_distance = min(SENSOR_RANGE * 0.7, distance * 0.8)
        return robot_position + unit_vector * move_distance



    def update_imaginary_safe_zone(self, robot_cell, imaginary_safe_zone):
        # 创建地面信念图（belief_ground）的深拷贝，以避免修改原始数据
        padded_robot_belief = deepcopy(self.env.belief_ground)

        # 将信念图中值为127的区域设置为255
        # 这可能表示将潜在危险区域暂时视为安全区域进行估计
        padded_robot_belief[padded_robot_belief == 127] = 255

        # 使用coverage_sensor函数更新imaginary_safe_zone
        # 传入机器人所在单元格、传感器范围（以单元格为单位）、imaginary_safe_zone的深拷贝、调整后的地面信念图和机器人高度
        new_imaginary_safe_zone = coverage_sensor(robot_cell, round(SENSOR_RANGE / CELL_SIZE),
                                                  deepcopy(imaginary_safe_zone), padded_robot_belief, robot_height=self.robot_height)

        # 计算更新前后imaginary_safe_zone中值为255的单元格数量变化
        # 这表示安全区域的增加量
        safe_increase = np.sum(new_imaginary_safe_zone == 255) + np.sum(new_imaginary_safe_zone == 0) - np.sum(imaginary_safe_zone == 255)

        # 返回更新后的imaginary_safe_zone和安全区域的增加量
        return new_imaginary_safe_zone, safe_increase

    # def find_next_best_views(self):
    #     # all_node_coords = self.robot_list[0].node_coords
    #     all_node_coords = self.robot_list[0].node_coords
    #     self.utility = self.robot_list[0].utility
    #     non_zero_utility_node_indices = np.argwhere(self.utility > 0)[:, 0].tolist()
    #     if len(non_zero_utility_node_indices) == 0:
    #         return None
    #     assert len(non_zero_utility_node_indices) > 0
    #     candidate_node_coords = all_node_coords[non_zero_utility_node_indices]
    #     imaginary_safe_zone = self.env.safe_zone
    #     best_imaginary_safe_zone = self.env.safe_zone
    #     max_increase = 1
    #     best_locations = []
    #     filtered_indices = []
    #     best_coords = None

    #     while True:
    #         for i, coords in enumerate(candidate_node_coords):
    #             if i in filtered_indices:
    #                 continue
    #             candidate_flag = True
    #             for location in best_locations:
    #                 if location is None:
    #                     pass
    #                 dist_to_candidate = np.linalg.norm(location - coords)
    #                 if (dist_to_candidate < SENSOR_RANGE) and (
    #                         # not check_collision(location, coords, self.env.belief_ground,
    #                         not check_collision(location, coords, self.robot_height,
    #                                             map_info=self.env.flight_info,
    #                                             max_cell_height=self.env.max_height_cell)):
    #                     candidate_flag = False
    #                     break
    #             if candidate_flag:
    #                 robot_cell = get_cell_position_from_coords(coords, self.env.ground_info)
    #                 new_imaginary_safe_zone, safe_increase = self.update_imaginary_safe_zone(robot_cell,
    #                                                                                          imaginary_safe_zone)
    #                 if safe_increase > max_increase:
    #                     max_increase = safe_increase
    #                     best_coords = coords
    #                     best_imaginary_safe_zone = new_imaginary_safe_zone
    #             else:
    #                 filtered_indices.append(i)

    #         if len(best_locations) > 0 and np.all(best_coords == best_locations[-1]):
    #             return best_locations

    #         imaginary_safe_zone = best_imaginary_safe_zone
    #         if best_coords is None:
    #             return None
    #         best_locations.append(best_coords)
    #         max_increase = 1

    def find_next_best_views(self):
        all_node_coords = self.robot_list[0].node_coords
        self.utility = self.robot_list[0].utility
        non_zero_utility_node_indices = np.argwhere(self.utility > 0)[:, 0].tolist()
        
        if len(non_zero_utility_node_indices) == 0:
            print("Not able to find next best view - using current positions")
            return [robot.location.tolist() for robot in self.robot_list]
        
        candidate_node_coords = all_node_coords[non_zero_utility_node_indices]
        imaginary_safe_zone = self.env.safe_zone
        best_imaginary_safe_zone = self.env.safe_zone
        max_increase = 2
        best_locations = []
        processed_candidates = set()
        iteration = 0
        max_iterations = 10  # Safety limit
        
        while iteration < max_iterations:
            iteration += 1
            best_coords = None
            best_increase = max_increase
            
            for i, coords in enumerate(candidate_node_coords):
                if tuple(coords) in processed_candidates:
                    continue
                    
                # Skip if candidate is too close to robot's current position
                robot_distances = [np.linalg.norm(coords - robot.location) for robot in self.robot_list]
                if min(robot_distances) < 1e-5:  # Essentially current position
                    processed_candidates.add(tuple(coords))
                    continue
                    
                candidate_flag = True
                for location in best_locations:
                    if location is None:
                        continue
                    dist_to_candidate = np.linalg.norm(location - coords)
                    if (dist_to_candidate < SENSOR_RANGE) and (
                        not check_collision(location, coords, self.robot_height,
                                            map_info=self.env.flight_info,
                                            max_cell_height=self.env.max_height_cell)):
                        candidate_flag = False
                        break
                        
                if not candidate_flag:
                    processed_candidates.add(tuple(coords))
                    continue
                    
                robot_cell = get_cell_position_from_coords(coords, self.env.ground_info)
                new_imaginary_safe_zone, safe_increase = self.update_imaginary_safe_zone(robot_cell, imaginary_safe_zone)
                
                if safe_increase > best_increase:
                    best_increase = safe_increase
                    best_coords = coords
                    best_imaginary_safe_zone = new_imaginary_safe_zone
                    
            if best_coords is None:
                break  # No improvement found
                
            # Add only if not current position (redundant but safe)
            robot_distances = [np.linalg.norm(best_coords - robot.location) for robot in self.robot_list]
            if min(robot_distances) > 1e-5:
                best_locations.append(best_coords)
                processed_candidates.add(tuple(best_coords))
                
            imaginary_safe_zone = best_imaginary_safe_zone
            max_increase = best_increase  # Maintain improvement threshold
            
        if len(best_locations) == 0:
            print("No valid viewpoints found - using current positions")
            return [robot.location.tolist() for robot in self.robot_list]
            
        return best_locations


    # def run_episode(self):
    #     t = time.time()
    #     # for robot in self.robot_list:
    #     for i, robot in enumerate(self.robot_list):
    #         robot.update_graph(self.env.flight_info, self.env.ground_info, deepcopy(self.env.robot_locations[robot.id]),
    #                            self.env.safe_info, self.env.global_frontiers, self.env.global_frontiers_ground,
    #                            self.env.global_safe_zone_frontiers,
    #                            self.env.uncovered_safe_frontiers, self.env.robot_locations, k=i)
    #
    #     for robot in self.robot_list:
    #         robot.update_past_locations()
    #
    #     for robot in self.robot_list:
    #         robot.update_planning_state(self.env.robot_locations)
    #
    #     if self.save_image:
    #         # self.env.plot_env(-1, -1, self.robot_list, self.node_manager,self.selected_frontiers)
    #         self.env.plot_env(-1, -1, self.robot_list, self.ground_truth_node_manager, self.selected_frontiers,
    #                           self.node_manager)
    #
    #     max_travel_dist = 0
    #
    #     length_list = [max_travel_dist]
    #     safe_rate_list = [self.env.safe_rate]
    #     explored_rate_list = [self.env.explored_rate]
    #
    #     for i in range(MAX_EPISODE_STEP):
    #         done = False
    #         self.agent_status = [0] * self.env.n_agents
    #         best_locations = self.find_next_best_views()
    #         if best_locations is None:
    #             continue
    #         self.assign_frontier_guard(best_locations)
    #         self.assign_follower()
    #
    #         next_locations = []
    #         dist_list = []
    #         for robot in self.robot_list:
    #             next_location = self.agent_next_location[robot.id]
    #             next_locations.append(next_location)
    #             dist_list.append(np.linalg.norm(next_location - robot.location))
    #
    #         next_locations = self.solve_path_confict(next_locations, dist_list)
    #
    #         selected_locations = np.array(next_locations).tolist()
    #         safe_ratio = self.env.step(i, selected_locations, self.robot_list, self.ground_truth_node_manager)
    #         if NUM_TEST == 1:
    #
    #             frontier_guards = [robot.id for robot in self.robot_list if self.agent_status[robot.id] == 1]
    #             followers = [robot.id for robot in self.robot_list if self.agent_status[robot.id] == 0]
    #
    #             print(f"Step {i}\tFrontier guards: {frontier_guards}\tFollowers: {followers}")
    #
    #         if self.save_image:
    #             for robot in self.robot_list:
    #                 # nodes = get_cell_position_from_coords(robot.location, robot.global_safe_info)
    #                 nodes = get_cell_position_from_coords(robot.location, robot.global_flight_info)
    #                 if nodes.ndim == 1:
    #                     continue
    #             # self.env.plot_env(i, SIM_STEP, self.robot_list, self.node_manager)
    #             self.env.plot_env(i, SIM_STEP, self.robot_list, self.ground_truth_node_manager,
    #             self.selected_frontiers)
    #
    #
    #         tmp_safe_zone_frontier = copy(self.env.global_safe_zone_frontiers)
    #         for _ in range(8):
    #             self.env.decrease_safety(next_locations)
    #             self.env.global_safe_zone_frontiers = get_safe_zone_frontier(self.env.safe_info, self.env.flight_info)
    #             if np.array_equal(tmp_safe_zone_frontier, self.env.global_safe_zone_frontiers):
    #                 break
    #             else:
    #                 tmp_safe_zone_frontier = copy(self.env.global_safe_zone_frontiers)
    #
    #
    #         self.env.classify_safe_frontier(next_locations)
    #
    #         for robot in self.robot_list:
    #             robot.update_graph(self.env.flight_info, self.env.ground_info,
    #                                deepcopy(self.env.robot_locations[robot.id]),
    #                                self.env.safe_info, self.env.global_frontiers, self.env.global_frontiers_ground,
    #                                self.env.global_safe_zone_frontiers,
    #                                self.env.uncovered_safe_frontiers, self.env.robot_locations, k=i)
    #
    #         for robot in self.robot_list:
    #             robot.update_planning_state(self.env.robot_locations)
    #
    #         max_travel_dist += np.max(dist_list)
    #         if safe_ratio > 0.96:
    #             done = True
    #
    #         length_list.append(max_travel_dist)
    #         safe_rate_list.append(self.env.safe_rate)
    #         explored_rate_list.append(self.env.explored_rate)
    #         if max_travel_dist >= 1000:
    #             max_travel_dist = 1000
    #             break
    #
    #         if done:
    #             if self.save_image:
    #                 # self.env.plot_env(i + 1, SIM_STEP, self.robot_list, self.ground_truth_node_manager)
    #                 self.env.plot_env(i + 1, 0, self.robot_list, self.ground_truth_node_manager,
    #                                   self.selected_frontiers)
    #                 # self.plot_local_env(i + 1)
    #             break
    #
    #     if NUM_TEST == 1:
    #         print(
    #             f"{TEST_N_AGENTS} agents, max travelled distance: {max_travel_dist}, explored rate: {self.env.explored_rate}, cleared rate: {self.env.safe_rate}")
    #
    #     if self.save_image:
    #         make_gif(gifs_path, self.global_step, self.env.frame_files, self.env.explored_rate)
    #
    #     perf_metrics = [max_travel_dist, self.env.explored_rate, self.env.safe_rate, done, length_list, safe_rate_list,
    #                     explored_rate_list]
    #     print(time.time() - t)
    #     return perf_metrics
    
    def run_episode(self): 
        t = time.time()
        for i, robot in enumerate(self.robot_list):
            robot.update_graph(
                self.env.flight_info,
                self.env.ground_info,
                deepcopy(self.env.robot_locations[robot.id]),
                self.env.safe_info,
                self.env.global_frontiers,
                self.env.global_frontiers_ground,
                self.env.global_safe_zone_frontiers,
                self.env.uncovered_safe_frontiers,
                self.env.robot_locations,
                k=i
            )
        for robot in self.robot_list:
            robot.update_planning_state(self.env.robot_locations)
        # if self.save_image:
        #     self.plot_local_env(-1)
        max_travel_dist = 0

        length_list = [max_travel_dist]
        safe_rate_list = [self.env.safe_rate]
        explored_rate_list = [self.env.explored_rate]

        for i in range(MAX_EPISODE_STEP):
            done = False
            self.verify_guard_coverage()
            
            # Only proceed if guards are verified
            # if all(self.guard_verified) or not any(self.agent_status):
            self.agent_status = [0] * self.env.n_agents
            best_locations = self.find_next_best_views()
            print('best locations',best_locations)
            if best_locations is None:
                print('No best locations')
                continue
            self.assign_frontier_guard(best_locations)
            self.assign_follower()

            next_locations = []
            dist_list = []
            for robot in self.robot_list:
                next_location = self.agent_next_location[robot.id]
                next_locations.append(next_location)
                dist_list.append(np.linalg.norm(next_location - robot.location))
            print('next locations', next_locations)
            selected_locations, _ = self.check_frontier_coverage(next_locations)
            # selected_locations = next_locations
            print('selected locations:', selected_locations)
            print('robot location', [robot.location.tolist() for robot in self.robot_list])

            if NUM_TEST == 1:
                frontier_guards = [robot.id for robot in self.robot_list if self.agent_status[robot.id] == 1]
                followers = [robot.id for robot in self.robot_list if self.agent_status[robot.id] == 0]
                print(f"Step {i}\tFrontier guards: {frontier_guards}\tFollowers: {followers}")

            if self.save_image:
                for robot in self.robot_list:
                    nodes = get_cell_position_from_coords(robot.location, robot.global_safe_info)
                    if nodes.ndim == 1:
                        continue
                # self.plot_local_env(i)

            # selected_locations = self.solve_path_confict(selected_locations, dist_list)

            # self.env.decrease_safety(selected_locations)

            tmp_safe_zone_frontier = deepcopy(self.env.global_safe_zone_frontiers)
            # for _ in range(8):
            #     self.env.decrease_safety(selected_locations)
            #     self.env.global_safe_zone_frontiers = get_safe_zone_frontier(self.env.safe_info, self.env.flight_info)
            #     if np.array_equal(tmp_safe_zone_frontier, self.env.global_safe_zone_frontiers):
            #         break
            #     else:
            #         tmp_safe_zone_frontier = deepcopy(self.env.global_safe_zone_frontiers)

            self.env.step(i, selected_locations, self.robot_list, self.ground_truth_node_manager)

            self.env.classify_safe_frontier(selected_locations)

            for robot in self.robot_list:
                robot.update_graph(self.env.flight_info,
                self.env.ground_info,
                deepcopy(self.env.robot_locations[robot.id]),
                self.env.safe_info,
                self.env.global_frontiers,
                self.env.global_frontiers_ground,
                self.env.global_safe_zone_frontiers,
                self.env.uncovered_safe_frontiers,
                self.env.robot_locations,
                k=i)
            # for robot in self.robot_list:
            #     robot.update_safe_graph(self.env.safe_info, self.env.uncovered_safe_frontiers,
            #                             self.env.counter_safe_info)
            for robot in self.robot_list:
                robot.update_planning_state(self.env.robot_locations)

            max_travel_dist += np.max(dist_list)

            done = self.env.check_done()

            length_list.append(max_travel_dist)
            safe_rate_list.append(self.env.safe_rate)
            explored_rate_list.append(self.env.explored_rate)

            if max_travel_dist >= 1000:
                max_travel_dist = 1000
                break

            if done:
                if self.save_image:
                    self.plot_local_env(i + 1)
                break

        if NUM_TEST == 1:
            print(
                f"{TEST_N_AGENTS} agents, max travelled distance: {max_travel_dist}, explored rate: {self.env.explored_rate}, cleared rate: {self.env.safe_rate}")

        if self.save_image:
            make_gif(gifs_path, self.global_step, self.env.frame_files, self.env.explored_rate)

        perf_metrics = [max_travel_dist, self.env.explored_rate, self.env.safe_rate, done, length_list, safe_rate_list,
                        explored_rate_list]
        print(time.time() - t)
        return perf_metrics

    def solve_path_confict(self, selected_locations, dist_list):
        selected_locations = np.array(selected_locations).reshape(-1, 2)
        arriving_sequence = np.argsort(np.array(dist_list))
        selected_locations_in_arriving_sequence = np.array(selected_locations)[arriving_sequence]
        for j, selected_location in enumerate(selected_locations_in_arriving_sequence):
            solved_locations = selected_locations_in_arriving_sequence[:j]
            while selected_location[0] + selected_location[1] * 1j in solved_locations[:, 0] + solved_locations[:,
                                                                                               1] * 1j:
                id = arriving_sequence[j]
                # nearby_nodes = self.robot_list[id].node_manager.local_nodes_dict.nearest_neighbors(
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

    def plot_local_env(self, step):
        plt.figure(figsize=(8, 6))
        color_list = ['r', 'b', 'g', 'y', 'c', 'm', 'k']
        # cell_height = get_cell_hgt(self.robot_height, self.env.max_height_cell)
        # hgt_scaling = get_hgt_scaling(self.env.max_height_cell)

        plt.subplot(2, 3, 1)
        plt.axis('off')
        plt.title('Actual map', y=-0.12, fontweight='bold')

        plt.imshow(np.where(self.env.colour_map == [255, 255, 255], [0, 0, 0], self.env.colour_map), alpha=1)
        for robot in self.robot_list:
            c = color_list[robot.id % 7]
            # robot_cell = get_cell_position_from_coords(robot.location, robot.global_safe_info)
            # robot_cell = get_cell_position_from_coords(robot.location, robot.global_safe_info)

            robot_cell_downsampled = get_cell_position_from_coords(robot.location, robot.global_safe_info)
            # 将降采样后的坐标转换为原始地图上的坐标
            robot_cell_original = (robot_cell_downsampled[0] * 2, robot_cell_downsampled[1] * 2)

            plt.plot(robot_cell_original[0], robot_cell_original[1], c + 'o', markersize=10, zorder=5)

        plt.subplot(2, 3, 4)
        plt.axis('off')
        plt.title('Ground truth', y=-0.12, fontweight='bold')
        plt.imshow(self.env.ground_truth, cmap='gray_r')
        # plt.imshow(self.env.ground_truth == 255, cmap=plt.cm.colors.ListedColormap(['none', 'gray']), alpha=0.8)
        # ground_truth_nodes = get_cell_position_from_coords(self.env.ground_truth_coords,
        #                                                    self.env.ground_truth_info)

        # ground_truth_down = get_cell_position_from_coords(self.env.ground_truth_coords,
        #                                                    self.env.ground_truth_info)
        # ground_truth_nodes = (ground_truth_down[0] * 2, ground_truth_down[1] * 2)
        # plt.scatter(ground_truth_nodes[:, 0], ground_truth_nodes[:, 1], c='g', s=5)
        ground_truth_down = get_cell_position_from_coords(self.env.robot_locations,
                                                          self.env.ground_truth_info)
        ground_truth_nodes = np.array([(x * 2, y * 2) for x, y in ground_truth_down])

        plt.scatter(ground_truth_nodes[:, 0], ground_truth_nodes[:, 1], c='g', s=5)
        plt.subplot(2, 3, 2)
        plt.axis('off')
        plt.title('Flight belief', y=-0.12, fontweight='bold')
        robot_belief_flight = self.env.flight_info.map.copy()
        # robot_belief_flight = self.env.robot_belief_flight.copy()

        # robot_belief_flight[
        #     (robot_belief_flight < cell_height - FLT_SAFETY_BUFFER * hgt_scaling) & (robot_belief_flight > 0)
        #     ] = 0
        plt.imshow(robot_belief_flight, cmap='gray_r')
        plt.imshow(robot_belief_flight == 127, cmap=plt.cm.colors.ListedColormap(['none', 'lightblue']), alpha=0.8)
        plt.imshow(robot_belief_flight == 255, cmap=plt.cm.colors.ListedColormap(['none', 'gray']), alpha=0.8)

        # Plot flight frontiers
        flight_frontiers = get_cell_position_from_coords(np.array(list(self.env.global_frontiers)),
                                                         # flight_frontiers = get_cell_position_from_coords(np.array(list(self.env.global_frontiers)),
                                                         self.env.flight_info).reshape(-1, 2)
        plt.scatter(flight_frontiers[:, 0], flight_frontiers[:, 1], c='r', s=1)
        for robot in self.robot_list:
            c = color_list[robot.id % 7]
            # nodes = get_cell_position_from_coords(robot.node_coords, robot.global_safe_info)
            nodes = get_cell_position_from_coords(robot.node_coords, robot.global_safe_info)
            if nodes.ndim == 2:
                plt.scatter(nodes[:, 0], nodes[:, 1], c=robot.utility_safe, zorder=2, s=5)
            robot_cell = get_cell_position_from_coords(robot.location, robot.global_safe_info)
            # robot_cell = get_cell_position_from_coords(robot.location, robot.global_flight_info)
            plt.plot(robot_cell[0], robot_cell[1], c + 'o', markersize=10, zorder=5)
            plt.plot((np.array(robot.trajectory_x) - robot.global_safe_info.map_origin_x) / robot.cell_size,
                     # plt.plot((np.array(robot.trajectory_x) - robot.global_flight_info.map_origin_x) / robot.cell_size,
                     (np.array(robot.trajectory_y) - robot.global_safe_info.map_origin_y) / robot.cell_size, c,
                     # (np.array(robot.trajectory_y) - robot.global_flight_info.map_origin_y) / robot.cell_size, c,
                     linewidth=2, zorder=1)

        plt.subplot(2, 3, 3)
        plt.axis('off')

        plt.title('Ground belief', y=-0.12, fontweight='bold')
        # robot_belief = self.env.robot_belief_ground.copy()
        robot_belief = self.env.belief_ground.copy()
        plt.imshow(robot_belief, cmap='gray_r')
        plt.imshow(robot_belief == 127, cmap=plt.cm.colors.ListedColormap(['none', 'lightblue']), alpha=0.8)
        plt.imshow(robot_belief == 255, cmap=plt.cm.colors.ListedColormap(['none', 'gray']), alpha=0.8)

        # Plot ground frontiers

        filtered_frontiers = [item for item in self.env.global_frontiers_ground if len(item) == 3]
        frontiers_ground = get_cell_position_from_coords(
            np.array(filtered_frontiers).reshape(-1, 3)[:, :2], self.env.ground_truth_info).reshape(-1, 2)

        # frontiers_ground = get_cell_position_from_coords(
        #     np.array(list(self.env.explore_frontiers)).reshape(-1, 3)[:, :2], self.env.ground_truth_info).reshape(-1, 2)
        # np.array(list(self.env.global_frontiers_ground)).reshape(-1, 3)[:, :2], self.env.ground_info).reshape(-1, 2)
        plt.scatter(frontiers_ground[:, 0], frontiers_ground[:, 1], c='b', s=2, zorder=6)
        # Plot occluded areas
        row_indices, col_indices = np.where(self.env.ground_truth_info.map == -1)
        # row_indices, col_indices = np.where(self.env.ground_info.map == -1)
        if len(row_indices) > 0:
            plt.scatter(col_indices, row_indices, c='tab:orange', s=1, zorder=2)

        # Plot robot location
        for robot in self.robot_list:

            nodes = get_cell_position_from_coords(robot.node_coords, robot.global_safe_info)
            # nodes = get_cell_position_from_coords(robot.node_coords, robot.global_flight_info)
            if nodes.ndim == 2:
                plt.scatter(nodes[:, 0], nodes[:, 1], c=robot.occupancy, zorder=2, s=5)
            # plt.scatter(nodes[:, 0], nodes[:, 1], c=robot.occluded_indicator, zorder=2, s=5)
            c = color_list[robot.id % 7]
            robot_cell = get_cell_position_from_coords(robot.location, robot.global_safe_info)
            # robot_cell = get_cell_position_from_coords(robot.location, robot.global_flight_info)
            plt.plot(robot_cell[0], robot_cell[1], c + 'o', markersize=10, zorder=5)

        plt.subplot(2, 3, 6)
        plt.axis('off')
        plt.title('Safe zone', y=-0.12, fontweight='bold')
        plt.imshow(robot_belief, cmap='gray_r')
        plt.imshow(robot_belief == 127, cmap=plt.cm.colors.ListedColormap(['none', 'lightblue']), alpha=0.8)
        plt.imshow(robot_belief == 255, cmap=plt.cm.colors.ListedColormap(['none', 'gray']), alpha=0.8)

        # Plot safe frontiers
        if len(self.env.covered_safe_frontiers) != 0:
            print('covered safe frontiers', len(self.env.covered_safe_frontiers))
            # covered_safe_frontier_cells = (get_cell_position_from_coords(
            #     np.array(list(self.env.covered_safe_frontiers)).reshape(-1, 3)[:, :2], self.env.safe_info)
            #                                .reshape(-1,2))
            # covered_safe_frontier_cells = get_cell_position_from_coords(np.array(list(self.env.covered_safe_frontiers))
            #                                 .reshape(-1,3)[:,:2], self.env.safe_info).reshape(-1, 2)
            data = np.array(list(self.env.covered_safe_frontiers))
            covered_safe_frontier_cells = get_cell_position_from_coords(data, self.env.safe_info).reshape(-1, 2)

            plt.scatter(covered_safe_frontier_cells[:, 0], covered_safe_frontier_cells[:, 1], c='g', s=1, zorder=6)
        if len(self.env.uncovered_safe_frontiers) != 0:
            filtered_frontiers = [item for item in self.env.uncovered_safe_frontiers if len(item) == 3]
            uncovered_safe_frontier_cells = get_cell_position_from_coords(
                np.array(filtered_frontiers).reshape(-1, 3)[:, :2], self.env.safe_info).reshape(-1, 2)

            # uncovered_safe_frontier_cells = get_cell_position_from_coords(
            #     np.array(list(self.env.uncovered_safe_frontiers)).reshape(-1, 3)[:, :2], self.env.safe_info).reshape(-1,
            #                                                                                                          2)
            plt.scatter(uncovered_safe_frontier_cells[:, 0], uncovered_safe_frontier_cells[:, 1], c='r', s=1, zorder=6)

        # Plot robot location and nodes
        for robot in self.robot_list:
            c = color_list[robot.id % 7]
            if robot.id == 0:
                nodes = get_cell_position_from_coords(robot.node_coords, robot.global_safe_info)
                # nodes = get_cell_position_from_coords(robot.node_coords, robot.global_flight_info)
                if nodes.ndim == 2:
                    plt.scatter(nodes[:, 0], nodes[:, 1], c=robot.utility_safe, zorder=2, s=5)
                # plt.scatter(nodes[:, 0], nodes[:, 1], c=robot.utility_safe, zorder=2, s=5)
                alpha_mask = robot.global_safe_info.map / 255 / 3
                # alpha_mask = robot.global_safe_info.map / 255 / 3
                plt.imshow(robot.global_safe_info.map, cmap='Greens', alpha=alpha_mask)
                # plt.imshow(robot.global_safe_info.map, cmap='Greens', alpha=alpha_mask)

            robot_cell = get_cell_position_from_coords(robot.location, robot.global_safe_info)
            # robot_cell = get_cell_position_from_coords(robot.location, robot.global_flight_info)
            plt.plot(robot_cell[0], robot_cell[1], c + 'o', markersize=10, zorder=8)

        plt.subplot(2, 3, 5)
        plt.axis('off')
        plt.title('Nodes (Uncovered safe Utility)', y=-0.12, fontweight='bold')
        plt.imshow(robot_belief_flight, cmap='gray_r')
        plt.imshow(robot_belief_flight == 127, cmap=plt.cm.colors.ListedColormap(['none', 'lightblue']), alpha=0.8)
        plt.imshow(robot_belief_flight == 255, cmap=plt.cm.colors.ListedColormap(['none', 'gray']), alpha=0.8)
        if nodes.ndim == 2:
            plt.scatter(nodes[:, 0], nodes[:, 1], c=self.robot_list[0].utility_safe, zorder=2, s=5)
        # plt.scatter(nodes[:, 0], nodes[:, 1], c=self.robot_list[0].safe_indicator, zorder=2, s=5)
        for i in range(len(self.node_manager.x)):
            plt.plot((self.node_manager.x[i] - self.env.ground_truth_info.map_origin_x) / robot.cell_size,
                     # plt.plot((self.node_manager.x[i] - self.env.flight_info.map_origin_x) / robot.cell_size,
                     (self.node_manager.y[i] - self.env.ground_truth_info.map_origin_y) / robot.cell_size, 'cyan',
                     # (self.node_manager.y[i] - self.env.flight_info.map_origin_y) / robot.cell_size, 'cyan',
                     linewidth=0.5, zorder=1)

        plt.suptitle(
            'Explored ratio: {:.4g}   |    Explored ratio ground: {:4g}    \n    Safe ratio: {:4g}    |    Travel distance: {:.4g}'.format(
                self.env.explored_rate,
                self.env.explored_rate,
                # self.env.explored_rate_ground,
                self.env.safe_rate,
                max([robot.travel_dist for robot in
                     self.robot_list])), fontweight='bold')
        plt.tight_layout()
        plt.savefig('{}/{}_{}_samples.png'.format(gifs_path, self.global_step, step), dpi=150)
        plt.close()
        frame = '{}/{}_{}_samples.png'.format(gifs_path, self.global_step, step)
        self.env.frame_files.append(frame)


@ray.remote(num_cpus=1, num_gpus=0)
class DurhamWorkerParallel:
    def __init__(self, meta_agent_id, save_image=False):
        self.meta_agent_id = meta_agent_id
        self.save_image = save_image

    def job(self, episode_number):
        print("starting episode {} on metaAgent {}".format(episode_number, self.meta_agent_id))
        worker = DurhamWorker(self.meta_agent_id, episode_number, self.save_image)
        metrics = worker.run_episode()
        return metrics, self.meta_agent_id


if __name__ == '__main__':
    curr_test = 0
    max_dist_history = []
    explored_rate_history = []
    safe_rate_history = []
    success_rate_history = []
    all_length_history = []
    all_safe_rate_history = []
    all_explored_rate_history = []

    if NUM_TEST == 1:
        worker = DurhamWorker(0, curr_test, SAVE_IMG)
        metrics = worker.run_episode()
        max_dist_history.append(metrics[0])
        explored_rate_history.append(metrics[1])
        safe_rate_history.append(metrics[2])
        success_rate_history.append(metrics[3])
        all_length_history.extend(metrics[4])
        all_safe_rate_history.extend(metrics[5])
        all_explored_rate_history.extend(metrics[6])
    else:
        ray.init()
        meta_agents = [DurhamWorkerParallel.remote(i, SAVE_IMG) for i in range(RAY_META_AGENT)]
        run_list = []
        for meta_agent in meta_agents:
            run_list.append(meta_agent.job.remote(curr_test))
            curr_test += 1

        try:
            while len(max_dist_history) < curr_test:
                done_id, run_list = ray.wait(run_list)
                done_runs = ray.get(done_id)

                for res in done_runs:
                    metrics, meta_id = res
                    max_dist_history.append(metrics[0])
                    explored_rate_history.append(metrics[1])
                    safe_rate_history.append(metrics[2])
                    if metrics[3] == 1:
                        success_rate_history.append(metrics[0])
                    all_length_history.extend(metrics[4])
                    all_safe_rate_history.extend(metrics[5])
                    all_explored_rate_history.extend(metrics[6])

                    if curr_test < NUM_TEST:
                        run_list.append(meta_agents[meta_id].job.remote(curr_test))
                        curr_test += 1

        except KeyboardInterrupt:
            print("CTRL_C pressed. Killing remote workers")
            for a in meta_agents:
                ray.kill(a)

    print('=====================================')
    print('|#Test:', FOLDER_NAME)
    print('|#Test:', TEST_N_AGENTS, " AGENTS")
    print('|#Total test:', NUM_TEST)
    print('|#Average max length:', np.array(max_dist_history).mean())
    print('|#Average success length:', np.array(success_rate_history).mean())
    print('|#Std max length:', np.array(max_dist_history).std())
    print('|#Average explored rate:', np.array(explored_rate_history).mean())
    print('|#Average safe rate:', np.array(safe_rate_history).mean())
    print('|#Total success:', len(success_rate_history))

    if SAVE_CSV:
        idx = np.array(all_length_history).argsort()
        all_length_history = np.array(all_length_history)[idx]
        all_safe_rate_history = np.array(all_safe_rate_history)[idx]
        all_explored_rate_history = np.array(all_explored_rate_history)[idx]
        with open(f'results/result_durham_n={TEST_N_AGENTS}.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['length', 'safe', 'explore'])
            csv_data = np.concatenate([all_length_history.reshape(-1, 1), all_safe_rate_history.reshape(-1, 1),
                                       all_explored_rate_history.reshape(-1, 1)], axis=-1)
            writer.writerows(csv_data)
        print('CSV saved')
