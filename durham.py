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
from env import Env
from agent import Agent
from ground_truth_node_manager import GroundTruthNodeManager
from utils import *
from node_manager import Node_manager
from sensor import coverage_sensor

TEST_N_AGENTS = 6
EXPLORATION = True
RAY_META_AGENT = 8

NUM_TEST = 1
MAX_EPISODE_STEP = 30
SENSOR_RANGE = 30
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
        self.env = Env(self.global_step, n_agents=TEST_N_AGENTS, plot=self.save_image, save_path=gifs_path)
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

        self.env.init_episode(self.node_manager)
        self.utility = None
        self.agent_status = [0] * self.env.n_agents  # 0: follower, 1: frontier guard
        self.agent_next_location = [None] * self.env.n_agents

    def assign_frontier_guard(self, best_locations):
        frontier_guards = []
        for location in best_locations:
            min_dist = 1e6
            frontier_guard = None
            for robot in self.robot_list:
                if robot.id in frontier_guards:
                    continue
                path_coords, dist = self.node_manager.a_star(robot.location, location)
                if dist < min_dist:
                    min_dist = dist
                    frontier_guard = robot.id
                    best_path = path_coords
            if frontier_guard is None:
                continue
            frontier_guards.append(frontier_guard)
            self.agent_status[frontier_guard] = 1
            self.agent_next_location[frontier_guard] = best_path[0]
        assert len(frontier_guards) > 0

    def assign_follower(self):
        for robot in self.robot_list:
            if self.agent_status[robot.id] == 0:  # follower
                min_dist = 1e6
                for status, guard_next_coords in zip(self.agent_status, self.agent_next_location):
                    if status == 1:  # frontier guard
                        path_coords, dist = self.node_manager.a_star(robot.location, guard_next_coords)
                        if dist < min_dist:
                            min_dist = dist
                            best_path = path_coords
                self.agent_next_location[robot.id] = best_path[0]

    def check_frontier_coverage(self, selected_locations):
        selected_locations = np.array(selected_locations).reshape(-1, 2)
        selected_frontier_uncoverage = np.zeros(selected_locations.shape[0])
        for frontier_loc in self.env.global_safe_zone_frontiers:
            nearby_selected_indices = np.argwhere(
                np.linalg.norm(selected_locations - frontier_loc, axis=1) < SENSOR_RANGE)
            nearby_selected_locations = selected_locations[nearby_selected_indices]
            uncovered = True
            for loc in nearby_selected_locations:
                if not check_collision(frontier_loc, loc, self.robot_height,
                                map_info=self.env.flight_info,
                                max_cell_height=self.env.max_height_cell):
                    uncovered = False
            if uncovered:
                nearest_index = np.argmin(np.linalg.norm(self.env.robot_locations - frontier_loc, axis=1))
                selected_frontier_uncoverage[nearest_index] += 1
        for i, loc in enumerate(selected_locations):
            if selected_frontier_uncoverage[i] > 1:
                selected_locations[i] = self.env.robot_locations[i]
        return selected_locations, selected_frontier_uncoverage

    def update_imaginary_safe_zone(self, robot_cell, imaginary_safe_zone):
        padded_robot_belief = deepcopy(self.env.belief_ground)
        padded_robot_belief[padded_robot_belief == 127] = 255
        new_imaginary_safe_zone = coverage_sensor(robot_cell, round(SENSOR_RANGE / CELL_SIZE),
                                                  deepcopy(imaginary_safe_zone), padded_robot_belief, robot_height=self.robot_height)
        safe_increase = np.sum(new_imaginary_safe_zone == 255) + np.sum(new_imaginary_safe_zone == 0) - np.sum(imaginary_safe_zone == 255)
        return new_imaginary_safe_zone, safe_increase

    def find_next_best_views(self):
        all_node_coords = self.robot_list[0].node_coords
        self.utility = self.robot_list[0].utility
        non_zero_utility_node_indices = np.argwhere(self.utility > 0)[:, 0].tolist()
        if len(non_zero_utility_node_indices) == 0:
            print("Not able to find next best view")
            return None
        assert len(non_zero_utility_node_indices) > 0
        candidate_node_coords = all_node_coords[non_zero_utility_node_indices]
        imaginary_safe_zone = self.env.safe_zone
        best_imaginary_safe_zone = self.env.safe_zone
        max_increase = 2
        best_locations = []
        filtered_indices = []
        best_coords = None

        while True:
            for i, coords in enumerate(candidate_node_coords):
                if i in filtered_indices:
                    continue
                candidate_flag = True
                for location in best_locations:
                    if location is None:
                        pass
                    dist_to_candidate = np.linalg.norm(location - coords)
                    if (dist_to_candidate < SENSOR_RANGE) and (
                            not check_collision(location, coords, self.robot_height,
                                                map_info=self.env.flight_info,
                                                max_cell_height=self.env.max_height_cell)):
                        candidate_flag = False
                        break
                if candidate_flag:
                    robot_cell = get_cell_position_from_coords(coords, self.env.ground_info)
                    new_imaginary_safe_zone, safe_increase = self.update_imaginary_safe_zone(robot_cell,
                                                                                             imaginary_safe_zone)
                    if safe_increase > max_increase:
                        max_increase = safe_increase
                        best_coords = coords
                        best_imaginary_safe_zone = new_imaginary_safe_zone
                else:
                    filtered_indices.append(i)

            if len(best_locations) > 0 and np.all(best_coords == best_locations[-1]):
                return best_locations

            imaginary_safe_zone = best_imaginary_safe_zone
            if best_coords is None:
                return None
            best_locations.append(best_coords)
            max_increase = 2

            if len(best_locations) == min(3, self.env.n_agents):
                break

        return best_locations
    
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
            self.agent_status = [0] * self.env.n_agents
            best_locations = self.find_next_best_views()
            if best_locations is None:
                continue
            self.assign_frontier_guard(best_locations)
            self.assign_follower()

            next_locations = []
            dist_list = []
            for robot in self.robot_list:
                next_location = self.agent_next_location[robot.id]
                next_locations.append(next_location)
                dist_list.append(np.linalg.norm(next_location - robot.location))

            selected_locations, _ = self.check_frontier_coverage(next_locations)

            if NUM_TEST == 1:
                frontier_guards = [robot.id for robot in self.robot_list if self.agent_status[robot.id] == 1]
                followers = [robot.id for robot in self.robot_list if self.agent_status[robot.id] == 0]
                print(f"Step {i}\tFrontier guards: {frontier_guards}\tFollowers: {followers}")

            if self.save_image:
                for robot in self.robot_list:
                    # FIXED: Use consistent coordinate system
                    nodes = get_cell_position_from_coords(robot.location, self.env.flight_info)
                    if nodes.ndim == 1:
                        continue
                # self.plot_local_env(i)

            # FIXED: Reduce safety decrease aggressiveness from 8 to 3 iterations
            tmp_safe_zone_frontier = deepcopy(self.env.global_safe_zone_frontiers)
            for _ in range(3):  # Reduced from 8 to 3
                self.env.decrease_safety(selected_locations)
                self.env.global_safe_zone_frontiers = get_safe_zone_frontier(self.env.safe_info, self.env.flight_info)
                if np.array_equal(tmp_safe_zone_frontier, self.env.global_safe_zone_frontiers):
                    break
                else:
                    tmp_safe_zone_frontier = deepcopy(self.env.global_safe_zone_frontiers)

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

            # if done:
            #     if self.save_image:
            #         self.plot_local_env(i + 1)
            #     break

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

        plt.subplot(2, 3, 1)
        plt.axis('off')
        plt.title('Actual map', y=-0.12, fontweight='bold')

        plt.imshow(np.where(self.env.colour_map == [255, 255, 255], [0, 0, 0], self.env.colour_map), alpha=1)
        for robot in self.robot_list:
            c = color_list[robot.id % 7]
            # FIXED: Use consistent coordinate system without arbitrary scaling
            robot_cell = get_cell_position_from_coords(robot.location, self.env.flight_info)
            plt.plot(robot_cell[0], robot_cell[1], c + 'o', markersize=10, zorder=5)

        plt.subplot(2, 3, 4)
        plt.axis('off')
        plt.title('Ground truth', y=-0.12, fontweight='bold')
        plt.imshow(self.env.ground_truth, cmap='gray_r')
        
        # FIXED: Use consistent coordinate system
        ground_truth_nodes = get_cell_position_from_coords(self.env.robot_locations, self.env.ground_truth_info)
        plt.scatter(ground_truth_nodes[:, 0], ground_truth_nodes[:, 1], c='g', s=5)
        
        plt.subplot(2, 3, 2)
        plt.axis('off')
        plt.title('Flight belief', y=-0.12, fontweight='bold')
        robot_belief_flight = self.env.flight_info.map.copy()

        plt.imshow(robot_belief_flight, cmap='gray_r')
        plt.imshow(robot_belief_flight == 127, cmap=plt.cm.colors.ListedColormap(['none', 'lightblue']), alpha=0.8)
        plt.imshow(robot_belief_flight == 255, cmap=plt.cm.colors.ListedColormap(['none', 'gray']), alpha=0.8)

        # Plot flight frontiers
        flight_frontiers = get_cell_position_from_coords(np.array(list(self.env.global_frontiers)),
                                                         self.env.flight_info).reshape(-1, 2)
        plt.scatter(flight_frontiers[:, 0], flight_frontiers[:, 1], c='r', s=1)
        for robot in self.robot_list:
            c = color_list[robot.id % 7]
            # FIXED: Use consistent coordinate system
            nodes = get_cell_position_from_coords(robot.node_coords, self.env.flight_info)
            if nodes.ndim == 2:
                plt.scatter(nodes[:, 0], nodes[:, 1], c=robot.utility_safe, zorder=2, s=5)
            robot_cell = get_cell_position_from_coords(robot.location, self.env.flight_info)
            plt.plot(robot_cell[0], robot_cell[1], c + 'o', markersize=10, zorder=5)
            # FIXED: Use consistent coordinate system
            plt.plot((np.array(robot.trajectory_x) - self.env.flight_info.map_origin_x) / robot.cell_size,
                     (np.array(robot.trajectory_y) - self.env.flight_info.map_origin_y) / robot.cell_size, c,
                     linewidth=2, zorder=1)

        plt.subplot(2, 3, 3)
        plt.axis('off')

        plt.title('Ground belief', y=-0.12, fontweight='bold')
        robot_belief = self.env.belief_flight.copy()
        plt.imshow(robot_belief, cmap='gray_r')
        plt.imshow(robot_belief == 127, cmap=plt.cm.colors.ListedColormap(['none', 'lightblue']), alpha=0.8)
        plt.imshow(robot_belief == 255, cmap=plt.cm.colors.ListedColormap(['none', 'gray']), alpha=0.8)

        # Plot ground frontiers
        filtered_frontiers = [item for item in self.env.global_frontiers_ground if len(item) == 3]
        frontiers_ground = get_cell_position_from_coords(
            np.array(filtered_frontiers).reshape(-1, 3)[:, :2], self.env.ground_truth_info).reshape(-1, 2)
        plt.scatter(frontiers_ground[:, 0], frontiers_ground[:, 1], c='b', s=2, zorder=6)
        
        # Plot occluded areas
        row_indices, col_indices = np.where(self.env.ground_truth_info.map == -1)
        if len(row_indices) > 0:
            plt.scatter(col_indices, row_indices, c='tab:orange', s=1, zorder=2)

        # Plot robot location
        for robot in self.robot_list:
            # FIXED: Use consistent coordinate system
            nodes = get_cell_position_from_coords(robot.node_coords, self.env.flight_info)
            if nodes.ndim == 2:
                plt.scatter(nodes[:, 0], nodes[:, 1], c=robot.occupancy, zorder=2, s=5)
            c = color_list[robot.id % 7]
            robot_cell = get_cell_position_from_coords(robot.location, self.env.flight_info)
            plt.plot(robot_cell[0], robot_cell[1], c + 'o', markersize=10, zorder=5)

        plt.subplot(2, 3, 6)
        plt.axis('off')
        plt.title('Safe zone', y=-0.12, fontweight='bold')
        plt.imshow(robot_belief, cmap='gray_r')
        plt.imshow(robot_belief == 127, cmap=plt.cm.colors.ListedColormap(['none', 'lightblue']), alpha=0.8)
        plt.imshow(robot_belief == 255, cmap=plt.cm.colors.ListedColormap(['none', 'gray']), alpha=0.8)

        # Plot safe frontiers
        if len(self.env.covered_safe_frontiers) != 0:
            data = np.array(list(self.env.covered_safe_frontiers))
            covered_safe_frontier_cells = get_cell_position_from_coords(data, self.env.safe_info).reshape(-1, 2)
            plt.scatter(covered_safe_frontier_cells[:, 0], covered_safe_frontier_cells[:, 1], c='g', s=1, zorder=6)
            
        if len(self.env.uncovered_safe_frontiers) != 0:
            filtered_frontiers = [item for item in self.env.uncovered_safe_frontiers if len(item) == 3]
            uncovered_safe_frontier_cells = get_cell_position_from_coords(
                np.array(filtered_frontiers).reshape(-1, 3)[:, :2], self.env.safe_info).reshape(-1, 2)
            plt.scatter(uncovered_safe_frontier_cells[:, 0], uncovered_safe_frontier_cells[:, 1], c='r', s=1, zorder=6)

        # Plot robot location and nodes
        for robot in self.robot_list:
            c = color_list[robot.id % 7]
            if robot.id == 0:
                # FIXED: Use consistent coordinate system
                nodes = get_cell_position_from_coords(robot.node_coords, self.env.flight_info)
                if nodes.ndim == 2:
                    plt.scatter(nodes[:, 0], nodes[:, 1], c=robot.utility_safe, zorder=2, s=5)
                alpha_mask = robot.global_safe_info.map / 255 / 3
                plt.imshow(robot.global_safe_info.map, cmap='Greens', alpha=alpha_mask)

            # FIXED: Use consistent coordinate system
            robot_cell = get_cell_position_from_coords(robot.location, self.env.flight_info)
            plt.plot(robot_cell[0], robot_cell[1], c + 'o', markersize=10, zorder=8)

        plt.subplot(2, 3, 5)
        plt.axis('off')
        plt.title('Nodes (Uncovered safe Utility)', y=-0.12, fontweight='bold')
        plt.imshow(robot_belief_flight, cmap='gray_r')
        plt.imshow(robot_belief_flight == 127, cmap=plt.cm.colors.ListedColormap(['none', 'lightblue']), alpha=0.8)
        plt.imshow(robot_belief_flight == 255, cmap=plt.cm.colors.ListedColormap(['none', 'gray']), alpha=0.8)
        if nodes.ndim == 2:
            plt.scatter(nodes[:, 0], nodes[:, 1], c=self.robot_list[0].utility_safe, zorder=2, s=5)
        
        # FIXED: Use consistent coordinate system
        for i in range(len(self.node_manager.x)):
            plt.plot((self.node_manager.x[i] - self.env.flight_info.map_origin_x) / robot.cell_size,
                     (self.node_manager.y[i] - self.env.flight_info.map_origin_y) / robot.cell_size, 'cyan',
                     linewidth=0.5, zorder=1)

        plt.suptitle(
            'Explored ratio: {:.4g}   |    Explored ratio ground: {:4g}    \n    Safe ratio: {:4g}    |    Travel distance: {:.4g}'.format(
                self.env.explored_rate,
                self.env.explored_rate,
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
