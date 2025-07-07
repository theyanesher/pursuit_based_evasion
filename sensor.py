import numpy as np
from parameter import *
from utils import *

from skimage import measure
from shapely.geometry import Polygon, Point
from skimage.draw import polygon as sk_polygon
from skimage.draw import line as sk_line

# def collision_check(x0, y0, x1, y1, ground_truth, robot_belief, robot_height):
#     x0 = float(x0) 
#     y0 = float(y0)
#     x1 = float(x1)
#     y1 = float(y1)

#     # Calculate the differences and steps
#     dx = x1 - x0
#     dy = y1 - y0
#     steps = max(abs(dx), abs(dy))  # Number of steps is the maximum of dx or dy
#     x_inc = dx / steps  # Increment for x
#     y_inc = dy / steps  # Increment for y

#     x, y = x0, y0
#     collision_flag = 0
#     max_collision = 3

#     # Traverse the line
#     for _ in range(int(steps) + 1):
#         grid_x = int(round(x))
#         grid_y = int(round(y))

#         if 0 <= grid_x < ground_truth.shape[1] and 0 <= grid_y < ground_truth.shape[0]:
#             building_height = ground_truth.item(grid_y, grid_x)
       
#             if robot_height <= building_height:
#                 collision_flag += 1

#                 if collision_flag >= max_collision:
#                     break
            
#             # Update robot belief
#             robot_belief.itemset((grid_y, grid_x), building_height)

#         x += x_inc
#         y += y_inc

#     return robot_belief

def collision_check_3D_radial(x0, y0, flight_belief, ground_belief, safe_belief, robot_height, sensor_range):
    x0 = int(round(x0))
    y0 = int(round(y0))
    
    map_height, map_width = flight_belief.shape
    num_azimuth_bins = 360  # Discretize azimuth angles into bins
    horizon_map = np.full(num_azimuth_bins, -np.inf)  # Stores max elevation angle per direction
    visible_map = np.zeros((map_height, map_width), dtype=bool)  # Visibility map
    
    # Iterate through all points in the height map within sensor range
    for y in range(max(0, y0 - sensor_range), min(map_height, y0 + sensor_range + 1)):
        for x in range(max(0, x0 - sensor_range), min(map_width, x0 + sensor_range + 1)):
            if x == x0 and y == y0:
                continue  # Skip the viewpoint itself
            
            dx = x - x0
            dy = y - y0
            dist = np.sqrt(dx**2 + dy**2)  # 2D distance from viewpoint
            
            if dist == 0 or dist > sensor_range:
                continue
            
            # Compute elevation angle from the viewpoint
            height_at_xy = flight_belief[y, x]
            elevation_angle = np.arctan2(height_at_xy - robot_height, dist)
            
            # Compute azimuth direction (angle in polar coordinates)
            azimuth = np.arctan2(dy, dx)  # Radians
            azimuth_index = int(((azimuth + np.pi) / (2 * np.pi)) * num_azimuth_bins) % num_azimuth_bins # Convert to bin index
            
            # Check if this elevation angle is the highest in its azimuth direction
            if elevation_angle > horizon_map[azimuth_index]:
                horizon_map[azimuth_index] = elevation_angle  # Update max elevation seen in this direction
                visible_map[y, x] = True  # Mark as visible
            
            # Update ground and safety beliefs
            if visible_map[y, x]:
                ground_belief[y, x] = height_at_xy
                safe_belief[y, x] = 255
    
    return ground_belief, safe_belief


def collision_check(x0, y0, x1, y1, ground_truth, robot_belief, robot_height):
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
    max_collision = 1

    while 0 <= x < ground_truth.shape[1] and 0 <= y < ground_truth.shape[0]:
        building_height = ground_truth.item(y, x)
        if building_height >= robot_height and collision_flag < max_collision:
            collision_flag += 1
            if collision_flag >= max_collision:
                robot_belief[max(0, y - 1):min(ground_truth.shape[0], y + 2), max(0, x - 1):min(ground_truth.shape[1], x + 2)] = \
                ground_truth[max(0, y - 1):min(ground_truth.shape[0], y + 2), max(0, x - 1):min(ground_truth.shape[1], x + 2)]
                break

        if x == x1 and y == y1:
            # robot_belief[max(0, y - 1):min(ground_truth.shape[0], y + 2), max(0, x - 1):min(ground_truth.shape[1], x + 2)] = \
            #     ground_truth[max(0, y - 1):min(ground_truth.shape[0], y + 2), max(0, x - 1):min(ground_truth.shape[1], x + 2)]
            break

        robot_belief.itemset((y, x), building_height)

        if error > 0:
            x += x_inc
            error -= dy
        else:
            y += y_inc
            error += dx

def collision_check_3D(x0, y0, x1, y1, flight_belief, ground_belief, safe_belief, robot_height):
    x0 = x0.round()
    y0 = y0.round()
    x1 = x1.round()
    y1 = y1.round()
    map = flight_belief

    dx, dy = abs(x1 - x0), abs(y1 - y0)
    x, y = x0, y0
    error = dx - dy
    x_inc = 1 if x1 > x0 else -1
    y_inc = 1 if y1 > y0 else -1
    dx *= 2
    dy *= 2

    start_cell = np.array([x0,y0])
    end_cell = np.array([x1,y1])
    end_height = 0     # Assume check until the ground level

    total_dist = np.linalg.norm(start_cell-end_cell)     # Total distance between start and end point in cell

    collision_count = 0
    max_collision = 1

    while 0 <= x < map.shape[1] and 0 <= y < map.shape[0]:
        # Compute the current voxel height
        next_point = np.array([x,y])
        current_dist = np.linalg.norm(next_point-end_cell)

        if total_dist != 0:
            current_height = (current_dist/total_dist)*(robot_height - end_height) + end_height   # in cell
        else:
            current_height = robot_height

        k = map.item(int(y), int(x))

        if x == x1 and y == y1:
            break

        if k >= current_height:
            collision_count += 1
            if collision_count > max_collision:
                break

        if k == -127:
            break

        # Update robot belief
        ground_belief.itemset((y, x), k)
        
        if k == 0:
            safe_belief.itemset((y, x), 255)

        if error > 0:
            x += x_inc
            error -= dy
        else:
            y += y_inc
            error += dx

    return ground_belief, safe_belief
# def collision_check_3D(x0, y0, x1, y1, flight_belief, ground_belief, safe_belief, robot_height):

#     x0, y0, z0 = float(x0), float(y0), float(robot_height)
#     x1, y1, z1 = float(x1), float(y1), float(0)

#     # Calculate the differences and steps
#     dx, dy, dz = x1 - x0, y1 - y0, z1 - z0
#     steps = max(abs(dx), abs(dy), abs(dz))  # Number of steps is the maximum of dx or dy
#     x_inc = dx / steps  # Increment for x
#     y_inc = dy / steps  # Increment for y
#     z_inc = dz / steps  # Increment for z

#     x, y, z = x0, y0, z0
#     collision_flag = 0
#     max_collision = 3

#     # Traverse the line
#     for _ in range(int(steps) + 1):
#         grid_x = int(round(x))
#         grid_y = int(round(y))

#         if 0 <= grid_x < flight_belief.shape[1] and 0 <= grid_y < flight_belief.shape[0]:
#             terrain_height = flight_belief.item(grid_y, grid_x)

#             if z <= terrain_height:
#                 collision_flag += 1
#                 if collision_flag >= max_collision:
#                     break

#             if terrain_height == -127:
#                 break

#             # Update robot belief
#             ground_belief.itemset((grid_y, grid_x), terrain_height)
            
#             if terrain_height == 0:
#                 safe_belief.itemset((grid_y, grid_x), 255)

#         x += x_inc
#         y += y_inc
#         z += z_inc

#     return ground_belief, safe_belief
def coverage_3D(x0, y0, x1, y1, ground_belief, safe_belief, robot_height):
    # Round and convert to integers
    x0, y0 = int(round(x0)), int(round(y0))
    x1, y1 = int(round(x1)), int(round(y1))
    
    # Get map dimensions for bounds checking
    rows, cols = ground_belief.shape
    
    # Use skimage's efficient line generation
    rr, cc = sk_line(y0, x0, y1, x1)
    
    # Compute total distance for scaling current_height
    total_dist = np.hypot(x1 - x0, y1 - y0)
    collision_count = 0
    max_collision = 1

    # Precompute the distances and corresponding heights for all points along the line
    if total_dist != 0:
        distances = np.hypot(x1 - cc, y1 - rr)
        heights = (distances / total_dist) * robot_height
    else:
        heights = np.full_like(rr, robot_height, dtype=float)

    for i in range(len(rr)):
        # Skip out-of-bound coordinates
        if rr[i] < 0 or rr[i] >= rows or cc[i] < 0 or cc[i] >= cols:
            continue
            
        # Stop if we've reached the end point
        if cc[i] == x1 and rr[i] == y1:
            break
        
        # Compute the euclidean distance from the current point to the end point
        current_height = heights[i]
        k = ground_belief[rr[i], cc[i]]
        
        if k >= current_height:
            collision_count += 1
            if collision_count > max_collision:
                break
        
        if k == -127 or k == -1:
            break
        
        if k == 0:
            r_start = max(0, rr[i] - 1)
            r_end = min(rows, rr[i] + 2)
            c_start = max(0, cc[i] - 1)
            c_end = min(cols, cc[i] + 2)
            patch = ground_belief[r_start:r_end, c_start:c_end]
            safe_belief[r_start:r_end, c_start:c_end][patch == 0] = 255
            safe_belief[rr[i], cc[i]] = 255

    return safe_belief


def update_coverage(x0, y0, x1, y1, ground_belief, safe_zone):
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

    offsets = np.array([(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)])
    
    while 0 <= x < ground_belief.shape[1] and 0 <= y < ground_belief.shape[0]:
        k = ground_belief.item(y, x)

        if k != 1:
            nx = x + offsets[:, 0]
            ny = y + offsets[:, 1]

            valid_indices = (0 <= nx) & (nx < ground_belief.shape[1]) & (0 <= ny) & (ny < ground_belief.shape[0])
            nx = nx[valid_indices]
            ny = ny[valid_indices]

            valid_positions = ground_belief[ny, nx] == 1
            safe_zone[ny[valid_positions], nx[valid_positions]] = 255
            break

        if x == x1 and y == y1:
            break
        
        safe_zone.itemset((y, x), 255)  
        # if k == 0:
        #     safe_zone.itemset((y, x), 255)            # Only account for ground level safe zone
        
        if error > 0:
            x += x_inc
            error -= dy
        else:
            y += y_inc
            error += dx

    return safe_zone

def safety_check(x0, y0, x1, y1, sub_safe_zone, sub_belief_ground, sub_intersection):
    x1 = x1.round()
    y1 = y1.round()
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    x, y = x0, y0
    error = dx - dy
    x_inc = 1 if x1 > x0 else -1
    y_inc = 1 if y1 > y0 else -1
    dx *= 2
    dy *= 2

    # plt.imshow(sub_safe_zone, cmap='gray')
    # plt.scatter(x0, y0, c='r')
    # plt.scatter(x1, y1, c='g')
    while 0 <= x < sub_safe_zone.shape[1] and 0 <= y < sub_safe_zone.shape[0]:
        # k1 = sub_safe_zone.item(y, x)
        ground_pixel = sub_belief_ground.item(y, x)   
        intersection_pixel = sub_intersection.item(y, x) if sub_intersection is not None else 0
        # print(ground_pixel)
        # print(intersection_pixel)
        # plt.scatter(x0, y0, c='r')

        if ground_pixel != 0: 
            break

        if intersection_pixel == 255:
            break

        if x == x1 and y == y1:
            break

        sub_safe_zone.itemset((y, x), 0)

        if error > 0:
            x += x_inc
            error -= dy
        else:
            y += y_inc
            error += dx

    # plt.show()

    return sub_safe_zone

def exploration_sensor(robot_position, sensor_range, robot_belief, ground_truth, robot_height):
    sensor_angle_inc = 0.5 / 180 * np.pi
    sensor_angle = 0
    x0 = robot_position[0]
    y0 = robot_position[1]
    while sensor_angle < 2 * np.pi:
        x1 = x0 + np.cos(sensor_angle) * sensor_range
        y1 = y0 + np.sin(sensor_angle) * sensor_range
        collision_check(x0, y0, x1, y1, ground_truth, robot_belief, robot_height)
        sensor_angle += sensor_angle_inc

def ground_sensor(robot_position, sensor_range, flight_belief, ground_belief, safe_belief, robot_height):
    sensor_angle_inc = 0.5 / 180 * np.pi
    sensor_angle = 0
    x0 = robot_position[0]
    y0 = robot_position[1]
    while sensor_angle < 2 * np.pi:
        x1 = x0 + np.cos(sensor_angle) * sensor_range
        y1 = y0 + np.sin(sensor_angle) * sensor_range
        ground_belief, safe_belief = collision_check_3D(x0, y0, x1, y1, flight_belief, ground_belief, safe_belief, robot_height)
        sensor_angle += sensor_angle_inc

    return ground_belief, safe_belief

def ground_sensor_radial(robot_position, sensor_range, flight_belief, ground_belief, safe_belief, robot_height):
    ground_belief, safe_belief = collision_check_3D_radial(robot_position[0], robot_position[1], flight_belief, ground_belief, safe_belief, robot_height, sensor_range)
    return ground_belief, safe_belief

def coverage_sensor(robot_position, sensor_range, ground_belief, safe_belief, robot_height):
    sensor_angle_inc = 0.5 / 180 * np.pi
    sensor_angle = 0
    x0 = robot_position[0]
    y0 = robot_position[1]
    while sensor_angle < 2 * np.pi:
        x1 = x0 + np.cos(sensor_angle) * sensor_range
        y1 = y0 + np.sin(sensor_angle) * sensor_range
        safe_belief  = coverage_3D(x0, y0, x1, y1, ground_belief, safe_belief, robot_height)
        sensor_angle += sensor_angle_inc
    return safe_belief

def decrease_safety_by_frontier(center, safety_range, sub_safe_zone, sub_belief_ground, sub_intersection=None):
    angle_inc = 0.5 / 180 * np.pi
    angle = 0
    x0 = center[0]
    y0 = center[1]
    while angle < 2 * np.pi:
        x1 = x0 + np.cos(angle) * safety_range
        y1 = y0 + np.sin(angle) * safety_range
        sub_safe_zone[:, :] = safety_check(x0, y0, x1, y1, sub_safe_zone, sub_belief_ground, sub_intersection)
        angle += angle_inc
    return sub_safe_zone



#===========================================================================================================================================
from skimage import measure
from shapely.geometry import Polygon, Point
from skimage.draw import polygon as sk_polygon
from skimage.draw import line as sk_line
def extract_contours(flight_map, show_img=False):
        all_contours = []
        new_robot_belief = np.where(flight_map <= 0, 1, flight_map)

        # Calculate the histogram of pixel values, excluding <1 and 255 ground and border
        frequency, _ = np.histogram(new_robot_belief, bins=256, range=(0, 255))
        
        # Set value to zero to skip check
        frequency[1] = 0
        frequency[255] = 0
        
        # Get frequency that is more than 20 pixels
        sorted_frequency = (frequency > 20)
        
        # Get the pixel_values of non-zero frequencies
        non_zero_indexes = np.nonzero(sorted_frequency)[0]

        # Loop through all non-zero frequency pixel value
        for pixel_value in non_zero_indexes:
            robot_belief = (new_robot_belief == pixel_value) * pixel_value
            contours = measure.find_contours(robot_belief)
            all_contours.extend([[contour, pixel_value] for contour in contours if len(contour) > 10])

        # Display the image and plot all contours found
        if show_img:
            plt.subplot(121)
            plt.imshow(flight_map, cmap=plt.cm.gray)

            for contour in all_contours:
                plt.plot(contour[0][:, 1], contour[0][:, 0], linewidth=2)
                
            plt.subplot(122)
            plt.imshow(new_robot_belief, cmap=plt.cm.gray)
            plt.axis('image')
            plt.show()
        
        return all_contours

def check_nearest_contour(contours, robot_cell, sensor_range_cell, robot_height_cell):
    valid_contours = []
    for contour in contours:
        polygon_vertices = contour[0][:, ::-1]  # Need to flip if using sklearn
        polygon = Polygon(polygon_vertices)
        centroid = np.array([polygon.centroid.x, polygon.centroid.y])
        distance = np.linalg.norm(robot_cell - centroid)
        if distance <= sensor_range_cell and robot_height_cell > contour[1]:
            valid_contours.append([polygon_vertices, contour])
    return valid_contours

def reorder_polygon_points(polygon, reference_point):
        # Calculate the distances from the reference point to all polygon points
        distances = np.linalg.norm(polygon - reference_point, axis=1)
        
        # Find the index of the closest point
        closest_point_index = np.argmin(distances)
        
        # Reorder the points to start with the closest point
        reordered_polygon = np.roll(polygon, -closest_point_index, axis=0)

        return reordered_polygon   

def check_collision_occlusion(point, robot_location, building_height, flight_map):
     # Only for ground info map
    # Bresenham line algorithm checking
    collision = False

    x0 = robot_location[0]
    y0 = robot_location[1]
    x1 = point[0]
    y1 = point[1]
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
    

    collision_count = 0
    max_collision = 2

    while 0 <= x < flight_map.shape[1] and 0 <= y < flight_map.shape[0]:

        k = flight_map.item(int(y), int(x))
            
        if x == x1 and y == y1:
            break

        if k == building_height:
            collision_count += 1
            if collision_count > max_collision:
                collision = True
                break

        if error > 0:
            x += x_inc
            error -= dy
        else:
            y += y_inc
            error += dx

    return collision
    
def identify_occlussion(flight_map, robot_cell, sensor_range_cell, robot_height_cell):
    contours = extract_contours(flight_map)
    valid_contours = check_nearest_contour(contours, robot_cell, sensor_range_cell, robot_height_cell)
    all_occluded_points = []
    occluded_points_list = []
    for contour in valid_contours:
        heights = np.full((contour[0].shape[0], 1), contour[1][1])
        vertices = np.hstack((contour[0], heights))
        robot_location = np.hstack((robot_cell, robot_height_cell)).astype(int)
        reordered_vertices = reorder_polygon_points(vertices, robot_location)
        current_polygon = Polygon(reordered_vertices) 
        robot_point = Point((robot_location[0], robot_location[1]))

        # Check if robot is above the building
        if not (flight_map[robot_location[1],robot_location[0]] == contour[1][1] and robot_point.within(current_polygon)):
            all_occluded_points = []
            for point in reordered_vertices:
                collision = check_collision_occlusion(point, robot_location, contour[1][1], flight_map)
                if collision:
                    all_occluded_points.append(point)

            if len(all_occluded_points) != 0:
                occluded_vertices = np.concatenate(all_occluded_points).reshape(-1,3)    # Might not calculate occluded point
                at_top_of_building = False
            else:
                occluded_vertices = np.array([])
        else:
            occluded_vertices = reordered_vertices
            at_top_of_building = True

        if occluded_vertices.shape[0] > 3:
            # Calculated the extent of the occluded point to the ground level
            direction_vectors = occluded_vertices - robot_location
            magnitudes = np.linalg.norm(direction_vectors, axis=1).reshape(-1,1)
            normalised_vectors = direction_vectors / magnitudes
            
            distances_to_ground = ((0 - occluded_vertices[0][2]) / normalised_vectors[:,2]).reshape(-1,1)
            pre_checked_projected_coordinates = occluded_vertices + distances_to_ground*normalised_vectors

            # Check if robot is above the building and if so all edges are occlueded
            if at_top_of_building == True:
                all_shadow_vertices = pre_checked_projected_coordinates[:,:2]
            else:
                all_shadow_vertices = np.concatenate((occluded_vertices[:,:2], pre_checked_projected_coordinates[::-1, :2]), axis=0)
            # Fit polygon to obtain occluded polygon and find the pixels occluded. Only vertices of shadow area in the belief size  
            polygon = Polygon(all_shadow_vertices)
            x_coords, y_coords = polygon.exterior.xy
            y_coords = np.rint(y_coords).astype(int)
            x_coords = np.rint(x_coords).astype(int)
            rr, cc = sk_polygon(
                    [int(round(y)) for y in y_coords],
                    [int(round(x)) for x in x_coords],
                    shape=flight_map.shape
                )
            occluded_area = np.column_stack((cc, rr))   # Did not concatenate outline as it may be outside the belief area

            # Extract only point that is in free space 
            valid_occluded_area_2D = occluded_area[(flight_map[rr, cc] == 0)] 
            heights = np.zeros((valid_occluded_area_2D.shape[0], 1))
            valid_occluded_area = np.hstack((valid_occluded_area_2D, heights)).reshape(-1,3)

            if valid_occluded_area.shape[0] == 0:
                continue
            else:
                occluded_points_list.append(valid_occluded_area) 

    if len(occluded_points_list) > 0:
        valid_occlusion = True
        polygons_occluded_pixels = np.concatenate(occluded_points_list, axis=0).reshape(-1,3)   
        polygons_occluded_pixels = np.unique(polygons_occluded_pixels.astype(int), axis=0, return_index=False)   # Ensure only unique occluded areas    
    else:
        valid_occlusion = False
        polygons_occluded_pixels = []

    return polygons_occluded_pixels, valid_occlusion            





