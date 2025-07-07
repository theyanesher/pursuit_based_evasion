import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops, find_contours, approximate_polygon
from skimage import io
from shapely.geometry import Polygon  # make sure Shapely is installed
import os
import csv

def main():
    # ----------------------------------------------------
    # 1. Load and process all images in the directory
        directory = os.path.expanduser('~/Desktop/evasive_target/new_map_oct/')
    # for file_name in os.listdir(directory):
    #     if not file_name.endswith('.png'):
    #         continue
        file_name = 'map_8864.png'    
        ground_truth = (io.imread(os.path.join(directory, file_name), as_gray=True) * 255).astype(int)

        building_percentile = 60 # Above 60% of the building height is considered as the robot height

        # ----------------------------------------------------
        # 2. Crop out the white border
        #    The region inside is 4m x 4m
        # ----------------------------------------------------
        non_white = np.where(ground_truth != 255)

        min_row, max_row = np.min(non_white[0]), np.max(non_white[0])
        min_col, max_col = np.min(non_white[1]), np.max(non_white[1])
        cropped_img = ground_truth[min_row:max_row+1, min_col:max_col+1]

        cropped_img[(cropped_img > 0) & (cropped_img <= 30)] = 0

        cropped_img_filtered = cropped_img[(cropped_img < 255) & (cropped_img > 0)] 
        max_height = np.max(cropped_img_filtered)
        height = np.round(np.percentile(cropped_img_filtered, building_percentile)) 
        height_per_cell = 0.5 / height

        # Dimensions of the cropped region (in pixels)
        h, w = cropped_img.shape
        # The problem states the cropped region is 4m x 4m
        meter_per_pixel = 4.0 / float(w)  # assuming square pixels

        # Create a copy of the cropped array for "denoising"
        labeled_image = label(cropped_img, connectivity=2, background=0)

        # Define area threshold (e.g., remove components smaller than 5 pixels)
        area_threshold = 15

        # Create a mask to remove small regions
        filtered_image = labeled_image.copy()

        # Loop through each region and remove small ones
        for region in regionprops(labeled_image):
            if region.area < area_threshold:
                # Set all pixels in small regions to 0
                filtered_image[labeled_image == region.label] = 0

        # Create a copy of the cropped array for "denoising"
        filtered_labeled_image = label(filtered_image, connectivity=2, background=0)

        plt.figure()
        plt.imshow(cropped_img, cmap='gray')
        h, w = cropped_img.shape
        plt.axhline(y=h/2, color='r', linestyle='--')
        plt.axvline(x=w/2, color='r', linestyle='--')
        for region in regionprops(filtered_labeled_image):
            if region.area < 10:
                continue
            
            # region.centroid is (row, col)
            cy, cx = region.centroid
            region_mask = (filtered_labeled_image == region.label)

            # Find the largest contour (outer boundary)
            contours = find_contours(region_mask, level=0.5)
            if not contours:
                continue
            contour = max(contours, key=lambda c: c.shape[0])

            poly = approximate_polygon(contour, tolerance=2.5)

            # If it's a 4-sided shape, make sure it forms a proper rectangle
            if len(poly) == 5:  # (5 because last point repeats)
                # Convert to a Shapely polygon (drop the duplicate last point)
                candidate_poly = Polygon(contour[:-1])
                # Compute the minimum-area bounding rectangle
                min_rect = candidate_poly.minimum_rotated_rectangle
                # Extract the rectangle coordinates (this returns a closed polygon: first == last)
                poly = np.array(min_rect.exterior.coords)

            if len(poly) == 4:
                poly  = contour

            # Measure side lengths (pixel distance -> meters)
            side_lengths_m = []
            for i in range(len(poly)):
                r1, c1 = poly[i]
                r2, c2 = poly[(i+1) % len(poly)]  # wrap to first point
                dist_px = np.sqrt((r2 - r1)**2 + (c2 - c1)**2)
                dist_m = dist_px * meter_per_pixel
                side_lengths_m.append(dist_m)

            # Print building info
            # Prepare strings for vertices and side lengths
            vertices_str = '; '.join([f"({rr:.1f},{cc:.1f})" for rr, cc in poly])
            sides_str = '; '.join([f"{sl:.2f}" for sl in side_lengths_m])
            building_height = height_per_cell * cropped_img[int(cy), int(cx)]

            # Print building info
            print(f"Building {region.label}:")
            print(f"  Polygon vertices (row,col):")
            for (rr, cc) in poly:
                print(f"    ({rr:.1f}, {cc:.1f})")
            print(f"  Side lengths (m): {[f'{sl:.2f}' for sl in side_lengths_m]}")
            
            # Write the data into a CSV file
            csv_path = os.path.join(os.path.expanduser('~/Desktop/evasive_target'), 'building_data.csv')
            # Open in append mode; consider writing headers before the loop if needed
            with open(csv_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([region.label, sides_str, building_height])
            
            # Display the contour
            color = 'r' if building_height >= 0.5 else 'b'
            plt.text(cx, cy, str(region.label), color=color, fontsize=12, 
                    ha='center', va='center')  # Center the text on the centroid
            # Draw the rectangle
            plt.plot([poly[i][1] for i in range(len(poly))] + [poly[0][1]], 
                    [poly[i][0] for i in range(len(poly))] + [poly[0][0]], color=color, linewidth=1)    
            plt.title("Region Labels at Centroids")
            plt.legend(['Buildings > 0.5m', 'Buildings < 0.5m'], ['red', 'blue'], loc='upper right')   
        # plt.show() 
        plt.savefig(os.path.join(os.path.expanduser('~/Desktop/evasive_target'), file_name), dpi=150)
        plt.close()
    #     # Optionally skip very small regions if needed (e.g., area < 30 px)
    #     # if region.area < 30:
    #     #     continue

    #     # region.centroid is (row, col)
    #     cy, cx = region.centroid

    #     # Extract the mask for just this region
    #     region_mask = (labeled == region.label)

    #     # Find the largest contour (outer boundary)
    #     contours = find_contours(region_mask, level=0.5)
    #     if not contours:
    #         continue
    #     contour = max(contours, key=lambda c: c.shape[0])

    #     # Approximate polygon to simplify edges
    #     poly = approximate_polygon(contour, tolerance=2.0)

    #     # Measure side lengths (pixel distance -> meters)
    #     side_lengths_m = []
    #     for i in range(len(poly)):
    #         r1, c1 = poly[i]
    #         r2, c2 = poly[(i+1) % len(poly)]  # wrap to first point
    #         dist_px = np.sqrt((r2 - r1)**2 + (c2 - c1)**2)
    #         dist_m = dist_px * meter_per_pixel
    #         side_lengths_m.append(dist_m)

    #     # Print building info
    #     print(f"Building {building_id}:")
    #     print(f"  Polygon vertices (row,col):")
    #     for (rr, cc) in poly:
    #         print(f"    ({rr:.1f}, {cc:.1f})")
    #     print(f"  Side lengths (m): {[f'{sl:.2f}' for sl in side_lengths_m]}")
    #     print()

    #     # Save label info for plotting
    #     labels_info.append((cx, cy, building_id))

    #     building_id += 1

    # # ----------------------------------------------------
    # # 6. Display the result with Matplotlib, labeling each building
    # # ----------------------------------------------------
    # plt.figure(figsize=(6, 6))
    # plt.imshow(denoised_arr, cmap='gray', origin='upper')
    # plt.axis('off')

    # # Add text labels at each building's centroid
    # for (x, y, bid) in labels_info:
    #     plt.text(
    #         x, y, str(bid),
    #         color='red', fontsize=12, ha='center', va='center',
    #         bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.2')
    #     )

    # plt.title("Labeled Buildings (Anti-Aliasing Removed)")
    # output_path = "map_labeled.png"
    # plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    # plt.show()
    # print(f"Labeled map saved to: {output_path}")

if __name__ == "__main__":
    main()
