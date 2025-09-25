import open3d as o3d
import numpy as np
import os

def load_ply_file(filename):
    """Load a PLY file from the same directory as the script."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"PLY file '{filename}' not found in current directory")
    
    point_cloud = o3d.io.read_point_cloud(filename)
    
    if len(point_cloud.points) == 0:
        raise ValueError(f"No points found in PLY file '{filename}'")
    
    print(f"Loaded point cloud with {len(point_cloud.points)} points")
    return point_cloud

def visualize_point_cloud(point_cloud, title="Point Cloud"):
    """Visualize the point cloud using Open3D's built-in visualizer."""
    print(f"Displaying {title}...")
    o3d.visualization.draw_geometries([point_cloud], 
                                     window_name=title,
                                     width=800, 
                                     height=600)

def rotate_point_cloud(point_cloud, angle_x, angle_y, angle_z):
    """
    Rotate point cloud by specified angles (in degrees) around X, Y, and Z axes.
    """
    # Convert degrees to radians
    rx = np.radians(angle_x)
    ry = np.radians(angle_y) 
    rz = np.radians(angle_z)
    
    # Create rotation matrices
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(rx), -np.sin(rx)],
                    [0, np.sin(rx), np.cos(rx)]])
    
    R_y = np.array([[np.cos(ry), 0, np.sin(ry)],
                    [0, 1, 0],
                    [-np.sin(ry), 0, np.cos(ry)]])
    
    R_z = np.array([[np.cos(rz), -np.sin(rz), 0],
                    [np.sin(rz), np.cos(rz), 0],
                    [0, 0, 1]])
    
    # Combined rotation matrix (order: Z * Y * X)
    R_combined = R_z @ R_y @ R_x
    
    # Apply rotation
    rotated_cloud = point_cloud.rotate(R_combined, center=(0, 0, 0))
    
    print(f"Rotated point cloud by X: {angle_x}°, Y: {angle_y}°, Z: {angle_z}°")
    return rotated_cloud

def slice_point_cloud(point_cloud, axis='x', threshold=0.0, keep_side='positive'):
    """
    Slice the point cloud along a specified axis.
    
    Parameters:
    - axis: 'x', 'y', or 'z' - the axis to slice along
    - threshold: the cutting plane position
    - keep_side: 'positive' or 'negative' - which side to keep
    """
    points = np.asarray(point_cloud.points)
    
    # Get the index of the axis to slice along
    axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis.lower()]
    
    # Create a mask to filter points based on the threshold
    if keep_side == 'positive':
        mask = points[:, axis_idx] >= threshold
    else:
        mask = points[:, axis_idx] <= threshold
    
    # Create new point cloud with filtered points
    sliced_cloud = o3d.geometry.PointCloud()
    sliced_cloud.points = o3d.utility.Vector3dVector(points[mask])
    
    # Copy colors if they exist
    if point_cloud.has_colors():
        colors = np.asarray(point_cloud.colors)
        sliced_cloud.colors = o3d.utility.Vector3dVector(colors[mask])
    
    # Copy normals if they exist
    if point_cloud.has_normals():
        normals = np.asarray(point_cloud.normals)
        sliced_cloud.normals = o3d.utility.Vector3dVector(normals[mask])
    
    print(f"Sliced point cloud along {axis.upper()}-axis at {threshold}")
    print(f"Kept {keep_side} side: {len(sliced_cloud.points)} points remaining")
    
    return sliced_cloud

def save_point_cloud(point_cloud, output_filename):
    """Save the processed point cloud to a new PLY file."""
    success = o3d.io.write_point_cloud(output_filename, point_cloud)
    if success:
        print(f"Saved processed point cloud to '{output_filename}'")
    else:
        print(f"Failed to save point cloud to '{output_filename}'")
    return success

def main():
    # Configuration
    PLY_FILENAME = "elmers_washable_no_run_school_glue.ply"  # Change this to correct PLY file name
    OUTPUT_FILENAME = "sliced_point_cloud.ply"
    
    # Rotation angles in degrees
    ANGLE_X = 0.0  # Rotation around X-axis (30 degrees)
    ANGLE_Y = 0.0  # Rotation around Y-axis (45 degrees)
    ANGLE_Z = 0.0  # Rotation around Z-axis (60 degrees)
    
    # Slicing parameters
    SLICE_AXIS = 'z'        # 'x', 'y', or 'z'
    SLICE_THRESHOLD = 0.0   # Cutting plane position (defines the position of the slicing plane in 3D space)
    KEEP_SIDE = 'positive'  # 'positive' or 'negative'
    
    try:
        # Step 1: Load the PLY file
        print("Step 1: Loading PLY file...")
        point_cloud = load_ply_file(PLY_FILENAME)

        # Visualize original point cloud
        visualize_point_cloud(point_cloud, "Original Point Cloud")
        
        # Step 2: Rotate the point cloud
        print("\nStep 2: Rotating point cloud...")
        rotated_cloud = rotate_point_cloud(point_cloud, ANGLE_X, ANGLE_Y, ANGLE_Z)
        
        # Visualize rotated point cloud
        visualize_point_cloud(rotated_cloud, "Rotated Point Cloud")
        
        # Step 3: Slice the point cloud
        print("\nStep 3: Slicing point cloud...")
        sliced_cloud = slice_point_cloud(rotated_cloud, SLICE_AXIS, SLICE_THRESHOLD, KEEP_SIDE)
        
        # Step 4: Visualize the final result
        print("\nStep 4: Visualizing final result...")
        visualize_point_cloud(sliced_cloud, "Final Processed Point Cloud")
        
        # Step 5: Save the processed point cloud
        print("\nStep 5: Saving processed point cloud...")
        save_point_cloud(sliced_cloud, OUTPUT_FILENAME)
        
        print("\nProcessing completed successfully!")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure your PLY file is in the same directory as this script.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
