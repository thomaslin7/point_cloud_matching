import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
import copy

def load_point_clouds(complete_ply_path, partial_ply_path):
    """
    Load both point clouds from PLY files.
    
    Args:
        complete_ply_path (str): Path to complete 360-degree scan PLY file
        partial_ply_path (str): Path to partial ~180-degree scan PLY file
    
    Returns:
        tuple: (complete_pcd, partial_pcd) - Open3D point cloud objects
    """
    print("Loading point clouds...")
    
    # Load complete point cloud
    complete_pcd = o3d.io.read_point_cloud(complete_ply_path)
    if len(complete_pcd.points) == 0:
        raise ValueError(f"Failed to load complete point cloud from {complete_ply_path}")
    
    # Load partial point cloud
    partial_pcd = o3d.io.read_point_cloud(partial_ply_path)
    if len(partial_pcd.points) == 0:
        raise ValueError(f"Failed to load partial point cloud from {partial_ply_path}")
    
    print(f"Complete point cloud: {len(complete_pcd.points)} points")
    print(f"Partial point cloud: {len(partial_pcd.points)} points")
    
    return complete_pcd, partial_pcd

def pca_initial_alignment(complete_pcd, partial_pcd):
    """
    Perform PCA-based initial alignment to get a rough pose estimate.
    
    Args:
        complete_pcd: Complete point cloud
        partial_pcd: Partial point cloud
    
    Returns:
        numpy.ndarray: 4x4 transformation matrix
    """
    print("Performing PCA-based initial alignment...")
    
    # Get point arrays
    complete_points = np.asarray(complete_pcd.points)
    partial_points = np.asarray(partial_pcd.points)
    
    # Center both point clouds
    complete_centroid = np.mean(complete_points, axis=0)
    partial_centroid = np.mean(partial_points, axis=0)
    
    complete_centered = complete_points - complete_centroid
    partial_centered = partial_points - partial_centroid
    
    # Perform PCA on both point clouds
    pca_complete = PCA(n_components=3)
    pca_partial = PCA(n_components=3)
    
    pca_complete.fit(complete_centered)
    pca_partial.fit(partial_centered)
    
    # Get principal component directions
    complete_pc = pca_complete.components_
    partial_pc = pca_partial.components_
    
    # Ensure consistent orientation of principal components
    for i in range(3):
        if np.dot(complete_pc[i], partial_pc[i]) < 0:
            partial_pc[i] = -partial_pc[i]
    
    # Compute rotation matrix to align principal components
    R = np.dot(complete_pc.T, partial_pc)
    
    # Ensure proper rotation matrix
    U, _, Vt = np.linalg.svd(R)
    R = np.dot(U, Vt)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(U, Vt)
    
    # Compute translation
    t = complete_centroid - np.dot(R, partial_centroid)
    
    # Create transformation matrix
    initial_transform = np.eye(4)
    initial_transform[:3, :3] = R
    initial_transform[:3, 3] = t
    
    print("PCA-based initial alignment completed")
    print(f"Initial transformation matrix:\n{initial_transform}")
    
    return initial_transform

def apply_transformation(pcd, transform):
    """
    Apply transformation to point cloud.
    
    Args:
        pcd: Point cloud to transform
        transform: 4x4 transformation matrix
    
    Returns:
        Transformed point cloud
    """
    pcd_transformed = copy.deepcopy(pcd)
    pcd_transformed.transform(transform)
    return pcd_transformed

def icp_fine_alignment(complete_pcd, partial_pcd, initial_transform, threshold=0.02, max_iteration=2000):
    """
    Perform ICP for fine alignment after PCA initialization.
    
    Args:
        complete_pcd: Complete point cloud (target)
        partial_pcd: Partial point cloud (source)
        initial_transform: Initial transformation from PCA
        threshold (float): Distance threshold for ICP
        max_iteration (int): Maximum number of ICP iterations
    
    Returns:
        tuple: (icp_result, final_transform)
    """
    print("Performing ICP fine alignment...")
    
    # Apply initial transformation to partial point cloud
    partial_transformed = apply_transformation(partial_pcd, initial_transform)
    
    # Perform ICP registration
    icp_result = o3d.pipelines.registration.registration_icp(
        partial_transformed,
        complete_pcd,
        threshold,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration)
    )
    
    # Combine transformations
    final_transform = np.dot(icp_result.transformation, initial_transform)
    
    print(f"ICP fitness: {icp_result.fitness:.6f}")
    print(f"ICP inlier RMSE: {icp_result.inlier_rmse:.6f}")
    print(f"Final transformation matrix:\n{final_transform}")
    
    return icp_result, final_transform

def visualize_initial_alignment(complete_pcd, partial_pcd, initial_transform):
    """
    Visualize the initial PCA alignment.
    
    Args:
        complete_pcd: Complete point cloud
        partial_pcd: Partial point cloud
        initial_transform: Initial transformation matrix
    """
    print("Visualizing initial PCA alignment...")
    
    # Create copies for visualization
    complete_vis = copy.deepcopy(complete_pcd)
    partial_vis = copy.deepcopy(partial_pcd)
    partial_initial = apply_transformation(partial_vis, initial_transform)
    
    # Color the point clouds
    complete_vis.paint_uniform_color([1, 0, 0])  # Red for complete
    partial_initial.paint_uniform_color([0, 1, 0])  # Green for partial (initial)
    
    o3d.visualization.draw_geometries([complete_vis, partial_initial],
                                    window_name="Initial PCA Alignment (Red: Complete, Green: Partial)")

def visualize_final_alignment(complete_pcd, partial_pcd, final_transform):
    """
    Visualize the final ICP alignment.
    
    Args:
        complete_pcd: Complete point cloud
        partial_pcd: Partial point cloud
        final_transform: Final transformation matrix
    """
    print("Visualizing final ICP alignment...")
    
    # Create copies for visualization
    complete_vis = copy.deepcopy(complete_pcd)
    partial_vis = copy.deepcopy(partial_pcd)
    partial_final = apply_transformation(partial_vis, final_transform)
    
    # Color the point clouds
    complete_vis.paint_uniform_color([1, 0, 0])  # Red for complete
    partial_final.paint_uniform_color([0, 0, 1])  # Blue for partial (final)
    
    o3d.visualization.draw_geometries([complete_vis, partial_final],
                                    window_name="Final ICP Alignment (Red: Complete, Blue: Partial)")

def save_registered_point_cloud(partial_pcd, final_transform, output_path="registered_partial.ply"):
    """
    Save the registered partial point cloud.
    
    Args:
        partial_pcd: Original partial point cloud
        final_transform: Final transformation matrix
        output_path (str): Path to save the registered point cloud
    """
    if final_transform is not None:
        registered_partial = apply_transformation(partial_pcd, final_transform)
        o3d.io.write_point_cloud(output_path, registered_partial)
        print(f"Registered partial point cloud saved to {output_path}")
    else:
        print("No final transformation available. Run registration first.")

def evaluate_registration(complete_pcd, partial_pcd, final_transform):
    """
    Evaluate the quality of registration.
    
    Args:
        complete_pcd: Complete point cloud
        partial_pcd: Partial point cloud
        final_transform: Final transformation matrix
    
    Returns:
        numpy.ndarray: Array of distances
    """
    if final_transform is None:
        print("No final transformation available. Run registration first.")
        return None
    
    # Apply final transformation
    partial_registered = apply_transformation(partial_pcd, final_transform)
    
    # Compute distances between registered partial and complete point clouds
    distances = partial_registered.compute_point_cloud_distance(complete_pcd)
    distances = np.asarray(distances)
    
    print(f"Registration evaluation:")
    print(f"Mean distance: {np.mean(distances):.6f}")
    print(f"Median distance: {np.median(distances):.6f}")
    print(f"Max distance: {np.max(distances):.6f}")
    print(f"Std distance: {np.std(distances):.6f}")
    
    return distances

def estimate_normals(pcd):
    """
    Estimate normals for a point cloud.
    
    Args:
        pcd: Point cloud to estimate normals for
    
    Returns:
        Point cloud with estimated normals
    """
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    return pcd

def create_coordinate_axes(center, pca_components, scale=1.0):
    """Create coordinate axes based on PCA principal components"""
    axes_cylinders = []
    axes_colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # RGB for XYZ
    
    for i in range(3):
        # Create thick cylinder for each axis
        start_point = center
        end_point = center + pca_components[i] * scale
        
        # Calculate cylinder parameters
        direction = end_point - start_point
        length = np.linalg.norm(direction)
        direction_normalized = direction / length
        
        # Create cylinder
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(
            radius=length * 0.01,  # Thick radius (1% of length)
            height=length
        )
        
        # Orient cylinder along the axis direction
        z_axis = np.array([0, 0, 1])
        if not np.allclose(direction_normalized, z_axis):
            rotation_axis = np.cross(z_axis, direction_normalized)
            if np.linalg.norm(rotation_axis) > 1e-6:
                rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                angle = np.arccos(np.clip(np.dot(z_axis, direction_normalized), -1, 1))
                rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * angle)
                cylinder.rotate(rotation_matrix, center=(0, 0, 0))
        
        # Translate to correct position
        cylinder.translate(start_point + direction / 2)
        
        # Set color
        cylinder.paint_uniform_color(axes_colors[i])
        
        axes_cylinders.append(cylinder)
    
    return axes_cylinders

def create_bounding_box(points, color=[0, 1, 0]):  # Light green
    """Create a thick 3D bounding box around the points"""
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    bbox_size = max_coords - min_coords
    
    # Define the 8 corners of the bounding box
    corners = np.array([
        [min_coords[0], min_coords[1], min_coords[2]],  # 0
        [max_coords[0], min_coords[1], min_coords[2]],  # 1
        [max_coords[0], max_coords[1], min_coords[2]],  # 2
        [min_coords[0], max_coords[1], min_coords[2]],  # 3
        [min_coords[0], min_coords[1], max_coords[2]],  # 4
        [max_coords[0], min_coords[1], max_coords[2]],  # 5
        [max_coords[0], max_coords[1], max_coords[2]],  # 6
        [min_coords[0], max_coords[1], max_coords[2]]   # 7
    ])
    
    # Define the 12 edges of the bounding box
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
        [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
        [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges
    ]
    
    # Create thick cylinders for each edge
    edge_cylinders = []
    edge_radius = np.min(bbox_size) * 0.008  # Thicker edges (0.8% of smallest dimension)
    
    for edge in edges:
        start_point = corners[edge[0]]
        end_point = corners[edge[1]]
        
        # Calculate cylinder parameters
        direction = end_point - start_point
        length = np.linalg.norm(direction)
        direction_normalized = direction / length
        
        # Create cylinder
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(
            radius=edge_radius,
            height=length
        )
        
        # Orient cylinder along the edge direction
        z_axis = np.array([0, 0, 1])
        if not np.allclose(direction_normalized, z_axis):
            rotation_axis = np.cross(z_axis, direction_normalized)
            if np.linalg.norm(rotation_axis) > 1e-6:
                rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                angle = np.arccos(np.clip(np.dot(z_axis, direction_normalized), -1, 1))
                rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * angle)
                cylinder.rotate(rotation_matrix, center=(0, 0, 0))
        
        # Translate to correct position
        cylinder.translate(start_point + direction / 2)
        
        # Set color to light green
        cylinder.paint_uniform_color(color)
        
        edge_cylinders.append(cylinder)
    
    return edge_cylinders

def visualize_with_complete_axes_and_bbox(partial_pcd, complete_pcd, final_transform):
    """
    Visualize registered partial point cloud with complete point cloud's PCA axes and bounding box.
    
    Args:
        partial_pcd: Partial point cloud
        complete_pcd: Complete point cloud
        final_transform: Final transformation matrix
    """
    print("Visualizing registered partial point cloud with complete point cloud's axes and bounding box...")
    
    # Create copies for visualization
    partial_vis = copy.deepcopy(partial_pcd)
    complete_vis = copy.deepcopy(complete_pcd)
    
    # Apply transformation to partial point cloud
    partial_vis.transform(final_transform)
    
    # Get points as numpy arrays
    complete_points = np.asarray(complete_vis.points)
    
    # Calculate center and perform PCA on complete point cloud
    center = np.mean(complete_points, axis=0)
    pca = PCA(n_components=3)
    pca.fit(complete_points)
    principal_components = pca.components_
    
    # Calculate appropriate scale for axes
    bbox_size = np.max(complete_points, axis=0) - np.min(complete_points, axis=0)
    axes_scale = np.max(bbox_size) * 0.3
    
    # Create coordinate axes and bounding box from complete point cloud
    axes_cylinders = create_coordinate_axes(center, principal_components, axes_scale)
    bbox_cylinders = create_bounding_box(complete_points)
    
    # Color the point clouds
    partial_vis.paint_uniform_color([0, 0, 1])  # Blue for partial
    complete_vis.paint_uniform_color([1, 0, 0])  # Red for complete
    
    # Create visualization geometries
    geometries = [partial_vis, complete_vis] + axes_cylinders + bbox_cylinders
    
    # Visualize
    o3d.visualization.draw_geometries(
        geometries,
        window_name="Registered Partial Point Cloud with Complete Point Cloud's Axes and Bounding Box",
        width=1200,
        height=800,
        left=50,
        top=50
    )

def visualize_partial_with_complete_axes_and_bbox(partial_pcd, complete_pcd, final_transform):
    """
    Visualize only the registered partial point cloud with complete point cloud's PCA axes and bounding box.
    
    Args:
        partial_pcd: Partial point cloud
        complete_pcd: Complete point cloud (used only for axes and bounding box)
        final_transform: Final transformation matrix
    """
    print("Visualizing registered partial point cloud with complete point cloud's axes and bounding box...")
    
    # Create copies for visualization
    partial_vis = copy.deepcopy(partial_pcd)
    complete_vis = copy.deepcopy(complete_pcd)
    
    # Apply transformation to partial point cloud
    partial_vis.transform(final_transform)
    
    # Get points as numpy arrays
    complete_points = np.asarray(complete_vis.points)
    
    # Calculate center and perform PCA on complete point cloud
    center = np.mean(complete_points, axis=0)
    pca = PCA(n_components=3)
    pca.fit(complete_points)
    principal_components = pca.components_
    
    # Calculate appropriate scale for axes
    bbox_size = np.max(complete_points, axis=0) - np.min(complete_points, axis=0)
    axes_scale = np.max(bbox_size) * 0.3
    
    # Create coordinate axes and bounding box from complete point cloud
    axes_cylinders = create_coordinate_axes(center, principal_components, axes_scale)
    bbox_cylinders = create_bounding_box(complete_points)
    
    # Color the partial point cloud
    partial_vis.paint_uniform_color([0, 0, 1])  # Blue for partial
    
    # Create visualization geometries
    geometries = [partial_vis] + axes_cylinders + bbox_cylinders
    
    # Visualize
    o3d.visualization.draw_geometries(
        geometries,
        window_name="Registered Partial Point Cloud with Reference Axes and Bounding Box",
        width=1200,
        height=800,
        left=50,
        top=50
    )

def perform_point_cloud_registration(complete_ply_path, partial_ply_path, 
                                   icp_threshold=0.02, 
                                   max_iterations=2000, visualize=True):
    """
    Main function to perform complete point cloud registration workflow.
    
    Args:
        complete_ply_path (str): Path to complete PLY file
        partial_ply_path (str): Path to partial PLY file
        icp_threshold (float): Distance threshold for ICP
        max_iterations (int): Maximum ICP iterations
        visualize (bool): Whether to show visualizations
    
    Returns:
        tuple: (final_transform, icp_result, distances)
    """
    try:
        # Load point clouds
        complete_pcd, partial_pcd = load_point_clouds(complete_ply_path, partial_ply_path)
        
        # Estimate normals for both point clouds
        print("Estimating normals...")
        complete_pcd = estimate_normals(complete_pcd)
        partial_pcd = estimate_normals(partial_pcd)
        
        # Perform PCA-based initial alignment
        initial_transform = pca_initial_alignment(complete_pcd, partial_pcd)
        
        # Perform ICP fine alignment
        icp_result, final_transform = icp_fine_alignment(
            complete_pcd, partial_pcd, initial_transform,
            threshold=icp_threshold, max_iteration=max_iterations
        )
        
        # Evaluate registration quality
        distances = evaluate_registration(complete_pcd, partial_pcd, final_transform)
        
        # Visualize results if requested
        if visualize:
            visualize_initial_alignment(complete_pcd, partial_pcd, initial_transform)
            visualize_final_alignment(complete_pcd, partial_pcd, final_transform)
            # Add visualization with complete point cloud's axes and bounding box
            visualize_with_complete_axes_and_bbox(partial_pcd, complete_pcd, final_transform)
            # Add visualization with only partial point cloud and complete point cloud's axes and bounding box
            visualize_partial_with_complete_axes_and_bbox(partial_pcd, complete_pcd, final_transform)
        
        # Save registered point cloud
        save_registered_point_cloud(partial_pcd, final_transform, "registered_partial.ply")
        
        print("\nRegistration completed successfully!")
        
        return final_transform, icp_result, distances
        
    except Exception as e:
        print(f"Error during registration: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None

def main():
    """Main function to run the point cloud registration."""
    # File paths (modify these to match your PLY files)
    complete_ply = "elmers_washable_no_run_school_glue.ply"  # Complete 360-degree scan
    partial_ply = "sliced_point_cloud.ply"    # Partial ~180-degree scan
    
    # Check if files exist
    if not os.path.exists(complete_ply):
        print(f"Error: {complete_ply} not found in current directory")
        return
    if not os.path.exists(partial_ply):
        print(f"Error: {partial_ply} not found in current directory")
        return
    
    # Perform registration
    final_transform, icp_result, distances = perform_point_cloud_registration(
        complete_ply, 
        partial_ply,
        icp_threshold=0.02,     # Adjust as needed
        max_iterations=2000,    # Adjust as needed
        visualize=True          # Set to False to skip visualization
    )
    
    if final_transform is not None:
        print("\nFinal transformation matrix:")
        print(final_transform)

if __name__ == "__main__":
    main()
