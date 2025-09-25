import numpy as np
import open3d as o3d
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_point_cloud(file_path):
    """Load point cloud from PLY file using Open3D"""
    try:
        point_cloud = o3d.io.read_point_cloud(file_path)
        if len(point_cloud.points) == 0:
            raise ValueError("Empty point cloud loaded")
        print(f"Loaded point cloud with {len(point_cloud.points)} points")
        return point_cloud
    except Exception as e:
        print(f"Error loading point cloud: {e}")
        return None

def perform_pca(points):
    """Perform PCA on 3D points and return PCA object and transformed points"""
    pca = PCA(n_components=3)   # compute 3 principal components (eigenvectors for x, y, z axes)
    
    # Fits PCA to the points, computing the mean and the covariance matrix
    # then extracts the eigenvectors and eigenvalues of that covariance matrix 
    points_transformed = pca.fit_transform(points)
    # pca.fit_transform(points) changes the internals of pca so that pca.components_ becomes the principal directions of the point cloud
    
    print("PCA Analysis Results:")
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Principal components (eigenvectors):\n{pca.components_}")
    
    return pca, points_transformed

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
        # Default cylinder is along Z-axis, need to rotate to match direction
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
    
    # Define the 12 edges of the bounding box with their connections
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

def visualize_results(point_cloud, center, pca_components, points):
    """Visualize the point cloud with PCA axes and bounding box"""
    # Calculate appropriate scale for axes based on point cloud size
    bbox_size = np.max(points, axis=0) - np.min(points, axis=0)
    axes_scale = np.max(bbox_size) * 0.3
    
    # Create coordinate axes (3 thick cylinders to represent axes)
    axes_cylinders = create_coordinate_axes(center, pca_components, axes_scale)
    
    # Create bounding box (thick cylinders representing edges)
    bbox_cylinders = create_bounding_box(points)
    
    # Create a list of geometries to visualize
    geometries = [point_cloud] + axes_cylinders + bbox_cylinders
    
    # Set up visualization
    print("Opening 3D visualization...")
    print("Controls:")
    print("- Mouse: Rotate view")
    print("- Scroll: Zoom")
    print("- Ctrl+Mouse: Pan")
    print("- Press 'Q' or close window to exit")
    
    # Visualize
    o3d.visualization.draw_geometries(
        geometries,
        window_name="3D Point Cloud with PCA Axes and Bounding Box",
        width=1200,
        height=800,
        left=50,
        top=50
    )

def main():
    # File path to PLY file
    file_path = "/Users/thomaslin/Documents/visual_studio_code/AFA_Innovation_Assignments/Lecture5_track1/elmers_washable_no_run_school_glue.ply"
    
    # Load point cloud
    point_cloud = load_point_cloud(file_path)
    if point_cloud is None:
        return
    
    # Convert to numpy array for processing
    points = np.asarray(point_cloud.points)
    
    # Calculate center of the point cloud
    center = np.mean(points, axis=0)
    print(f"Point cloud center: {center}")
    
    # Perform PCA
    pca, points_transformed = perform_pca(points)
    
    # Get principal components (eigenvectors)
    principal_components = pca.components_
    
    # Print bounding box information
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    bbox_size = max_coords - min_coords
    print(f"Bounding box min: {min_coords}")
    print(f"Bounding box max: {max_coords}")
    print(f"Bounding box size: {bbox_size}")
    
    # Visualize results
    visualize_results(point_cloud, center, principal_components, points)

if __name__ == "__main__":
    main()
