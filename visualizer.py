import open3d as o3d
import numpy as np
import matplotlib
import cv2
import matplotlib.pyplot as plt

def add_to_visualize_buffer(visualize_buffer, visualize_points, visualize_colors):
    assert visualize_points.shape[0] == visualize_colors.shape[0], f'got {visualize_points.shape[0]} for points and {visualize_colors.shape[0]} for colors'
    if len(visualize_points) == 0:
        return
    assert visualize_points.shape[1] == 3
    assert visualize_colors.shape[1] == 3
    # assert visualize_colors.max() <= 1.0 and visualize_colors.min() >= 0.0
    visualize_buffer["points"].append(visualize_points)
    visualize_buffer["colors"].append(visualize_colors)

def generate_nearby_points(point, num_points_per_side=5, half_range=0.005):
    if point.ndim == 1:
        offsets = np.linspace(-1, 1, num_points_per_side)
        offsets_meshgrid = np.meshgrid(offsets, offsets, offsets)
        offsets_array = np.stack(offsets_meshgrid, axis=-1).reshape(-1, 3)
        nearby_points = point + offsets_array * half_range
        return nearby_points.reshape(-1, 3)
    else:
        assert point.shape[1] == 3, "point must be (N, 3)"
        assert point.ndim == 2, "point must be (N, 3)"
        # vectorized version
        offsets = np.linspace(-1, 1, num_points_per_side)
        offsets_meshgrid = np.meshgrid(offsets, offsets, offsets)
        offsets_array = np.stack(offsets_meshgrid, axis=-1).reshape(-1, 3)
        nearby_points = point[:, None, :] + offsets_array
        return nearby_points

def img_save(image, output):
    image.save(output)
     
def img_show(image, title):
    plt.figure()
    plt.imshow(image)
    plt.axis("off")
    plt.title(title)
    plt.show(block=False)  
    plt.pause(0.2)
     
class Visualizer:
    def __init__(self, config, env):
        self.config = config
        self.env = env
        self.bounds_min = np.array(self.config['bounds_min'])
        self.bounds_max = np.array(self.config['bounds_max'])
        self.color = np.array([0.05, 0.55, 0.26])
        self.world2viewer = np.array([
            [0.3788, 0.3569, -0.8539, 0.0],
            [0.9198, -0.0429, 0.3901, 0.0],
            [-0.1026, 0.9332, 0.3445, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ]).T

    def show_img(self, rgb):
        cv2.imshow('img', rgb[..., ::-1])
        cv2.waitKey(0)
        print('showing image, click on the window and press "ESC" to close and continue')
        cv2.destroyAllWindows()
    
    def show_pointcloud(self, points, colors):
        # transform to viewer frame
        # points = np.dot(points, self.world2viewer[:3, :3].T) + self.world2viewer[:3, 3]
        # clip color to [0, 1]
        colors = np.clip(colors, 0.0, 1.0)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))  # float64 is a lot faster than float32 when added to o3d later
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
        print('visualizing pointcloud, click on the window and press "ESC" to close and continue')
        o3d.visualization.draw_geometries([pcd])

    def _get_scene_points_and_colors(self):
        """Retrieves 3D scene points and colors."""
        # scene
        cam_obs = [0]
        cam_obs = self.env.get_cam_obs()
        scene_points = []
        scene_colors = []
        for cam_id in range(len(cam_obs)):
            cam_points = cam_obs[cam_id]['points'].reshape(-1, 3)
            cam_colors = cam_obs[cam_id]['rgb'].reshape(-1, 3) / 255.0
            # clip to workspace
            # within_workspace_mask = filter_points_by_bounds(cam_points, self.bounds_min, self.bounds_max, strict=False)
            # cam_points = cam_points[within_workspace_mask]
            # cam_colors = cam_colors[within_workspace_mask]
            scene_points.append(cam_points)
            scene_colors.append(cam_colors)
        scene_points = np.concatenate(scene_points, axis=0)
        scene_colors = np.concatenate(scene_colors, axis=0)
        return scene_points, scene_colors

    def generate_shell_sphere_points(self, center, radius=0.003, num_points=600):
        phi = np.linspace(0, 2 * np.pi, int(np.sqrt(num_points)))
        theta = np.linspace(0, np.pi, int(np.sqrt(num_points)))
        phi, theta = np.meshgrid(phi, theta)

        x = radius * np.sin(theta) * np.cos(phi)
        y = radius * np.sin(theta) * np.sin(phi)
        z = radius * np.cos(theta)

        points = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=-1)
        return points + center

    def visualize_points(self, points):
        """Visualizes the given points within a 3D scene."""
        visualize_buffer = {
            "points": [],
            "colors": []
        }
        # scene
        scene_points, scene_colors = self._get_scene_points_and_colors()
        add_to_visualize_buffer(visualize_buffer, scene_points, scene_colors)
        # add points
        num_points = points.shape[0]
        color_map = matplotlib.colormaps["gist_rainbow"]
        points_colors = [color_map(i / num_points)[:3] for i in range(num_points)]
        for i in range(num_points):
            nearby_points = generate_nearby_points(points[i], num_points_per_side=5, half_range=0.005)
            nearby_colors = np.tile(points_colors[i], (nearby_points.shape[0], 1))
            nearby_colors = 0.5 * nearby_colors + 0.5 * np.array([1, 1, 1])
            add_to_visualize_buffer(visualize_buffer, nearby_points, nearby_colors)
        # visualize
        visualize_points = np.concatenate(visualize_buffer["points"], axis=0)
        visualize_colors = np.concatenate(visualize_buffer["colors"], axis=0)
        self.show_pointcloud(visualize_points, visualize_colors)
    