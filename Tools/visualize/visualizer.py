import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from collections import defaultdict
import argparse
import numpy as np
import yaml
import os

parser = argparse.ArgumentParser()
parser.add_argument('--frame', default='0')
parser.add_argument('--folder', default='')
parser.add_argument('--dataset', default = "carla")
parser.add_argument('--voxel_grid', default = False)
parser.add_argument('--config_file', default = 'carla.yaml')
parser.add_argument('--label_map', default = False)

class SpheresApp:
    MENU_SCENE = 1
    MENU_BEFORE = 2
    MENU_QUIT = 3
    SAVE_SCENE = 4

    def __init__(self, opt):
        self._id = -1
        self.opt = opt
        self.file_list = self.get_file_list()
        
        self.window = gui.Application.instance.create_window("Pyramid ", 1500, 1000)
        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(self.window.renderer)
        self.scene.scene.set_background([255, 255, 255, 1])
        self.scene.scene.scene.set_sun_light(
            [-0.577, 0.577, -0.577],
            [2, 2, 2],
            60000)  
        
        self.scene.scene.scene.enable_sun_light(True)
        bbox = o3d.geometry.AxisAlignedBoundingBox([128, 128, -120], [128, 128, 120])

        self.scene.setup_camera(60, bbox, [0, 0, 1])

        self.window.add_child(self.scene)

        if gui.Application.instance.menubar is None:
            debug_menu = gui.Menu()
            debug_menu.add_item("Next Scene", SpheresApp.MENU_SCENE)
            debug_menu.add_separator()
            debug_menu.add_item("Before Scene", SpheresApp.MENU_BEFORE)
            debug_menu.add_separator()
            debug_menu.add_item("Save Scene", SpheresApp.SAVE_SCENE)
            debug_menu.add_separator()
            debug_menu.add_item("Quit", SpheresApp.MENU_QUIT)
            menu = gui.Menu()
            menu.add_menu("SSC", debug_menu)
            gui.Application.instance.menubar = menu

        self.window.set_on_menu_item_activated(SpheresApp.MENU_SCENE,self._on_menu_scene)
        self.window.set_on_menu_item_activated(SpheresApp.MENU_QUIT,self._on_menu_quit)
        self.window.set_on_menu_item_activated(SpheresApp.MENU_BEFORE,self._on_menu_before)
        self.window.set_on_menu_item_activated(SpheresApp.SAVE_SCENE,self._on_menu_transparent)
        self.window.set_on_key(self._on_key_event)

    def _on_key_event(self, event):
        if event.type == gui.KeyEvent.Type.DOWN:
            if event.key == 263:  # Left arrow
                self._on_menu_before()
            elif event.key == 264:  # Right arrow
                self._on_menu_scene()

    def _on_menu_before(self):
        self._id = (self._id - 1) % len(self.file_list)
        self.update_scene()

    def _on_menu_quit(self):
        gui.Application.instance.quit()

    def _on_menu_scene(self):
        self._id = (self._id + 1) % len(self.file_list)
        self.update_scene()

    def get_file_list(self):
        folder_path = './' + self.opt.folder
        file_list = [os.path.join(folder_path, filename) for filename in sorted(os.listdir(folder_path))]
        return file_list
    
    def update_scene(self):
        current_file = os.path.basename(self.file_list[self._id])
        self.window.title=f"Pyramid - {current_file}"

        points, colors = self.get_voxel(self.file_list[self._id])
        mat = rendering.MaterialRecord()
        mat.shader = "defaultLit"
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors / 255)
        self.scene.scene.clear_geometry()
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=1)
        self.scene.scene.add_geometry("scene_voxels" + str(self._id), voxel_grid, mat)

        if self.opt.voxel_grid:
            line_set = self.create_voxel_grid_lines(voxel_grid)
            self.scene.scene.add_geometry("voxel_lines", line_set, mat)

    
    def _on_menu_transparent(self): 
        points, colors = self.get_voxel(self.file_list[self._id])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        pcd.colors = o3d.utility.Vector3dVector(colors/255)

        red_area_x_min_1, red_area_x_max_1 = (0, 286)
        red_area_y_min_1, red_area_y_max_1 = (112, 174)

        red_area_x_min_2, red_area_x_max_2 = (112, 174)
        red_area_y_min_2, red_area_y_max_2 = (0, 286)

        red_area_z_range = (0, 16)

        np_points = np.asarray(pcd.points)
        np_colors = np.asarray(pcd.colors)

        alpha = 0.7
        for i, (x, y, z) in enumerate(np_points):
            if ((red_area_x_min_1 <= x <= red_area_x_max_1 and red_area_y_min_1 <= y <= red_area_y_max_1) or
                (red_area_x_min_2 <= x <= red_area_x_max_2 and red_area_y_min_2 <= y <= red_area_y_max_2)) and red_area_z_range[0] <= z <= red_area_z_range[1]:
                mixed_color = np.multiply(np_colors[i], alpha) + np.multiply([1, 1, 1], (1 - alpha))
                np_colors[i] = mixed_color

        pcd.colors = o3d.utility.Vector3dVector(np_colors)

        self.scene.scene.clear_geometry()

        mat = rendering.MaterialRecord()
        mat.shader = "defaultLit"

        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=1)
        self.scene.scene.add_geometry("scene_voxels" + str(self._id), voxel_grid, mat)

        if self.opt.voxel_grid:
            line_set = self.create_voxel_grid_lines(voxel_grid)
            self.scene.scene.add_geometry("voxel_lines", line_set, mat)


    def get_remap_lut(self, config_file_path):
        carla_config = yaml.safe_load(open(config_file_path, 'r'))
        labels_remap = carla_config["learning_map"]
        
        remap_lut = np.asarray(list(labels_remap.values()))
        return remap_lut


    def get_voxel(self, file_path):

        _, file_extension = os.path.splitext(file_path)

        if file_extension == ".npy":
            remap_lut = self.get_remap_lut(self.opt.config_file)
            labels = np.load(file_path)
            labels = np.expand_dims(labels, axis=-1)

            points_colors = []

            for i in range(labels.shape[0]):
                for j in range(labels.shape[1]):
                    for k in range(labels.shape[2]):
                        label = labels[i, j, k][0]
                        if self.opt.label_map:
                            mapped_label = remap_lut[label]
                        else:
                            mapped_label = label
                        if mapped_label != 0:
                            points_colors.append([mapped_label, i, j, k])

            points_colors = np.array(points_colors)
        elif file_extension == ".txt":
            if os.path.getsize(file_path) == 0:
                points_colors = np.zeros((1, 4))
            else:
                points_colors = np.loadtxt(file_path, delimiter=' ')
        else:
            raise ValueError("Unsupported file format!")

        points = points_colors[:, -3:]

        colors = points_colors[:, 0]
        if opt.dataset == 'carla' : 
            config = yaml.safe_load(open(opt.config_file, 'r'))
            color_map = config["remap_color_map"]
        elif opt.dataset == 'kitti':
            config = yaml.safe_load(open(opt.config_file, 'r'))
            color_map = config["color_map"]
        color = np.asarray([color_map[c] for c in colors])

        return points, color


    def create_voxel_grid_lines(self, voxel_grid):
        lines = []
        colors = []
        voxel_size = voxel_grid.voxel_size
        half_voxel = voxel_size / 2

        for voxel in voxel_grid.get_voxels():
            voxel_center = np.array(voxel.grid_index) * voxel_size
            
            offsets = [[-half_voxel, -half_voxel, -half_voxel], [half_voxel, -half_voxel, -half_voxel],
                    [-half_voxel, half_voxel, -half_voxel], [half_voxel, half_voxel, -half_voxel],
                    [-half_voxel, -half_voxel, half_voxel], [half_voxel, -half_voxel, half_voxel],
                    [-half_voxel, half_voxel, half_voxel], [half_voxel, half_voxel, half_voxel]]
            vertices = [voxel_center + offset for offset in offsets]

            vertex_indices = [
                (0, 1), (1, 3), (3, 2), (2, 0),
                (4, 5), (5, 7), (7, 6), (6, 4),
                (0, 4), (1, 5), (2, 6), (3, 7)
            ]
            for start, end in vertex_indices:
                lines.append([vertices[start], vertices[end]])
                colors.append([0, 0, 0])

        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(np.array(lines).reshape(-1, 3)),
            lines=o3d.utility.Vector2iVector(np.arange(0, len(lines)*2).reshape(-1, 2))
        )
        line_set.colors = o3d.utility.Vector3dVector(np.array(colors))

        return line_set
    
    def count_instance_labels(self, data):
        semantic_instance_counts = defaultdict(int)
        
        for item in data:
            semantic_label, instance_label = item
            semantic_instance_counts[semantic_label] = max(semantic_instance_counts[semantic_label], instance_label + 1)
        
        total_instances = sum(semantic_instance_counts.values())
        return total_instances

def main(opt):
    gui.Application.instance.initialize()
    SpheresApp(opt)
    gui.Application.instance.run()

if __name__ == "__main__":
    opt = parser.parse_args()
    main(opt)