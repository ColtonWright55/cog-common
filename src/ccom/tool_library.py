from pathlib import Path

import numpy as np
import open3d as o3d
from loguru import logger

from ccom.tool import Tool

_SUPPORTED_EXTS = {".stl", ".ply", ".obj"}
_PROFILE_SAMPLES = 512


def _raycast(mesh, x_vals, theta_vals, fill_value=0.0):
    scene = o3d.t.geometry.RaycastingScene()
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene.add_triangles(mesh_t)

    X, T = np.meshgrid(x_vals, theta_vals, indexing="ij")
    X_flat, T_flat = X.flatten(), T.flatten()
    origins = np.zeros((len(X_flat), 3))
    origins[:, 0] = X_flat
    directions = np.zeros((len(X_flat), 3))
    directions[:, 1] = np.cos(T_flat)
    directions[:, 2] = np.sin(T_flat)
    rays = np.hstack([origins, directions])
    rays_t = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)

    ans = scene.cast_rays(rays_t)
    t_hit = ans["t_hit"].numpy()
    mask = np.isfinite(t_hit)
    dists = np.where(mask, t_hit, fill_value)
    return origins + dists[:, np.newaxis] * directions


def _profile(mesh: o3d.geometry.TriangleMesh) -> tuple[np.ndarray, np.ndarray]:
    """Raycast a radial profile r(x) off a rotated COPY of `mesh` (face at origin, body at
    +Y) for the finishing-pass policy — used only to compute profile_x/profile_r, then
    discarded. Does not affect the canonical raw mesh/SDF stored on the Tool."""
    rotated = o3d.geometry.TriangleMesh(mesh)
    T = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    rotated.transform(T)
    rotated.translate([0, -rotated.get_min_bound()[1], 0])
    rotated.compute_vertex_normals()

    x_min, x_max = rotated.get_min_bound()[0], rotated.get_max_bound()[0]
    y_max = rotated.get_max_bound()[1]
    x_vals = np.linspace(x_min, x_max, _PROFILE_SAMPLES)

    pcd = _raycast(rotated, x_vals, [0.0])
    pcd[0] = [x_min, y_max, 0]
    pcd[-1] = [x_max, y_max, 0]
    profile_x = x_vals - (x_min + x_max) / 2
    profile_r = pcd[:, 1]
    return profile_x, profile_r


def _load_mesh(path: Path) -> tuple[o3d.geometry.TriangleMesh, float]:
    """Load a tool mesh as exported from Inventor, untransformed — must match gen-sdf's frame,
    since the SDF grid origin is baked in this same raw frame."""
    mesh = o3d.io.read_triangle_mesh(str(path))
    if not mesh.has_vertices():
        raise ValueError(f"Failed to read mesh: {path}")
    mesh.compute_vertex_normals()
    width = float(mesh.get_max_bound()[0] - mesh.get_min_bound()[0])
    return mesh, width


class ToolLibrary:
    def __init__(self, folder: Path | str):
        self.folder = Path(folder)
        if not self.folder.is_dir():
            raise NotADirectoryError(f"Tool directory not found: {self.folder}")

        paths = sorted(p for p in self.folder.iterdir() if p.is_file() and p.suffix.lower() in _SUPPORTED_EXTS)
        if not paths:
            logger.warning("No tool meshes found in {}", self.folder)

        self.tools = tuple(self._load(p) for p in paths)
        self.tool_names = tuple(p.stem for p in paths)

    def _load(self, path: Path) -> Tool:
        mesh, width = _load_mesh(path)
        profile_x, profile_r = _profile(mesh)

        sdf = sdf_origin = sdf_voxel_size = None
        sdf_path = path.with_name(path.stem + "_sdf.npz")
        if sdf_path.exists():
            data = np.load(sdf_path)
            sdf = data["sdf"].astype(np.float32)
            sdf_origin = data["origin"].astype(np.float32)
            sdf_voxel_size = float(data["voxel_size"])
            logger.info("Loaded SDF for '{}' grid={}", path.stem, sdf.shape)
        else:
            logger.warning("No SDF found for tool '{}' (expected {})", path.stem, sdf_path)

        return Tool(
            mesh=mesh,
            width=width,
            sdf=sdf,
            sdf_origin=sdf_origin,
            sdf_voxel_size=sdf_voxel_size,
            profile_x=profile_x,
            profile_r=profile_r,
        )

    def __getitem__(self, idx: int | str) -> Tool:
        if isinstance(idx, str):
            if idx not in self.tool_names:
                raise KeyError(f"Tool '{idx}' not found. Available: {self.tool_names}")
            idx = self.tool_names.index(idx)
        return self.tools[idx]

    def __len__(self) -> int:
        return len(self.tools)
