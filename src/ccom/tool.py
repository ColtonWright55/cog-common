from dataclasses import dataclass

import numpy as np
import open3d as o3d
from scipy.ndimage import map_coordinates


@dataclass
class Tool:
    mesh: o3d.geometry.TriangleMesh  # raw, untransformed frame — matches gen-sdf's frame exactly
    width: float
    sdf: np.ndarray | None = None  # signed distance grid [mm], shape (Nx, Ny, Nz), raw frame
    sdf_origin: np.ndarray | None = None  # grid origin [mm], shape (3,), raw frame
    sdf_voxel_size: float | None = None  # grid spacing [mm]
    profile_x: np.ndarray | None = None  # centered x coords [mm] of a raycast radial profile, shape (N,)
    profile_r: np.ndarray | None = None  # radial profile r(x) [mm], shape (N,)

    def query_sdf(self, points: np.ndarray) -> np.ndarray:
        """Query the tool SDF at points given in the tool's raw mesh frame.

        Args:
            points: (N, 3) float array of query positions in the tool's raw (untransformed) frame.

        Returns:
            (N,) signed distances in mm; +inf where points fall outside the SDF grid.
        """
        coords = (points - self.sdf_origin) / self.sdf_voxel_size
        return map_coordinates(self.sdf, coords.T, order=1, mode="constant", cval=np.inf)
