import hashlib
from pathlib import Path

import numpy as np
import open3d as o3d
import tbxdata
from foxglove_schemas_protobuf.SceneUpdate_pb2 import SceneUpdate


def decode_scene_update_mesh(data: bytes) -> o3d.geometry.TriangleMesh | None:
    """Decode a foxglove.SceneUpdate message's first triangle list into an open3d mesh."""
    su = SceneUpdate()
    su.ParseFromString(data)
    if not su.entities or not su.entities[0].triangles:
        return None

    tri = su.entities[0].triangles[0]
    points = np.array([[p.x, p.y, p.z] for p in tri.points], dtype=np.float64)
    if len(points) == 0:
        return None

    if tri.indices:
        triangles = np.array(tri.indices, dtype=np.int32).reshape(-1, 3)
    else:
        triangles = np.arange(len(points), dtype=np.int32).reshape(-1, 3)

    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(points), o3d.utility.Vector3iVector(triangles))
    mesh.compute_vertex_normals()
    return mesh


def load_triangle_mesh(path: str | Path) -> o3d.geometry.TriangleMesh:
    path = tbxdata.get_data(str(path))
    mesh = o3d.io.read_triangle_mesh(str(path))
    if not mesh.has_vertices():
        raise FileNotFoundError(f"Failed to load a mesh from {path}")
    mesh.compute_vertex_normals()
    return mesh


def mesh_volume(mesh: o3d.geometry.TriangleMesh) -> float:
    """Signed volume via the divergence theorem — same formula as `mesh.get_volume()`,
    but tolerant of the hair-width self-intersecting facets that make Open3D's strict
    `is_watertight()` gate reject real (closed) Inventor STL exports."""
    v = np.asarray(mesh.vertices)
    t = np.asarray(mesh.triangles)
    v0, v1, v2 = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    return float(np.sum(np.einsum("ij,ij->i", v0, np.cross(v1, v2))) / 6.0)


def mesh_hash(mesh: o3d.geometry.TriangleMesh) -> str:
    h = hashlib.md5(np.asarray(mesh.vertices).tobytes())
    h.update(np.asarray(mesh.triangles).tobytes())
    return h.hexdigest()
