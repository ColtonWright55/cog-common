from pathlib import Path

import tbxdata
from loguru import logger
from typer import Exit

from ccog.tool import Tool
from ccog.tool_library import ToolLibrary

_TOOLDIR_PREFIX = "tooldir"


def default_tool_dir() -> Path:
    return tbxdata.get_data_dir(_TOOLDIR_PREFIX)


def load_tool(tool_dir: Path, tool_index: int = 0) -> Tool:
    """Load a single tool (mesh + SDF + raycast profile) from a tool directory. Raises typer.Exit on failure."""
    lib = ToolLibrary(tool_dir)
    if len(lib) == 0:
        logger.error("No tools found in {}", tool_dir)
        raise Exit(1)
    return lib[tool_index]


def tool_source_paths(tool_dir: Path, tool_index: int = 0) -> list[Path]:
    """Source file(s) backing a tool index: its mesh, plus a sibling `_sdf.npz` if present.

    For copying the exact tool bytes into a frozen test fixture — not for use at test time.
    """
    supported_exts = {".stl", ".ply", ".obj"}
    tool_paths = sorted(p for p in Path(tool_dir).iterdir() if p.is_file() and p.suffix.lower() in supported_exts)
    if not tool_paths:
        raise NotADirectoryError(f"No tool meshes found in directory: {tool_dir}")
    mesh_path = tool_paths[tool_index]
    sdf_path = mesh_path.with_name(mesh_path.stem + "_sdf.npz")
    return [mesh_path, sdf_path] if sdf_path.exists() else [mesh_path]
