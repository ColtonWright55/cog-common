import json
import math
from pathlib import Path

import numpy as np

from ccog.forge_action import ForgeAction


def actions_to_u(actions: list[ForgeAction], n_phi_disc: int) -> np.ndarray:
    """Pack a list of ForgeActions into the (1, N, 3) warp SlabModel input tensor."""
    n = len(actions)
    u = np.zeros((1, n, 3), dtype=np.float32)
    step = 2.0 * math.pi / n_phi_disc
    for i, a in enumerate(actions):
        u[0, i, 0] = a.rho
        u[0, i, 1] = int(round(a.phi / step)) % n_phi_disc
        u[0, i, 2] = a.z
    return u


def action_to_line(action: ForgeAction, workpiece_length_mm: float | None = None) -> str:
    d: dict = {
        "action": {
            "rho": round(float(action.rho), 2),
            "phi": round(float(np.degrees(action.phi)), 1),
            "z": round(float(action.z), 2),
            "anvil_tool": 2,
            "hammer_tool": 2,
            "scan_after": False,
            "heat_after": False,
            "no_op": False,
        }
    }
    if workpiece_length_mm is not None:
        d["workpiece_length_mm"] = round(float(workpiece_length_mm), 1)
    return json.dumps(d)


def write_toolpath(
    actions: list[ForgeAction],
    path: Path,
    lengths: list[float] | None = None,
) -> None:
    with path.open("w") as f:
        for i, a in enumerate(actions):
            length = lengths[i] if lengths is not None else None
            f.write(action_to_line(a, workpiece_length_mm=length) + "\n")


def load_toolpath(path: Path) -> list[bytes]:
    return [line.encode() for line in path.read_text().splitlines() if line.strip()]


def parse_toolpath(path: Path) -> list[ForgeAction]:
    actions = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        a = json.loads(line)["action"]
        if not a.get("no_op", False):
            actions.append(
                ForgeAction(
                    rho=float(a["rho"]),
                    phi=float(np.radians(a["phi"])),
                    z=float(a["z"]),
                    tool_index_0=int(a.get("anvil_tool", 0)),
                    tool_index_1=int(a.get("hammer_tool", 0)),
                )
            )
    return actions
