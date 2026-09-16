import struct
from dataclasses import dataclass, field

import numpy as np

_FMT = "<fffBB"  # rho, phi, z as f32; tool_index_0, tool_index_1 as uint8
_SIZE = struct.calcsize(_FMT)


@dataclass
class ForgeAction:
    """A single planned forging action in cylindrical coordinates.

    Args:
        rho: Radial depth [mm].
        phi: Tool angle [rad] (0 = +Y, pi/2 = +Z).
        z: Axial position [mm].
        tool_index_0: Anvil tool index (default 0).
        tool_index_1: Hammer tool index (default 0).
    """

    rho: float
    phi: float
    z: float
    tool_index_0: int = field(default=0)
    tool_index_1: int = field(default=0)

    def pack(self) -> bytes:
        return struct.pack(_FMT, self.rho, self.phi, self.z, self.tool_index_0, self.tool_index_1)

    @staticmethod
    def unpack(data: bytes) -> "ForgeAction":
        rho, phi, z, t0, t1 = struct.unpack(_FMT, data[:_SIZE])
        return ForgeAction(rho=rho, phi=phi, z=z, tool_index_0=t0, tool_index_1=t1)

    def as_array(self) -> np.ndarray:
        """Return [rho, phi, z] as float64 — for visualization only, does not include tool indices."""
        return np.array([self.rho, self.phi, self.z], dtype=np.float64)
