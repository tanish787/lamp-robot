"""PyBullet's mesh loader only understands binary STL; the supplied
lamp_shade.stl is ASCII. Convert it once per run into a cache directory and
patch a copy of the URDF to point at the binary mesh, leaving the checked-in
assets untouched. Ported from the lab's src/render_lamp.py.
"""

import struct
from pathlib import Path

ASSETS_DIR = Path(__file__).resolve().parent / "assets" / "robot"


def ascii_stl_to_binary(src: Path, dst: Path) -> int:
    """Convert an ASCII STL to binary STL. Returns the triangle count."""
    normals: list[tuple[float, float, float]] = []
    triangles: list[list[tuple[float, float, float]]] = []
    current: list[tuple[float, float, float]] = []

    with open(src, "r") as handle:
        for line in handle:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "facet" and parts[1] == "normal":
                normals.append(tuple(float(v) for v in parts[2:5]))
                current = []
            elif parts[0] == "vertex":
                current.append(tuple(float(v) for v in parts[1:4]))
            elif parts[0] == "endfacet":
                triangles.append(current)

    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "wb") as handle:
        handle.write(b"\0" * 80)
        handle.write(struct.pack("<I", len(triangles)))
        for normal, tri in zip(normals, triangles):
            handle.write(struct.pack("<3f", *normal))
            for vertex in tri:
                handle.write(struct.pack("<3f", *vertex))
            handle.write(struct.pack("<H", 0))

    return len(triangles)


def prepare_urdf(cache_dir: Path) -> Path:
    """Write a URDF copy in `cache_dir` that references a binary shade mesh."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    binary_stl = cache_dir / "lamp_shade_binary.stl"
    ascii_stl_to_binary(ASSETS_DIR / "lamp_shade.stl", binary_stl)

    urdf_text = (ASSETS_DIR / "dummy_lamp_5dof.urdf").read_text()
    urdf_text = urdf_text.replace(
        'filename="assets/lamp_shade.stl"', f'filename="{binary_stl}"'
    )
    patched = cache_dir / "lamp_patched.urdf"
    patched.write_text(urdf_text)
    return patched
