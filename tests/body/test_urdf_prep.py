from pathlib import Path

from body.urdf_prep import ascii_stl_to_binary, prepare_urdf

ASSETS = Path(__file__).resolve().parent.parent.parent / "body" / "assets" / "robot"


def test_ascii_stl_to_binary_converts_and_counts_triangles(tmp_path):
    dst = tmp_path / "lamp_shade_binary.stl"
    count = ascii_stl_to_binary(ASSETS / "lamp_shade.stl", dst)
    assert count > 0
    assert dst.exists()
    # Binary STL: 80-byte header + 4-byte count + 50 bytes/triangle.
    assert dst.stat().st_size == 84 + count * 50


def test_prepare_urdf_points_at_the_binary_mesh(tmp_path):
    patched = prepare_urdf(tmp_path)
    content = patched.read_text()
    assert 'filename="assets/lamp_shade.stl"' not in content
    assert str(tmp_path) in content
