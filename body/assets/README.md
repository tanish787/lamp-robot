# Body assets

## `robot/`

The 5-DOF lamp URDF and its shade mesh. `body/urdf_prep.py` patches the URDF
and converts the mesh into a cached, PyBullet-loadable form at startup.

## `sfx/` and `music/`

**These are placeholder clips, not final audio.** They are short generated
tones committed so the demo runs end to end out of the box — every
`play_sfx("chime")` fired on engagement would otherwise raise
`FileNotFoundError`. Replace them with real recordings before recording the
actual demo.

Regenerate them with `python -m scripts.generate_placeholder_audio` from the
repository root. The set of legal clip names is enumerated in
`shared.action_vocabulary.SFX_NAMES` / `MUSIC_TRACKS` — adding a file here
without adding its name there means Brain can never ask for it.

| File | Placeholder content |
|---|---|
| `sfx/chime.wav` | rising two-tone chime, ~0.35 s |
| `sfx/confirm.wav` | short single blip, ~0.20 s |
| `sfx/alert.wav` | low buzz, ~0.30 s |
| `music/ambient.wav` | quiet sustained drone, ~2.0 s |
