# texture-finder-cuda
- Most of the resources in this README have been provided from [19MisterX98](https://github.com/19MisterX98)
- Handles vanilla texture rotations (1.13 – 1.21.1 and 1.21.2+)
- Runs on the GPU using CUDA
- *Does not require the world seed*

## Building
Linux/macOS (CUDA toolkit required; `nvcc` found on PATH or at `/usr/local/cuda`):

```
make            # builds build/main for the GPU in the machine (ARCH=native)
make ARCH=sm_86 # or target a specific architecture
make test       # host-only unit tests, no GPU needed
```

Windows: `make windows` (edit `CCBIN` in the Makefile if your Visual Studio
install path differs).

## Obtaining Rotations

### Texture Packs
There are two texture packs that can be used to get texture rotations:\
[Manual Texture Rotations](https://github.com/19MisterX98/TextureRotations/releases/download/1/Manual_texture_rotations.zip)
[Textures to Numbers](https://github.com/19MisterX98/TextureRotations/releases/download/1/Textures_to_numbers.zip)

### Usage
The "Manual Texture Rotations" pack retextures certain blocks that have block states into blocks that have texture rotations.
When you change the blockstate via debug stick, these blocks will change their visual rotation. This can be used to recreate screenshots and match all the block rotations used. Read the usage instructions at https://github.com/19MisterX98/TextureRotations

## Obtaining Orientation
The orientation of the formation is very important. It is important to try and get the direction of the recreation correct, or the results will be inaccurate. There are certain blocks that always face a certain direction no matter how they are placed.

These blocks are as follows:

<pre>
Glowstone
Any Ore
Prismarine
Cobblestone
...
</pre>


## Textures to Numbers
After recreating the formation, overlay the "Textures to Numbers" resource pack.
This will display the rotation number on each rotatable block.

## Inputting Data
Formation files can live anywhere, and have the `.txt` file extension.
The format is as follows:
`x`, `y`, `z`, `rotation`, `isSide`

<pre>
1 0 0 1 0
</pre>

The `x`, `y`, and `z` coordinates are relative coordinates to an origin block of your choosing.
The above example shows the rotation info of a block that is 1 block in the positive `x` direction from the origin.
The block has the same relative `y` and `z` values as the origin.

The fourth number is the rotation number. This is the value from the "Textures to Numbers" resource pack.

The final number is the boolean value for `isSide`. If the block being entered only has the side exposed, as opposed to the top, this should be set to `1`. If the block only shows the top, it should be set to `0`. (A side face only distinguishes 2 of the 4 rotation states, so a side block carries half the information of a top block.)

Blank lines and lines starting with `#` are ignored.

It is recommended to get as many blocks as possible, and narrow down the possible results to 1 — the program prints a warning when the formation is too weak for the searched volume.

## Parameters

```
main <x_min> <x_max> <y_min> <y_max> <z_min> <z_max> <version> <file> <direction>
```

- `x_min` / `x_max`: range of the searched `x` coordinates (inclusive)
- `y_min` / `y_max`: range of the searched `y` coordinates (inclusive)
- `z_min` / `z_max`: range of the searched `z` coordinates (inclusive)
- `version`: `0` for Minecraft 1.21.2+, `1` for 1.13 – 1.21.1
- `file`: the formation file
- `direction`: the direction the formation is facing — `0` North, `1` West,
  `2` South, `3` East, or `all` to try every orientation and report which
  one matched (recommended when unsure)

Example:

```
./build/main -10000 10000 -64 100 -10000 10000 0 formation.txt all
```

## Version Table
Depending on the version of the client, the mode will need to be changed.

| MC Version  | Reference mode      | This program        |
|-------------|---------------------|---------------------|
| \<=1.12.2   | Vanilla12Textures   | not implemented yet |
| 1.13-1.21.1 | Vanilla21_1Textures | `version = 1`       |
| 1.21.2+     | VanillaTextures     | `version = 0`       |


| Sodium Version | MC Version  | Mode                       |
|----------------|-------------|----------------------------|
| 1.0-4.1        | 1.16-1.18.2 | SodiumTextures (not implemented yet) |
| 4.2-4.8        | 1.19-1.19.3 | Sodium19Textures (not implemented yet) |
| 4.9+           | 1.19.3+     | Uses the MC implementation |

## Correctness and verification
The rotation formulas and the direction convention are documented in
[docs/SEMANTICS.md](docs/SEMANTICS.md), and every change is validated by a
four-layer test pyramid (unit vectors, a real verified formation, a ~29M
value differential against the Java reference implementation, and GPU
end-to-end tests) — see that document and
[docs/ENGINEERING_LOG.md](docs/ENGINEERING_LOG.md) for details and
performance history.

# Credits
https://github.com/19MisterX98/TextureRotations
