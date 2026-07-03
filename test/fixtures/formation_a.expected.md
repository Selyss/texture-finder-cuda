# Ground truth for formation_a.txt

Real-world verified formation, confirmed against the modern texture-rotation
formula (24/24 blocks).

- Expected match: **x=-108723 y=-54 z=-69736**
- Version: modern (1.21.2+), i.e. `version = 0`
- Direction: 0 (no formation rotation applied)
- Search bounds for the e2e test: x in [-175000, -75000], z in [-75000, -25000]
- Formation: 24 blocks, all top faces (isSide=0), mod 4

Verified by evaluating the modern RNG at the expected origin: all 24 rotations
match; other directions and the legacy formula score at chance level (~6/24),
so version and direction are unambiguous.

Any refactor of the RNG, rotation, parser, or kernel must still find exactly
this coordinate (uniquely, within the bounds above) for this file.
