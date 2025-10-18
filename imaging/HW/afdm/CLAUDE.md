# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a GPGN658 (Seismic Wavefield Imaging) homework assignment focused on acoustic and elastic finite-difference modeling (FDM) for seismic wave propagation. The repository contains implementations of time-domain finite-difference solvers that integrate with Madagascar (a seismic processing software suite).

**Assignment Goal**: Modify the constant-density acoustic wave equation solver to include variable density terms.

## Repository Structure

```
HW/
├── CODE/                    # Shared C implementations
│   ├── MAFDM.c             # Modified Acoustic FDM (with density)
│   ├── MEFDM.c             # Modified Elastic FDM
│   ├── fdutil.c/h          # FD utilities (grid, ABC, interpolation)
│   ├── omputil.c/h         # OpenMP utilities
│   └── SConstruct          # Build script for C programs
├── afdm/                   # Acoustic FDM assignment
│   ├── handout.tex         # Assignment description
│   └── exercise/           # Modeling exercise workspace
│       └── SConstruct      # Madagascar workflow
├── efdm/                   # Elastic FDM assignment
├── artm/                   # Adjoint RTM assignment
└── ertm/                   # Elastic RTM assignment
```

## Core Architecture

### Finite-Difference Solvers

**MAFDM.c** - 2D Acoustic FDM with variable density:
- Solves: ∇·(1/ρ ∇p) = (1/v²) ∂²p/∂t²
- 4th-order spatial accuracy, 2nd-order temporal
- Key features:
  - Variable density support via gradient term: `grad_dot = ∇ρ · ∇u / ρ`
  - OpenMP parallelization
  - Absorbing boundary conditions (ABC + sponge)
  - Free surface support
  - Wavefield snapshots

**MEFDM.c** - 2D Elastic FDM:
- Solves elastic wave equation with orthorhombic anisotropy
- 8th-order spatial accuracy using staggered grid
- Displacement-stress formulation
- Outputs both displacement and P/S-wave potentials (divergence/curl)

### Key Components (fdutil.c/h)

- **fdm2d**: FD grid structure with padding for boundary conditions
- **lint2d**: Linear interpolation for source/receiver injection/extraction
- **abcone2d**: One-way absorbing boundary conditions
- **sponge**: Exponential damping sponge layers

## Build System

### Environment Setup

Source the environment file before building:
```bash
source ../CODE/env.sh
```

This sets up:
- `RSFROOT`: Madagascar installation path
- `RSFSRC`: Madagascar source directory
- OpenMP thread count (`OMP_NUM_THREADS=16`)

### Building C Programs

From `CODE/` directory:
```bash
scons
```

This compiles:
- `sfAFDM` - Acoustic FDM executable
- `sfEFDM` - Elastic FDM executable

The executables link against Madagascar's RSF library.

**Important**: The SConstruct uses `rsfsrc` instead of `rsfroot` to find API files (see line 6 in CODE/SConstruct).

### Running Modeling Exercises

From an exercise directory (e.g., `afdm/exercise/`):
```bash
scons         # Run Madagascar workflow
scons view    # Display results
scons -c      # Clean build artifacts
```

The exercise SConstruct:
1. Generates velocity/density models
2. Creates source wavelets and geometry
3. Calls compiled C programs (via `./CODE/sfAFDM`)
4. Produces plots and results

### Building Documentation

From assignment directories (afdm/, efdm/, etc.):
```bash
./run.sh      # Build PDF handout
# Or step-by-step:
scons
dvips handout.dvi -o handout.ps
ps2pdf handout.ps handout.pdf
```

## Development Workflow

### Modifying FD Solvers

1. Edit source in `CODE/MAFDM.c` or `CODE/MEFDM.c`
2. Rebuild: `cd ../CODE && scons`
3. Run tests: `cd ../afdm/exercise && scons -c && scons`
4. View results: Check generated `.rsf` files or use `scons view`

### Key Modification Points in MAFDM.c

The density term was added at lines 306-312:
```c
/* gradient dot product term */
float grad_dot = DX(ro, ix, iz, idx) * DX(uo, ix, iz, idx)
            + DZ(ro, ix, iz, idz) * DZ(uo, ix, iz, idz);

ua[ix][iz] = lap - grad_dot / (ro[ix][iz] + 1e-10);
```

This implements: acceleration = Laplacian - (∇ρ · ∇u)/ρ

## Common Tasks

### Running a simulation
```bash
cd afdm/exercise
source ../../CODE/env.sh
scons
```

### Viewing specific outputs
```bash
sfin dat.rsf        # Inspect seismogram metadata
< dat.rsf sfgrey | sfpen  # Display seismogram
```

### Debugging density implementation
The code previously had debug output (lines 225-234 in MAFDM.c, now commented) that writes binary velocity/density to check input correctness.

### Performance tuning
- Adjust `OMP_NUM_THREADS` in env.sh
- Modify OpenMP chunk size in fdutil initialization
- Change grid padding `nb` parameter (default: max(NOP, user-specified))

## Madagascar/RSF Integration

The C code uses Madagascar's API:
- `sf_*` functions for I/O (sf_floatread, sf_floatwrite)
- `sf_axis` for dimension metadata
- RSF files (.rsf) are headers pointing to binary data

Exercise SConstructs use Madagascar's Python API:
- `Flow()`: Define processing steps
- `Result()`: Create visualizations
- Helper modules: `awe` (wavefield), `geom` (geometry), `wplot` (plotting)

## Special Notes

- The density array `ro` is preprocessed to `dt²/ρ` for efficiency (MEFDM.c:305)
- Velocity is preprocessed to `v² dt²` for time-stepping (MAFDM.c:237-242)
- Free surface is implemented by zeroing fields in top `nb` rows
- The staggered-grid scheme in MEFDM uses forward (Fx, Fz), backward (Bx, Bz), and centered (Cx, Cz) derivative operators
