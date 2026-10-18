@page volume_filtering Offline volume-filtered closures

# Offline volume-filtered closures

`mfc.volume_filter` reads one native MFC snapshot at a time, computes compressible
volume-filtered momentum and total-energy closures on the full periodic domain,
samples particle-centered maps, and accumulates statistics. It does not change
simulation results or add work to the time integrator. Full-domain closure fields
are discarded as each component's maps are accumulated.

This initial implementation targets a **single calorically perfect ideal gas,
constant dynamic viscosity/conductivity, uniform three-dimensional Cartesian
mesh, fully periodic boundaries, and impermeable spheres**. Particle positions,
radii, translational velocities, and angular velocities may change between
snapshots. All spheres use the configured velocity and thermal boundary type.
Phase change, mixtures, nonideal EOS, variable transport, nonuniform grids, and
non-spherical bodies require additional reconstruction logic and are not covered.

Execution is **single-node**, with threaded SciPy FFTs (`fft_workers`). One global
snapshot and several work arrays must fit in RAM. Reading simulation MPI shards
does not make this a distributed FFT implementation. Memory is independent of the
number of snapshots; it still grows with grid size, surface quadrature points,
map size, and the number of retained statistics. A double scalar field on a
512 x 256 x 256 grid is 256 MiB; budget several GiB beyond the input state for
filtered fields, derivatives, and FFT temporaries. This is a draft analysis tool:
validate interface reconstruction against a representative DNS before production.

## Run

From the repository root, using Python with NumPy and SciPy installed:

```bash
cp misc/volume_filter/config.example.json /path/to/case/closures.json
# Edit the copied configuration to match the actual simulation and saved file indices.
PYTHONPATH=toolchain python -m mfc.volume_filter /path/to/case/closures.json
```

The example's gas constants, mesh, indices, and filter widths are
illustrative, not inferred from a case. All paths are relative to the JSON file.
Use mutually consistent dimensional or nondimensional units. In particular,
`mu` is the dynamic viscosity used in tau = mu*S(u), **not MFC's Reynolds number
input**; `gamma` is the physical heat-capacity ratio, not MFC's stored 1/(gamma-1).
`gas_constant` is the specific gas constant, not the universal gas constant.
`grid.shape` is the actual cell count: `(m_glb+1,n_glb+1,p_glb+1)` for MFC.
`origin` is the lower domain face; the first cell center is origin + dx/2.
The grid in the JSON is an explicit assertion of the saved mesh: verify it
against the case/grid files. No coordinates or precision are guessed from bytes.

Progress is printed and flushed to the terminal or batch-job log automatically.
The processor reports particle reading, fluid-state reconstruction, each filter's
closure computation, and final output writing. During map accumulation, each
snapshot/filter/component reports its own processed-particle counter, for example:

```text
Snapshot 1/3 step=1000 filter=1/2 sigma=0.025 F_x: particles 143/1000 (14.3%), elapsed 5.0s
```

Each counter starts at zero and ends at the total particle count. Intermediate
updates are limited to one every five seconds, checked after each particle
finishes. Override the interval without changing the configuration:

```bash
PYTHONPATH=toolchain python -m mfc.volume_filter /path/to/case/closures.json --progress-interval 1
```

Use `--progress-interval 0` to print every particle. Counts reset for each field
and filter because the code maps one full-domain closure component at a time.
They count processed particles, including any with invalid map locations; valid
statistical support is tracked internally when samples are accumulated.
The percentage and elapsed time describe the current field's mapping pass, not
the entire snapshot. Full-domain FFT/closure calculations occur between mapping
passes and do not have a particle counter. Progress reporting does not change
numerical results.

The processor accumulates statistics in memory over all configured snapshots and
writes **one final NPZ file only**, after every snapshot succeeds. It does not
write checkpoints or per-snapshot map files. Final output is streamed to a
temporary file, flushed, and atomically published under the configured name;
the temporary file is not a recoverable checkpoint. An interrupted or failed run
must restart from the first configured snapshot. The removed `--resume` and
`--max-snapshots` options are no longer accepted. Legacy `save_snapshot_maps`
configuration entries are ignored and can be deleted.

Only consume completely written, immutable snapshots. Input fingerprints are
recorded as provenance, and each snapshot is checked for changes during
processing. Existing output files are refused; choose a new output path or
remove the old file before running. Only one job may write a particular output
path. Raw simulation files are never deleted.

## Native output layouts

Use full-resolution conservative output (`down_sample = F`). The processor reads
five explicitly selected equations: density, three momentum components, and total
energy density. `indices` are **one-based** and must match the actual MFC equation
layout. Extra state equations count towards `nvars` even when not consumed. The
single-gas assumptions must hold in all physical fluid cells.

- `format: "shared"`: MPI-IO `restart_data/{step}.dat`, contiguous global arrays
  in variable-major order, each array flattened in Fortran order. No halos or
  record markers. File size must equal nvars * Nx * Ny * Nz * dtype.itemsize.
- `format: "shards"`: `restart_data/lustre_{step}/{step}_{rank:07d}.dat`.
  Supply `blocks` with each file's `rank`, global `offset`, and local `shape`.
  Blocks must cover the global mesh exactly once. Files contain consecutive
  Fortran-order local arrays, without halos. This reader assembles them on one node.
- `format: "serial"`: `p_all/p{rank}/{step}/q_cons_vf{var}.dat`, one Fortran
  sequential unformatted record per variable. Also supply `blocks` for multiple
  ranks. `record_header_dtype` defaults to `<u4`; split/subrecord records are not
  supported by SciPy's FortranFile and must be converted or written via MPI-IO.

Example sharded I/O configuration (choose actual shapes/offsets from the run):

```json
{
  "format": "shards",
  "dtype": "<f8",
  "particle_dtype": "<f8",
  "integer_dtype": "<i4",
  "nvars": 6,
  "indices": {"rho": 1, "momentum": [2, 3, 4], "energy": 5},
  "flow": "restart_data/lustre_{step}/{step}_{rank:07d}.dat",
  "particles": "restart_data/lustre_{step}/ib_state_{step}_{rank:07d}.dat",
  "particle_shards": true,
  "blocks": [
    {"rank": 0, "offset": [0, 0, 0], "shape": [256, 256, 256]},
    {"rank": 1, "offset": [256, 0, 0], "shape": [256, 256, 256]}
  ]
}
```

`dtype` is flow storage precision/endianness; `particle_dtype` is IB working
precision/endianness and can differ. The default shared IB file is the native
`ib_state_{step}.dat`: 20 reals per body, in global ID order. Shards contain an
integer count followed by integer global ID + 20 reals per local body. Identical
replicated particle records are deduplicated; inconsistent duplicates fail.
All IDs 1..N must be present. Select snapshots using saved file indices only:

```json
"snapshots": [1000, 2000, 3000]
```

Each index replaces `{step}` in both the flow and particle filename patterns.
It is the **saved file index**, which need not be the integration step when
adaptive output is used. Indices must be unique nonnegative integers; processing
follows the supplied order. Physical times are not user inputs and are not used
for weighting or snapshot selection. The first particle record's timestamp is
retained as descriptive output metadata only. Ensure that the flow and particle
files come from the same saved output.

Legacy objects such as `{"step": 1000}` or `{"step": 1000, "time": 0.001}`
are also accepted. Any supplied `time` is ignored, including stale example times.
Legacy `time_weighting` is ignored as well: all realizations have equal weight.
These obsolete fields are removed before recording the normalized configuration.
Physical time is retained only as descriptive metadata read from the files.

The 20 fields are time, force(3), torque(3), velocity(3), angular_velocity(3),
angles(3), centroid(3), radius. The geometry and kinematics are used directly.
Forces and torques are retained in the input, but never substituted for distributed
surface traction. MFC's force history has stage-timing considerations; its
`patch_ib%%force` can also include collisions/body forces. Such values are not
unconditionally valid hydrodynamic force checks.

## Execution order and map performance

For each configured saved index, the processor:

1. Reads particle state, builds the periodic fluid/solid mask, and reads the five
   conservative fluid components. Reconstructs velocity, pressure, and temperature.
2. For each filter width, constructs the Gaussian Fourier multiplier and filters
   the fluid indicator, density, momentum, and pressure. Produces alpha, intrinsic
   density, Favre velocity components, and Favre temperature.
3. Builds surface quadrature and reconstructed traces, then yields closure
   components sequentially: R_u and R_mu tensor components; R_uu, R_Tu, R_tau_u,
   and R_q vector components; finally F_x/F_y/F_z, Q, and F_u. Intrinsic R terms
   also yield their alpha-weighted companions. Dependencies and work arrays are
   retained as needed, but all resulting full-domain components are not stored
   simultaneously.
4. For each selected yielded component, loops over particles. Samples the closure
   at the particle-centered map, averages theta first for axisymmetric maps, then
   updates pointwise count, mean, and central sums through order four. There is
   no Python loop over the individual interpolation points.
5. Merges that particle distribution into the cumulative ensemble over all
   particle/timestep pairs, preserving within-snapshot scatter.
6. Continues to the next saved index without writing map data. After every
   snapshot/filter/component finishes, writes one final statistics file.

`fields` currently selects **mapping and storage**, not a dependency-pruned
closure calculation: unselected fields are still computed/yielded and skipped by
the caller. Thus selecting only `F_x` reduces map work and statistics storage but
does not eliminate the preceding closure computations. `fft_workers` controls
FFTs, not particle mapping. Particle loops run sequentially on one process.

Box interpolation uses the Cartesian product structure: indices and fractions
are calculated on three 1D axes, then eight corner contributions are broadcast
across the cube. This avoids recomputing million-point coordinate/index arrays
for every corner. The unstructured periodic sampler remains available for
surface traces and axisymmetric rings. Both paths reject an invalid corner
when its weight exceeds 1e-14; zero-weight invalid neighbors do not invalidate a
sample. Unit-weight moment updates use the singleton central-moment recurrence;
general weighted/grouped merges retain the existing formulas. Roundoff can differ
from the generic implementation, but the mathematical interpolation/statistics
are unchanged.

A reproducible benchmark compares the generic path against the optimized path:

```bash
PYTHONPATH=toolchain python misc/volume_filter/benchmark_maps.py --points 100 --particles 4 --repeats 3
```

For one scalar field on a 128^3 source grid and four 100^3 particle maps, a local
three-repeat run measured median mapping times of 2.48 s (generic) and 0.31 s
(optimized), about 8x faster. Interpolation was 1.95 vs 0.23 s; moment updates
were 0.52 vs 0.08 s. The script checks counts and central sums against the generic
path. These are local kernel timings, excluding input reads, full-domain closure
FFTs, final output, and production DNS validation.

A 100^3 map contains one million locations, 1000 times a 10^3 map. Mapping cost
still scales with map locations x particles x selected components x filter
widths x snapshots. All 59 scalar outputs are selected if `fields` is omitted.
At 100^3 locations, the cumulative accumulator uses about 48 MB per scalar
component/filter, or 2.8 GB for all 59 outputs at one filter width, excluding
DNS/work arrays and temporaries. The final NPZ stores four arrays per
component/filter: 32 MB each, or 1.9 GB for all 59. This is about 88 percent less
array storage than the earlier 34-array output, while removing the separate
snapshot-average accumulators halves persistent statistics memory. Output is
written only once at the end. These figures use decimal GB and scale linearly
with filter count. Restrict `fields` to needed components to reduce mapping work,
statistics memory, and final output size.

## Closure conventions

This follows Conrad Delgado's supplied *Volume Filtering* derivation. Let
C[f] = G_sigma * (I_f f), alpha = C[1], m = C[rho], and
u_tilde = C[rho u]/m. An overbar is intrinsic (C[f]/alpha), and a tilde is Favre
(C[rho f]/m). The kernel is the periodized continuum Gaussian, evaluated with the
Fourier multiplier exp(-sigma^2 |k|^2/2). Each `sigma` is a standard deviation in
physical coordinate units. It is not a diameter, FWHM, or number of grid cells.
This spectral discretization is not identical to truncating a sampled Gaussian
in real space. Check convergence if sigma is comparable to a cell width.

| Output | Definition |
|---|---|
| `R_u_ij` | rho_bar (tilde(u_i u_j) - u_tilde_i u_tilde_j) |
| `R_mu_ij` | tau_bar_ij - mu*S_ij(u_tilde) |
| `R_uu_i` | rho_bar/2 (tilde((u dot u)u_i) - tilde(u dot u)u_tilde_i) |
| `R_Tu_i` | gamma/(gamma-1) (overline(p u_i) - p_bar u_tilde_i) |
| `R_tau_u_i` | overline((u dot tau)_i) - (u_tilde dot tau_bar)_i |
| `R_q_i` | overline(q_i) + k*partial_i T_tilde |
| `F_i` | sum of surface integrals of [(p I - tau) n]_i G |
| `Q` | sum of surface integrals of (q dot n) G |
| `F_u` | sum of surface integrals of u_surface dot [(p I - tau)n] G |

Here S(u) = grad(u) + grad(u)^T - 2/3 div(u) I, q = -k grad(T), and
n points **outward from the particle into the fluid**. The nine closure fields
include six independent components (`xx,xy,xz,yy,yz,zz`) per symmetric tensor,
three Cartesian components (`x,y,z`) per vector, and two interfacial scalars.
Each `R_*` also has an `alpha_R_*` output. `F`, `Q`, and `F_u` are already source
densities and must not be multiplied by alpha again. Additional outputs are
`alpha`, `rho_bar`, `u_tilde_x/y/z`, and `T_tilde`.

Optional `fields` selects the names to map/store, e.g.
`["R_u_xx", "alpha_R_u_xx", "F_x"]`. Omitting it stores all components.
Selection currently reduces map/statistics work, not all shared closure work.
No full-domain closure volumes are written.

Intrinsic quantities are undefined where alpha < `alpha_min` or filtered mass is
nonpositive: these values are NaN, not clipped into apparently valid results.
Their weighted companions are also invalidated because their nonlinear terms
use the same mass denominator. Invalid samples are excluded internally from the
ensemble; unsupported locations produce NaN statistics.

## Interface and derivative discretization

The fluid indicator is reconstructed from periodic spheres at cell centers.
Solid/ghost values are excluded from all fluid integrals, including derivatives.
Fluid gradients use second-order centered differences where both neighbors are
fluid and second-order one-sided differences otherwise. Stencils do not cross
solid cells. A gap without a contiguous three-fluid-cell stencil raises an error.
Derivatives of filtered quantities use spectral derivatives and the quotient rule
for Favre ratios; the odd derivative's Nyquist coefficient is zero.

Sphere surfaces use Gauss-Legendre quadrature in cos(theta) and uniform azimuth
(`polar_points` by `2*polar_points`). Pressure, stress, and (when requested) heat
flux at the surface are extrapolated as 2*f(d)-f(2d), with trilinear fluid-only
samples along the outward normal and d = `offset_cells` * max(dx). Both sample
stencils must be fluid. Neighbor interference causes an error instead of quietly
using solid data. Changing offsets is not a substitute for a grid convergence
study; close gaps may need a more sophisticated interface reconstruction.

Weighted surface sources are deposited conservatively with cloud-in-cell weights,
divided by cell volume, and then Gaussian-filtered. This introduces a discrete
surface-delta approximation and an additional grid-scale assignment smoothing.
Refine the grid, quadrature, and sample offsets at fixed physical sigma to assess
its error. **This reconstruction is not bitwise equivalent to MFC's IB force
operator and has not yet been validated against a production particle DNS.**

`velocity_bc: "no_slip"` uses U_p + Omega_p cross (y-X_p) for surface power.
For `"slip"`, the extrapolated tangential fluid velocity is retained and the
normal velocity is set to the rigid-body normal velocity. `thermal_bc:
"adiabatic"` enforces Q=0; `"reconstruct"` uses extrapolated q dot n (appropriate
when assessing resolved nonzero heat transfer). The saved fluid state must
already represent the intended thermal boundary condition.

For a normalized kernel, the integral of F equals minus the total hydrodynamic
force on particles. Under no slip, the integral of F_u equals minus the sum of
U_p dot F_p^hydro + Omega_p dot torque_p^hydro. These are useful checks, but they
do not imply that local source fields can be recovered from integrated forces.
Constant pressure produces a nonzero local interfacial F even though its net
force on a closed sphere is zero. In equilibrium this balances the pressure/
volume-fraction contribution; numerical surface and mask discretizations need
not cancel pointwise at finite resolution.

## Maps and statistics

A box map uses one cube side length and one point count shared by all directions:

```json
"map": {
  "kind": "box",
  "length": 0.2,
  "points": 21
}
```

`length` is the full side length in simulation coordinate units, **not a half-width
or a particle-diameter multiple**. `points` is an integer >= 2 giving the number
of points per direction. The code constructs
`axis = np.linspace(-length/2, length/2, points)` and a 3D meshgrid with
`indexing="ij"`. Both endpoints are included, with spacing `length/(points-1)`.
Odd point counts include the particle center; even counts straddle it symmetrically.
The example gives a 21 x 21 x 21 map spanning -0.1 to +0.1 in each direction.

At each snapshot the cube is translated to each particle's current center, and
the full-domain closure fields are sampled using periodic trilinear interpolation.
Each output statistic has shape `(points, points, points)` in x/y/z order. This
produces a spatial map; it does not collapse the cube to one number. Axes remain
laboratory axes. The old independent x/y/z lists are no longer accepted; replace
them with `length` and `points`, and use a new output path for the changed map.
All particles have equal weight, including when radii differ. Filters include the
entire suspension; a neighborhood overlapping another particle is intentional.
The cube side length must be strictly smaller than every periodic domain length
to avoid repeated periodic images at its endpoints. Axisymmetric offsets must
also remain inside half the domain size.

For an axisymmetric map, use independent axial, radial, and angular counts:

```json
{
  "kind": "axisymmetric",
  "axis": 0,
  "axial_length": 0.2,
  "axial_points": 21,
  "radius": 0.1,
  "radial_points": 11,
  "azimuthal_points": 32
}
```

`axial_length` is the full axial extent centered on the particle; `radius` is the
maximum radial distance from the axis (not the particle radius). Both use the
simulation's coordinate units. The code constructs a cylindrical meshgrid from:

- Axial: `np.linspace(-axial_length/2, axial_length/2, axial_points)`.
- Radial: `np.linspace(0, radius, radial_points)`.
- Angular: `np.arange(azimuthal_points) * 2*np.pi/azimuthal_points`.

Axial and radial endpoints are included. Theta covers the full circle in radians,
with 2*pi excluded to avoid counting the same angle twice. The three counts are
independent: axial/radial counts must be integers >= 2, and the angular count
must be >= 4 (default 32). An odd axial count includes the center plane. The
axial length must be smaller than the domain length along `axis`, and twice the
radial extent must be smaller than both transverse domain lengths.

The mesh is translated to each particle's current center, with periodic
trilinear interpolation of the closure fields. Each (axial,radial) ring is
averaged over theta before particle/timestep statistics are accumulated. The
example samples 21 x 11 x 32 locations per particle and returns 21 x 11 statistic
maps. Repeated angular samples at r=0 still contribute just one ring realization.
A complete configuration is provided in
`misc/volume_filter/config.axisymmetric.example.json`. Legacy `axial`/`radial`
coordinate lists must be replaced with the extent/count parameters above.

 **Components remain Cartesian**; this is not a
cylindrical-component tensor/vector transform. The angularly averaged Cartesian
radial components can cancel. A future cylindrical-component or slip-aligned
map should be a separate explicit choice. An incomplete ring is invalid rather
than averaging only its valid angles. The axis is x=0, y=1, z=2.

For the primary ensemble, **each valid particle/timestep pair is one equally
weighted realization at each map location**. For a mapped closure component X,
with N valid pairs, the population estimators are

\f[
\mu = \frac{1}{N}\sum_{t,p}X_{p,t}, \qquad
c_r = \frac{1}{N}\sum_{t,p}(X_{p,t}-\mu)^r, \quad r=2,3,4.
\f]

The saved statistics are mean, population variance c2, skewness c3/c2^(3/2),
and Pearson kurtosis c4/c2^2. A Gaussian population has kurtosis 3. Variance uses
N rather than N-1; the statistics are empirical population estimators without
finite-sample bias corrections.

Particle realizations are accumulated before particle averaging. Snapshot batches
are merged using internal counts, means, and all three central sums M2, M3, M4;
merging only particle-average maps would lose within-snapshot scatter. These
pairwise formulas follow
[Pebay, SAND2008-6212](https://doi.org/10.2172/1028931). Accumulators use float64,
with int64 valid-realization counts, and memory remains independent of the number
of snapshots. Counts, weights, and central sums are needed internally for stable
accumulation, but are not written to disk. Final output entries are streamed
instead of duplicating all derived arrays in memory.

The version-3 NPZ uses **field, statistic, filter index** ordering:

| Example key | Meaning |
|---|---|
| `F_x.mean.sigma0` | Ensemble mean of F_x at the first configured filter width |
| `F_x.variance.sigma0` | Population variance of F_x |
| `F_x.skewness.sigma0` | Standardized third central moment of F_x |
| `F_x.kurtosis.sigma0` | Pearson kurtosis of F_x |
| `R_u_xy.mean.sigma1` | Mean xy momentum-flux closure at the second filter width |

`metadata` is the only entry other than these four statistics per selected
field/filter. No `.time`, `.pooled`, raw moments, central moments, counts,
weights, central sums, or excess-kurtosis arrays are saved. All statistics use
the equally weighted particle/timestep ensemble. With all 59 fields and one
filter width, the NPZ has 237 keys including metadata, versus 2007 previously.

An unsupported location has NaN statistics. A constant realization sequence has
zero variance and NaN skewness/kurtosis. Invalid samples are excluded separately
at each map location, so the internal realization count may vary spatially.
For axisymmetric maps, one realization is a particle's ring average at one
timestep; azimuthal points are not additional realizations. Statistics therefore
exclude variability removed by the angular average.

Every valid particle/timestep pair has unit weight even with irregular output
times or changing particle counts. Exclude startup transients by selecting the
desired saved file indices. Separate statistics over instantaneous
particle-average maps are no longer computed.
`metadata` contains the full configuration, filter widths, processed count,
particle counts, file-derived `snapshot_times`, input fingerprints, and explicit ensemble/normalization
conventions. Box coordinates can be reconstructed from `config["map"]["length"]`
and `config["map"]["points"]` using the linspace above. Axisymmetric coordinates
can likewise be reconstructed from the stored extent/count parameters. Metadata
also records `output_mode: "final_only"`, output format version 3, the key naming
convention, retained statistic names, and the total completed snapshot count.

The ensemble calculation treats particle/timestep pairs as distinct realizations.
It does not demonstrate physical independence: nearby particles and closely
spaced times may be correlated. No effective sample size, confidence intervals,
or autocorrelation estimates are inferred.

```python
import json
import numpy as np

with np.load("closure_statistics.npz", allow_pickle=False) as data:
    metadata = json.loads(str(data["metadata"]))
    mean = data["F_x.mean.sigma0"]
    variance = data["F_x.variance.sigma0"]
    skewness = data["F_x.skewness.sigma0"]
    kurtosis = data["F_x.kurtosis.sigma0"]
```

Completed version-2 output files can be reduced without rerunning the simulation
or closure calculations:

```bash
PYTHONPATH=toolchain python -m mfc.volume_filter.compact old_statistics.npz compact_statistics.npz
```

The converter copies only the four pooled statistics, renames their keys, and
updates metadata. It streams one array at a time, leaves the original intact,
refuses an existing destination, and rejects incomplete runs or incompatible
weighting. Existing plotting scripts must switch to the new key names.

## Validation

Run the focused suite without compiling MFC:

```bash
PYTHONPATH=toolchain python -m unittest mfc.volume_filter.test_volume_filter mfc.volume_filter.test_moments -v
```

It exercises Fourier-mode attenuation, filtered momentum and energy identities,
fluid-only differences, cubic mesh construction, affine interpolation, periodic
particle registration, conservative deposition, analytic
surface-force/power integrals, map statistics, shared/sharded/Fortran-record input,
and a single final write after all snapshots complete. Higher-moment
tests compare streamed and unevenly grouped data with direct sample statistics,
including nonzero skewness, large mean offsets, missing/constant data, variable
particle counts, irregular file timestamps, step-only input, ignored legacy times,
final statistics serialization, compact-output key sets, legacy-file conversion,
and failure cleanup. These tests
establish numerical contracts; they do not replace a DNS comparison or a grid/
filter/quadrature convergence study. For complete equation-balance tests, also
include the simulation's external forcing and enough snapshots to estimate time
derivatives. Such forcing is not part of the nine closure definitions here.
