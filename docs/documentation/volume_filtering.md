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
# Edit the copied configuration to match the actual simulation and output times.
PYTHONPATH=toolchain python -m mfc.volume_filter /path/to/case/closures.json
```

The example's gas constants, mesh, indices, times, and filter widths are
illustrative, not inferred from a case. All paths are relative to the JSON file.
Use mutually consistent dimensional or nondimensional units. In particular,
`mu` is the dynamic viscosity used in tau = mu*S(u), **not MFC's Reynolds number
input**; `gamma` is the physical heat-capacity ratio, not MFC's stored 1/(gamma-1).
`gas_constant` is the specific gas constant, not the universal gas constant.
`grid.shape` is the actual cell count: `(m_glb+1,n_glb+1,p_glb+1)` for MFC.
`origin` is the lower domain face; the first cell center is origin + dx/2.
The grid in the JSON is an explicit assertion of the saved mesh: verify it
against the case/grid files. No coordinates or precision are guessed from bytes.

```bash
# Process a bounded number of new snapshots, then continue in another job.
PYTHONPATH=toolchain python -m mfc.volume_filter /path/to/case/closures.json --max-snapshots 10
PYTHONPATH=toolchain python -m mfc.volume_filter /path/to/case/closures.json --resume
```

Only consume completely written, immutable snapshots. Checkpoints are replaced
atomically after all filter widths and selected maps for a snapshot succeed.
Resume checks the complete configuration, format version, and processed input
paths/sizes/modification times. It does not re-count completed snapshots. Changing
the configuration (including its snapshot list) requires a new output path.
Keep the full planned snapshot list when using `--max-snapshots`. Only one job
may write a particular output path. Interrupted temporary files can be removed
once that writer is no longer running. Raw input files are never deleted.

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
All IDs 1..N must be present. Particle timestamps must match snapshot `time`.
`step` is the **saved file index**, which need not be the integration step when
adaptive output is used. Supply physical times, not step numbers as time weights.

The 20 fields are time, force(3), torque(3), velocity(3), angular_velocity(3),
angles(3), centroid(3), radius. The geometry and kinematics are used directly.
Forces and torques are retained in the input, but never substituted for distributed
surface traction. MFC's force history has stage-timing considerations; its
`patch_ib%%force` can also include collisions/body forces. Such values are not
unconditionally valid hydrodynamic force checks.

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
use the same mass denominator. Sample weights reveal missing map support.

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

`map.kind: "box"` samples the Cartesian product of supplied x/y/z offsets about
each particle, using periodic trilinear interpolation. This produces a spatial
map; it does not collapse the entire box to one number. Offsets are in fixed
physical length units, **not radius-normalized**, and axes remain laboratory axes.
All particles have equal weight, including when radii differ. Filters include the
entire suspension; a neighborhood overlapping another particle is intentional.
Offsets must remain inside half the domain size to avoid repeated periodic images.

An alternative is:

```json
{
  "kind": "axisymmetric",
  "axis": 0,
  "axial": [-0.1, 0.0, 0.1, 0.2],
  "radial": [0.0, 0.025, 0.05, 0.1],
  "azimuthal_points": 32
}
```

This averages uniformly spaced angular samples on each (axial,radial) ring,
then averages particles. **Components remain Cartesian**; this is not a
cylindrical-component tensor/vector transform. The angularly averaged Cartesian
radial components can cancel. A future cylindrical-component or slip-aligned
map should be a separate explicit choice. An incomplete ring is invalid rather
than averaging only its valid angles. The axis is x=0, y=1, z=2.

For each filter/component, the NPZ checkpoint contains:

- `sigma0.R_u_xx.time.mean/variance/weight/m2`: mean and population variance over
  instantaneous particle-average maps. Weight is the sum of valid time weights.
- `sigma0.R_u_xx.pooled.mean/variance/weight/m2`: population statistics over
  individual particle maps at all times, retaining within-snapshot particle
  scatter. Weight sums time weight * valid particle count.
- `metadata`: JSON containing the full configuration, filter widths, processed
  count, particle counts, and input fingerprints. The map axes are in this JSON.

All accumulators use float64 weighted Welford updates. No unbiased-variance or
independent-sample claim is made: particles and times can be correlated.
For axisymmetric maps, pooled variance measures variability of **ring averages**,
not unresolved azimuthal fluctuations. Sample support may vary; compare weights.
`time_weighting: "sample"` gives equal snapshot weights. `"trapezoid"` uses
physical-time trapezoidal weights over the configured interval, including half
intervals at its endpoints. Exclude startup transients in the snapshot list.
This supplies moments, not an estimate of temporal autocorrelation or confidence
intervals. `save_snapshot_maps: true` additionally writes one small NPZ per
snapshot containing particle mean, particle variance, counts, and time. Those
maps can support later stationarity/correlation studies without full DNS fields.

```python
import json
import numpy as np

with np.load("closure_statistics.npz", allow_pickle=False) as data:
    metadata = json.loads(str(data["metadata"]))
    mean = data["sigma0.R_u_xx.time.mean"]
    temporal_variance = data["sigma0.R_u_xx.time.variance"]
    pooled_variance = data["sigma0.R_u_xx.pooled.variance"]
```

## Validation

Run the focused suite without compiling MFC:

```bash
PYTHONPATH=toolchain python -m unittest mfc.volume_filter.test_volume_filter -v
```

It exercises Fourier-mode attenuation, filtered momentum and energy identities,
fluid-only differences, periodic registration, conservative deposition, analytic
surface-force/power integrals, map statistics, shared/sharded/Fortran-record input,
and equality of uninterrupted versus checkpoint/resume processing. These tests
establish numerical contracts; they do not replace a DNS comparison or a grid/
filter/quadrature convergence study. For complete equation-balance tests, also
include the simulation's external forcing and enough snapshots to estimate time
derivatives. Such forcing is not part of the nine closure definitions here.
