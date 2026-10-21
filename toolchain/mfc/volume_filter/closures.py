"""Instantaneous closures in Delgado's intrinsic volume-filtered convention."""

from time import perf_counter

import numpy as np
from numpy.polynomial.legendre import leggauss

from .numerics import Gaussian, LocalFluidFit, SpectralDerivatives, fluid_derivative, sphere_mask


class Surface:
    """Sphere quadrature with fluid-only, one-sided extrapolation of surface traces.

    Two trilinear samples at d and 2d give 2*f(d)-f(2d). Obstructed pairs
    reduce d, then use a local fluid-only affine fit if no pair is supported.
    This is not a reproduction of MFC's IB force stencil.
    """

    def __init__(self, grid, particles, fluid, config):
        self.grid, self.fluid = grid, fluid
        order = config.get("polar_points", 16)
        if not isinstance(order, int) or order < 2:
            raise ValueError("surface.polar_points must be an integer >= 2")
        z, w = leggauss(order)
        phi = 2 * np.pi * np.arange(2 * order) / (2 * order)
        zz, pp = np.meshgrid(z, phi, indexing="ij")
        normal = np.stack([np.sqrt(1 - zz**2) * np.cos(pp), np.sqrt(1 - zz**2) * np.sin(pp), zz], axis=-1).reshape(-1, 3)
        area = np.repeat(w, len(phi)) * (2 * np.pi / len(phi))
        points, normals, areas, velocities = [], [], [], []
        buried = 0
        for owner, particle in enumerate(particles):
            radial = particle[19] * normal
            surface_points = particle[16:19] + radial
            exposed = np.ones(len(normal), dtype=bool)
            for other, neighbor in enumerate(particles):
                if other == owner:
                    continue
                separation = grid.delta(particle[16:19], neighbor[16:19])
                if np.dot(separation, separation) >= (particle[19] + neighbor[19]) ** 2:
                    continue
                delta = grid.delta(surface_points, neighbor[16:19])
                exposed &= np.sum(delta**2, axis=1) >= neighbor[19] ** 2 * (1 - 32 * np.finfo(float).eps)
                # Identical periodic surfaces belong to the first body only.
                if other < owner and np.all(separation == 0) and particle[19] == neighbor[19]:
                    exposed[:] = False
            buried += int(np.count_nonzero(~exposed))
            points.append(surface_points[exposed])
            normals.append(normal[exposed])
            areas.append(particle[19] ** 2 * area[exposed])
            velocities.append((particle[7:10] + np.cross(particle[10:13], radial))[exposed])
        self.points = np.concatenate(points) if points else np.empty((0, 3))
        self.normals = np.concatenate(normals) if normals else np.empty((0, 3))
        self.areas = np.concatenate(areas) if areas else np.empty(0)
        self.velocity = np.concatenate(velocities) if velocities else np.empty((0, 3))
        d = config.get("offset_cells", 2.0) * max(grid.dx)
        if not np.isfinite(d) or d <= 0:
            raise ValueError("surface.offset_cells must be positive")
        self.near, self.far = self.points + d * self.normals, self.points + 2 * d * self.normals
        unit = np.broadcast_to(1.0, grid.shape)

        def supported(near, far):
            return np.isfinite(grid.sample(unit, near, fluid)) & np.isfinite(grid.sample(unit, far, fluid))

        valid = supported(self.near, self.far)
        original = valid.copy()
        for factor in (0.5, 0.25, 0.125, 0.0625):
            rows = np.flatnonzero(~valid)
            if not len(rows):
                break
            near = self.points[rows] + factor * d * self.normals[rows]
            far = self.points[rows] + 2 * factor * d * self.normals[rows]
            good = supported(near, far)
            self.near[rows[good]], self.far[rows[good]] = near[good], far[good]
            valid[rows[good]] = True
        self.unsupported = ~valid
        coordinates = (self.points[self.unsupported] - grid.origin) / grid.dx - 0.5
        self.fit = LocalFluidFit(fluid, coordinates) if len(coordinates) else None
        self.metadata = {
            "quadrature_points": len(self.points),
            "buried_points": buried,
            "reduced_offset_points": int(np.count_nonzero(valid & ~original)),
            "affine_fit_points": int(np.count_nonzero(self.unsupported)),
            "rank_deficient_fits": self.fit.rank_deficient if self.fit is not None else 0,
            "expanded_search_points": self.fit.expanded if self.fit is not None else 0,
        }
        self.force = np.zeros_like(self.points)
        self.heat = np.zeros(len(self.points))

    def trace(self, field):
        values = 2 * self.grid.sample(field, self.near, self.fluid) - self.grid.sample(field, self.far, self.fluid)
        if self.fit is not None:
            values[self.unsupported] = self.fit(field)
        if not np.all(np.isfinite(values)):
            raise ValueError("Nonfinite fluid values in surface reconstruction")
        return values

    def sources(self, gaussian):
        for i, axis in enumerate("xyz"):
            yield f"F_{axis}", gaussian(self.grid.deposit(self.points, self.areas * self.force[:, i]))
        yield "Q", gaussian(self.grid.deposit(self.points, self.areas * self.heat))
        power = np.einsum("ni,ni->n", self.force, self.velocity)
        yield "F_u", gaussian(self.grid.deposit(self.points, self.areas * power))


class State:
    """Primitive reconstruction for a calorically perfect single gas."""

    def __init__(self, fields, fluid, physics, report=None):
        self.fluid, self.physics = fluid, physics
        self.derivative_fallbacks = {}
        self.surface_reconstruction = None
        self.rho = next(fields)
        if not np.all(np.isfinite(self.rho), where=fluid) or np.any((self.rho <= 0) & fluid):
            raise ValueError("Nonpositive/nonfinite fluid density")
        self.rho[~fluid] = 1.0
        self.u = []
        for axis in "xyz":
            momentum = next(fields)
            if not np.all(np.isfinite(momentum), where=fluid):
                raise ValueError("Nonfinite fluid momentum")
            momentum[~fluid] = 0
            momentum /= self.rho
            self.u.append(momentum)
            if report is not None:
                report(f"velocity {axis} reconstructed")
        energy = next(fields)
        if not np.all(np.isfinite(energy), where=fluid):
            raise ValueError("Nonfinite fluid total energy")
        energy[~fluid] = 0
        if report is not None:
            report("reconstructing pressure and temperature")
        self.speed2 = np.empty_like(self.rho)
        np.multiply(self.u[0], self.u[0], out=self.speed2)
        for velocity in self.u[1:]:
            self.speed2 += velocity * velocity
        # Reuse the energy allocation for pressure; bound arithmetic scratch
        # space to slabs instead of allocating several extra full-domain arrays.
        for first in range(0, len(self.rho), 8):
            slab = slice(first, first + 8)
            energy[slab] -= 0.5 * self.rho[slab] * self.speed2[slab]
            energy[slab] *= physics["gamma"] - 1
        self.p = energy
        if not np.all(np.isfinite(self.p), where=fluid) or np.any((self.p <= 0) & fluid):
            raise ValueError("Nonpositive/nonfinite fluid pressure; check EOS and conservative indices")
        self.p[~fluid] = 0
        self.temperature = np.empty_like(self.p)
        for first in range(0, len(self.rho), 8):
            slab = slice(first, first + 8)
            self.temperature[slab] = self.p[slab] / (self.rho[slab] * physics["gas_constant"])
        if report is not None:
            report("fluid primitive state ready")

    def derivative(self, field, grid, axis):
        return fluid_derivative(field, self.fluid, axis, grid.dx[axis], self.derivative_fallbacks)

    def reconstruction_metadata(self):
        return {
            "derivatives": {"xyz"[axis]: {name: value for name, value in info.items() if name != "fit"} for axis, info in sorted(self.derivative_fallbacks.items())},
            "surface": self.surface_reconstruction,
        }


def closure_fields(grid, state, particles, sigma, config):
    """Yield (name, scalar field) so maps can be accumulated before advancing.

    R_* are intrinsic. F, Q, F_u are already source densities. Each intrinsic
    field also has an alpha-weighted companion; undefined intrinsic values are NaN.
    """
    physics = state.physics
    mu, conductivity = physics["mu"], physics["conductivity"]
    gaussian = Gaussian(grid, sigma, config.get("fft_workers", 1))
    fluid = state.fluid

    def filt(field):
        return gaussian(np.where(fluid, field, 0))

    alpha = gaussian(fluid.astype(float))
    mass = filt(state.rho)
    floor = config.get("alpha_min", 1e-6)
    valid = (alpha >= floor) & (mass > 0)
    denominator = np.where(valid, mass, 1)
    momentum = [filt(state.rho * u) for u in state.u]
    ubar = [m / denominator for m in momentum]
    pbar_volume = filt(state.p)
    temperature = pbar_volume / (physics["gas_constant"] * denominator)
    yield "alpha", alpha
    yield "rho_bar", np.where(valid, mass / np.where(valid, alpha, 1), np.nan)
    for axis, value in zip("xyz", ubar):
        yield f"u_tilde_{axis}", np.where(valid, value, np.nan)
    yield "T_tilde", np.where(valid, temperature, np.nan)

    def intrinsic(name, weighted):
        yield name, np.where(valid, weighted / np.where(valid, alpha, 1), np.nan)
        yield f"alpha_{name}", np.where(valid, weighted, np.nan)

    # Derivatives of ratios use smooth filtered numerators, avoiding NaN stencils.
    derivatives = SpectralDerivatives(gaussian)

    def velocity_gradient(i, j):
        return (derivatives.derivative(momentum[i], j) - ubar[i] * derivatives.derivative(mass, j)) / denominator

    raw_div = sum(state.derivative(state.u[i], grid, i) for i in range(3)) if mu else None
    resolved_div = sum(velocity_gradient(i, i) for i in range(3)) if mu else None
    surface = Surface(grid, particles, fluid, config["surface"])
    state.surface_reconstruction = surface.metadata
    surface.force = surface.trace(state.p)[:, None] * surface.normals
    if config["surface"].get("velocity_bc", "no_slip") == "slip":
        trace_u = np.column_stack([surface.trace(u) for u in state.u])
        normal_difference = np.einsum("ni,ni->n", surface.velocity - trace_u, surface.normals)
        surface.velocity = trace_u + normal_difference[:, None] * surface.normals

    raw_work = [np.zeros(grid.shape) for _ in range(3)] if mu else None
    resolved_work = [np.zeros(grid.shape) for _ in range(3)] if mu else None
    for i in range(3):
        for j in range(i, 3):
            suffix = "xyz"[i] + "xyz"[j]
            weighted = filt(state.rho * state.u[i] * state.u[j]) - momentum[i] * ubar[j]
            yield from intrinsic(f"R_u_{suffix}", weighted)
            del weighted
            if mu:
                tau = mu * (state.derivative(state.u[i], grid, j) + state.derivative(state.u[j], grid, i))
                gradient = velocity_gradient(i, j)
                resolved_tau = mu * (gradient + (gradient if i == j else velocity_gradient(j, i)))
                del gradient
                if i == j:
                    tau -= (2 * mu / 3) * raw_div
                    resolved_tau -= (2 * mu / 3) * resolved_div
                filtered_tau = filt(tau)
                yield from intrinsic(f"R_mu_{suffix}", filtered_tau - alpha * resolved_tau)
                traction = surface.trace(tau)
                surface.force[:, i] -= traction * surface.normals[:, j]
                raw_work[j] += state.u[i] * tau
                resolved_work[j] += ubar[i] * filtered_tau
                if i != j:
                    surface.force[:, j] -= traction * surface.normals[:, i]
                    raw_work[i] += state.u[j] * tau
                    resolved_work[i] += ubar[j] * filtered_tau
                del tau, resolved_tau, filtered_tau
            else:
                yield from intrinsic(f"R_mu_{suffix}", np.zeros(grid.shape))
    del raw_div, resolved_div

    speed2_volume = filt(state.rho * state.speed2)
    gamma_factor = physics["gamma"] / (physics["gamma"] - 1)
    for i, axis in enumerate("xyz"):
        yield from intrinsic(f"R_uu_{axis}", 0.5 * (filt(state.rho * state.speed2 * state.u[i]) - speed2_volume * ubar[i]))
        yield from intrinsic(f"R_Tu_{axis}", gamma_factor * (filt(state.p * state.u[i]) - pbar_volume * ubar[i]))
        if mu:
            yield from intrinsic(f"R_tau_u_{axis}", filt(raw_work[i]) - resolved_work[i])
            raw_work[i] = resolved_work[i] = None
        else:
            yield from intrinsic(f"R_tau_u_{axis}", np.zeros(grid.shape))
        if conductivity:
            heat_flux = -conductivity * state.derivative(state.temperature, grid, i)
            grad_temperature = (derivatives.derivative(pbar_volume, i) / physics["gas_constant"] - temperature * derivatives.derivative(mass, i)) / denominator
            yield from intrinsic(f"R_q_{axis}", filt(heat_flux) + alpha * conductivity * grad_temperature)
            if config["surface"]["thermal_bc"] == "reconstruct":
                surface.heat += surface.trace(heat_flux) * surface.normals[:, i]
        else:
            yield from intrinsic(f"R_q_{axis}", np.zeros(grid.shape))
    derivatives.clear()
    yield from surface.sources(gaussian)


def reconstruct(grid, fields, particles, physics, report=None):
    start = perf_counter()
    if report is not None:
        report("constructing periodic fluid/solid mask")
    fluid = sphere_mask(grid, particles)
    if report is not None:
        report(f"fluid/solid mask constructed in {perf_counter() - start:.2f}s")
    return State(iter(fields), fluid, physics, report)
