"""Galaxy parameter distributions and scaling relations.

The tabulated values in this module are the values explored in
``../scaling.ipynb``. Functions accept scalars or array-like inputs and
return NumPy scalars/arrays with the corresponding broadcast shape.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
from astropy.cosmology import Planck18
from scipy.interpolate import LinearNDInterpolator
from scipy.optimize import brentq
from scipy.spatial import Delaunay


GalaxyType = Literal["early", "late"]
_REDSHIFT_BINS = np.arange(0.25, 2.76, 0.5)
_MASS_BINS = np.array([9.25, 9.75, 10.25, 10.75, 11.25])

_EARLY_MEDIAN = np.array([
	[0.27, 0.28, 0.38, 0.67, 0.76],
	[0.23, 0.21, 0.23, 0.45, 0.81],
	[np.nan, 0.18, 0.09, 0.30, 0.58],
	[np.nan, 0.22, 0.02, 0.19, 0.45],
	[np.nan, np.nan, -0.04, 0.08, 0.36],
	[np.nan, np.nan, np.nan, 0.10, 0.39],
])
_EARLY_P16 = np.array([
	[0.03, 0.04, 0.13, 0.42, 0.65],
	[-0.02, -0.14, 0.02, 0.26, 0.62],
	[np.nan, -0.15, -0.15, 0.07, 0.41],
	[np.nan, -0.02, -0.27, -0.04, 0.28],
	[np.nan, np.nan, -0.37, -0.2, 0.16],
	[np.nan, np.nan, np.nan, -0.22, 0.07],
])
_EARLY_P84 = np.array([
	[0.46, 0.46, 0.58, 0.92, 1.08],
	[0.43, 0.44, 0.42, 0.64, 0.97],
	[np.nan, 0.42, 0.36, 0.54, 0.81],
	[np.nan, 0.48, 0.35, 0.50, 0.74],
	[np.nan, np.nan, 0.36, 0.54, 0.55],
	[np.nan, np.nan, np.nan, 0.50, 0.68],
])
_LATE_MEDIAN = np.array([
	[0.49, 0.61, 0.66, 0.83, np.nan],
	[0.43, 0.56, 0.64, 0.75, 0.90],
	[0.37, 0.48, 0.57, 0.67, 0.82],
	[0.33, 0.42, 0.52, 0.61, 0.70],
	[np.nan, 0.35, 0.44, 0.53, 0.64],
	[np.nan, np.nan, 0.43, 0.47, 0.55],
])
_LATE_P16 = np.array([
	[0.24, 0.36, 0.42, 0.61, np.nan],
	[0.18, 0.32, 0.39, 0.51, 0.77],
	[0.11, 0.23, 0.33, 0.47, 0.62],
	[0.07, 0.16, 0.28, 0.35, 0.53],
	[np.nan, 0.10, 0.17, 0.26, 0.40],
	[np.nan, np.nan, 0.16, 0.19, 0.33],
])
_LATE_P84 = np.array([
	[0.70, 0.80, 0.85, 1.01, np.nan],
	[0.65, 0.76, 0.83, 0.90, 1.12],
	[0.60, 0.69, 0.77, 0.83, 0.96],
	[0.57, 0.65, 0.72, 0.80, 0.87],
	[np.nan, 0.57, 0.64, 0.70, 0.84],
	[np.nan, np.nan, 0.65, 0.71, 0.76],
])

_BOVERA_BINS = np.arange(0.025, 1.0, 0.05)
_SPIRAL_BOVERA = np.array([0, 0, 0.1, 0.65, 1.2, 1.2, 1.4, 1.4, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.25, 1.25, 1.15, 0.9, 0.5])
_ELLIPTICAL_BOVERA = np.array([0, 0, 0, 0.05, 0.25, 0.55, 0.75, 0.85, 1, 1.2, 1.3, 1.45, 1.6, 1.75, 1.8, 1.9, 1.75, 1.6, 1.3, 0.9])


def _log10(quantity):
	"""Return the base-10 logarithm of a quantity, preserving NaN values."""
	quantity = np.asarray(quantity, dtype=float)
	result = np.full(quantity.shape, np.nan, dtype=float)
	valid = quantity > 0
	result[valid] = np.log10(quantity[valid])
	return result.item() if result.ndim == 0 else result


def log10(value):
	"""Return the base-10 logarithm of a positive scalar or array-like value."""
	result = np.log10(np.asarray(value, dtype=float))
	return result.item() if result.ndim == 0 else result


def log10_kpc_to_arcsec(log10_kpc, redshift):
	"""Convert a physical size in ``log10(kpc)`` to arcseconds."""
	size_kpc, redshift = np.broadcast_arrays(10.0 ** np.asarray(log10_kpc, float), redshift)
	distance_kpc = Planck18.angular_diameter_distance(redshift).value * 1000.0
	result = size_kpc / distance_kpc * (180.0 / np.pi) * 3600.0
	return result.item() if result.ndim == 0 else result


def _make_size_interpolator(values):
	z_grid, mass_grid = np.meshgrid(_REDSHIFT_BINS, _MASS_BINS, indexing="ij")
	valid = np.isfinite(values)
	points = np.column_stack((z_grid[valid], mass_grid[valid]))
	triangulation = Delaunay(points)
	interpolator = LinearNDInterpolator(triangulation, values[valid])

	def evaluate(redshift, log_mass):
		redshift, log_mass = np.broadcast_arrays(redshift, log_mass)
		queries = np.column_stack((redshift.ravel(), log_mass.ravel()))
		inside = triangulation.find_simplex(queries) >= 0
		result = np.full(queries.shape[0], np.nan)
		result[inside] = interpolator(queries[inside])
		result = result.reshape(redshift.shape)
		return result.item() if result.ndim == 0 else result

	return evaluate


_SIZE_TABLES = {
	"early": (_EARLY_MEDIAN, _EARLY_P16, _EARLY_P84),
	"late": (_LATE_MEDIAN, _LATE_P16, _LATE_P84),
}
_SIZE_INTERPOLATORS = {
	galaxy_type: tuple(_make_size_interpolator(table) for table in tables)
	for galaxy_type, tables in _SIZE_TABLES.items()
}


def effective_radius_kpc(redshift, log_mass, galaxy_type: GalaxyType = "early", percentile: str = "median"):
	"""Return the effective radius in kpc from the notebook scaling tables."""
	if galaxy_type not in _SIZE_INTERPOLATORS:
		raise ValueError("galaxy_type must be 'early' or 'late'")
	percentiles = {"median": 0, "p16": 1, "p84": 2}
	if percentile not in percentiles:
		raise ValueError("percentile must be 'median', 'p16', or 'p84'")
	log_radius = _SIZE_INTERPOLATORS[galaxy_type][percentiles[percentile]](redshift, log_mass)
	return np.power(10.0, log_radius)


def sample_effective_radius_kpc(redshift, log_mass, galaxy_type: GalaxyType = "early", rng=None):
	"""Sample an effective radius using the tabulated log-size uncertainty."""
	if galaxy_type not in _SIZE_INTERPOLATORS:
		raise ValueError("galaxy_type must be 'early' or 'late'")
	median = _SIZE_INTERPOLATORS[galaxy_type][0](redshift, log_mass)
	p16 = _SIZE_INTERPOLATORS[galaxy_type][1](redshift, log_mass)
	p84 = _SIZE_INTERPOLATORS[galaxy_type][2](redshift, log_mass)
	standard_deviation = (p84 - p16) / 2.0
	generator = np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng
	return np.power(10.0, generator.normal(median, standard_deviation))


def bulge_effective_radius_kpc(stellar_mass, coefficient=1.836, slope=0.267):
	"""Return the Lange et al. bulge mass-size relation in kpc."""
	stellar_mass, coefficient, slope = np.broadcast_arrays(stellar_mass, coefficient, slope)
	if np.any(stellar_mass <= 0):
		raise ValueError("stellar_mass must be positive")
	result = coefficient * np.power(stellar_mass / 1.0e10, slope)
	return result.item() if result.ndim == 0 else result


def sample_bulge_effective_radius_kpc(stellar_mass, rng=None):
	"""Sample the Lange et al. relation with Gaussian coefficient and slope errors."""
	generator = np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng
	coefficient = generator.normal(1.836, 0.054, size=np.broadcast(np.asarray(stellar_mass)).shape)
	slope = generator.normal(0.267, 0.026, size=np.broadcast(np.asarray(stellar_mass)).shape)
	return bulge_effective_radius_kpc(stellar_mass, coefficient, slope)


_SMHM_PARAMETER_NAMES = (
	"EFF_0", "EFF_0_A", "EFF_0_A2", "EFF_0_Z", "M_1", "M_1_A", "M_1_A2", "M_1_Z",
	"ALPHA", "ALPHA_A", "ALPHA_A2", "ALPHA_Z", "BETA", "BETA_A", "BETA_Z", "DELTA",
	"GAMMA", "GAMMA_A", "GAMMA_Z", "CHI2",
)


def _smhm_parameters(path: str | Path | None = None) -> np.ndarray:
	path = Path(path) if path is not None else _default_data_path(
		"umachine-dr1/umachine-dr1/data/smhm/params/smhm_med_params.txt"
	)
	values = []
	for line in path.read_text().splitlines():
		parts = line.split()
		if parts and not parts[0].startswith("#"):
			values.append(float(parts[1]))
	return np.asarray(values, dtype=float)


@lru_cache(maxsize=4)
def _smhm_uncertainty_parameters(path: str | Path | None = None) -> np.ndarray:
	path = Path(path) if path is not None else _default_data_path(
		"umachine-dr1/umachine-dr1/data/smhm/params/uncertainties_smhm_med_params.txt"
	)
	rows = []
	for line in path.read_text().splitlines():
		parts = line.split()
		if not parts or parts[0].startswith("#"):
			continue
		values = np.asarray([float(value) for value in parts], dtype=float)
		if values.size >= 19:
			rows.append(values[:19])
	return np.asarray(rows, dtype=float)


def _evaluate_smhm(log_halo_mass, redshift, parameters):
	parameters = np.asarray(parameters)
	if parameters.shape[-1] < 19:
		raise ValueError("SMHM parameter set must contain at least 19 values")
	eff_0, eff_0_a, eff_0_a2, eff_0_z, m_1, m_1_a, m_1_a2, m_1_z = parameters[..., :8].T
	alpha, alpha_a, alpha_a2, alpha_z = parameters[..., 8:12].T
	beta, beta_a, beta_z = parameters[..., 12:15].T
	delta, gamma, gamma_a, gamma_z = parameters[..., 15:19].T
	log_halo_mass, redshift = np.broadcast_arrays(log_halo_mass, redshift)
	a = 1.0 / (1.0 + redshift)
	a1 = a - 1.0
	log_a = np.log(a)
	redshift_params = parameters[..., None] if parameters.ndim == 1 else parameters
	if parameters.ndim == 1:
		m_1 = m_1 + a1 * m_1_a - log_a * m_1_a2 + redshift * m_1_z
		sm_0 = m_1 + eff_0 + a1 * eff_0_a - log_a * eff_0_a2 + redshift * eff_0_z
		alpha = alpha + a1 * alpha_a - log_a * alpha_a2 + redshift * alpha_z
		beta = beta + a1 * beta_a + redshift * beta_z
		gamma = np.power(10.0, gamma + a1 * gamma_a + redshift * gamma_z)
		delta = np.asarray(delta)
		dm = log_halo_mass - m_1
		return sm_0 - np.log10(np.power(10.0, -alpha * dm) + np.power(10.0, -beta * dm)) + gamma * np.exp(-0.5 * np.square(dm / delta))
	# Parameter ensembles are evaluated against one broadcast halo/redshift pair.
	return np.stack([_evaluate_smhm(log_halo_mass, redshift, row) for row in parameters])


def smhm_log_stellar_mass(log_halo_mass, redshift, parameter_path: str | Path | None = None):
	"""Return median log10 stellar mass from the Behroozi et al. (2019) SMHM fit."""
	return _evaluate_smhm(log_halo_mass, redshift, _smhm_parameters(parameter_path))


def _invert_smhm(log_stellar_mass, redshift, parameters):
	log_stellar_mass, redshift = np.broadcast_arrays(log_stellar_mass, redshift)
	if parameters.ndim != 1:
		return np.stack([_invert_smhm(log_stellar_mass, redshift, row) for row in parameters])

	result = np.empty(log_stellar_mass.shape, dtype=float)
	for index in np.ndindex(log_stellar_mass.shape):
		stellar_mass = log_stellar_mass[index]
		z = redshift[index]
		lower = 10.5
		upper = 14.5 - 0.35 * z
		function = lambda halo_mass: _evaluate_smhm(halo_mass, z, parameters) - stellar_mass
		if function(lower) * function(upper) > 0:
			raise ValueError(
				f"log_stellar_mass={stellar_mass} is outside the SMHM range at redshift={z}"
			)
		result[index] = brentq(function, lower, upper)
	return result.item() if result.ndim == 0 else result


def smhm_log_halo_mass(log_stellar_mass, redshift, parameter_path: str | Path | None = None):
	"""Return log10 halo mass corresponding to a log10 stellar mass."""
	return _invert_smhm(log_stellar_mass, redshift, _smhm_parameters(parameter_path))


def sample_smhm_log_stellar_mass(log_halo_mass, redshift, rng=None, parameter_path=None, uncertainty_path=None):
	"""Sample log10 stellar mass from the published SMHM parameter uncertainties."""
	uncertainties = _smhm_uncertainty_parameters(uncertainty_path)
	if uncertainties.ndim != 2 or uncertainties.shape[1] < 19:
		raise ValueError("SMHM uncertainty file does not contain complete parameter samples")
	generator = np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng
	values = _evaluate_smhm(log_halo_mass, redshift, uncertainties)
	indices = generator.integers(
		uncertainties.shape[0],
		size=np.broadcast(np.asarray(log_halo_mass), np.asarray(redshift)).shape,
	)
	selected = np.take_along_axis(values, np.expand_dims(indices, axis=0), axis=0)
	return selected[0]


def sample_smhm_log_halo_mass(log_stellar_mass, redshift, rng=None, parameter_path=None, uncertainty_path=None):
	"""Sample log10 halo mass corresponding to a stellar mass uncertainty realization."""
	uncertainties = _smhm_uncertainty_parameters(uncertainty_path)
	if uncertainties.ndim != 2 or uncertainties.shape[1] < 19:
		raise ValueError("SMHM uncertainty file does not contain complete parameter samples")
	generator = np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng
	values = _invert_smhm(log_stellar_mass, redshift, uncertainties)
	indices = generator.integers(
		uncertainties.shape[0],
		size=np.broadcast(np.asarray(log_stellar_mass), np.asarray(redshift)).shape,
	)
	selected = np.take_along_axis(values, np.expand_dims(indices, axis=0), axis=0)
	return selected[0]


def sample_halo_ellipticity(size=None, rng=None):
	"""Sample halo ellipticity uniformly in the documented [0, 0.3] interval."""
	generator = np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng
	return generator.uniform(0.0, 0.3, size=size)


def sample_halo_galaxy_misalignment(size=None, rng=None):
	"""Sample halo-galaxy position-angle misalignment in degrees."""
	generator = np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng
	return generator.normal(0.0, 35.4, size=size)


def angular_size(redshift, log_mass, galaxy_type: GalaxyType = "early"):
	"""Return the median angular size in arcseconds."""
	if galaxy_type not in _SIZE_INTERPOLATORS:
		raise ValueError("galaxy_type must be 'early' or 'late'")
	median = log10_kpc_to_arcsec(_SIZE_INTERPOLATORS[galaxy_type][0](redshift, log_mass), redshift)
	return median


def sample_angular_size(redshift, log_mass, galaxy_type: GalaxyType = "early", rng=None):
	"""Sample angular size in arcseconds from the interpolated P16-P84 region."""
	if galaxy_type not in _SIZE_INTERPOLATORS:
		raise ValueError("galaxy_type must be 'early' or 'late'")
	median = _SIZE_INTERPOLATORS[galaxy_type][0](redshift, log_mass)
	p16 = _SIZE_INTERPOLATORS[galaxy_type][1](redshift, log_mass)
	p84 = _SIZE_INTERPOLATORS[galaxy_type][2](redshift, log_mass)
	standard_deviation = (p84 - p16) / 2.0
	generator = np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng
	sampled_log_radius = generator.normal(median, standard_deviation)
	return log10_kpc_to_arcsec(sampled_log_radius, redshift)


def axis_ratio_distribution(galaxy_type: str = "spiral") -> tuple[np.ndarray, np.ndarray]:
	"""Return b/a bin centres and normalized probabilities."""
	values = {"spiral": _SPIRAL_BOVERA, "spirals": _SPIRAL_BOVERA,
			  "elliptical": _ELLIPTICAL_BOVERA, "ellipticals": _ELLIPTICAL_BOVERA}.get(galaxy_type)
	if values is None:
		raise ValueError("galaxy_type must be 'spiral' or 'elliptical'")
	return _BOVERA_BINS.copy(), values / values.sum()


def sample_axis_ratio(size=None, galaxy_type: str = "spiral", rng=None):
	"""Sample axis ratios from the tabulated spiral/elliptical distribution."""
	bins, probabilities = axis_ratio_distribution(galaxy_type)
	generator = np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng
	return generator.choice(bins, size=size, p=probabilities)


def sample_ellipticity(size=None, galaxy_type: str = "spiral", rng=None):
	"""Sample ellipticity, defined by the notebook as ``e = 1 - b/a``."""
	return 1.0 - sample_axis_ratio(size=size, galaxy_type=galaxy_type, rng=rng)


def _sample_values(values, size=None, rng=None):
	if values.size == 0:
		raise ValueError("The empirical distribution contains no valid values")
	generator = np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng
	return generator.choice(values, size=size)


@lru_cache(maxsize=8)
def _load_structured_data(path: str):
	return np.genfromtxt(path, names=True, dtype=None, encoding=None, invalid_raise=False)


def _default_data_path(filename: str) -> Path:
	for root in (Path(__file__).resolve().parent.parent, Path(__file__).resolve().parents[2]):
		candidate = root / filename
		if candidate.exists():
			return candidate
	return Path(__file__).resolve().parent.parent / filename


def disk_to_total_distribution(data_path: str | Path | None = None) -> np.ndarray:
	"""Return valid Type 3 disk-to-total mass ratios from the notebook table."""
	path = Path(data_path) if data_path is not None else _default_data_path("apjs488156t6_mrt.txt")
	data = _load_structured_data(str(path))
	ratio = np.power(10.0, data["d_mass"] - data["mass"])
	total_mass = np.power(10.0, data["mass"])
	valid = (data["type"] == 3) & np.isfinite(ratio) & (ratio >= 0) & (ratio <= 1) & (total_mass > 1e8)
	return np.asarray(ratio[valid], dtype=float)


def sample_disk_to_total(size=None, data_path: str | Path | None = None, rng=None):
	"""Sample the Type 3 disk-to-total mass ratio distribution."""
	return _sample_values(disk_to_total_distribution(data_path), size=size, rng=rng)


def _table1_values(data_path: str | Path | None = None) -> tuple[np.ndarray, np.ndarray]:
	path = Path(data_path) if data_path is not None else _default_data_path("table1.dat")
	data = _load_structured_data(str(path))
	valid = (data["e"] != -99.99) & (data["Pps"] != -99.99) & (data["Pps"] < 0.32)
	ellipticity = np.asarray(data["e"][valid], dtype=float)
	pa_difference = (data["phid"][valid] - data["phib"][valid] + 180.0) % 360.0 - 180.0
	return ellipticity, np.asarray(pa_difference, dtype=float)


def bulge_ellipticity_distribution(data_path: str | Path | None = None) -> np.ndarray:
	"""Return the filtered bulge ellipticity values from ``table1.dat``."""
	return _table1_values(data_path)[0]


def sample_bulge_ellipticity(size=None, data_path: str | Path | None = None, rng=None):
	"""Sample the filtered bulge ellipticity distribution."""
	return _sample_values(bulge_ellipticity_distribution(data_path), size=size, rng=rng)


def bulge_disk_pa_difference_distribution(data_path: str | Path | None = None) -> np.ndarray:
	"""Return ``phid - phib`` in the notebook's [-180, 180) convention."""
	return _table1_values(data_path)[1]


def sample_bulge_disk_pa_difference(size=None, data_path: str | Path | None = None, rng=None):
	"""Sample the filtered bulge/disk position-angle difference distribution."""
	return _sample_values(bulge_disk_pa_difference_distribution(data_path), size=size, rng=rng)


def nfw_concentration(mass_vir, redshift, model: str = "diemer19"):
	"""Return ``cvir`` for ``Mvir`` using Colossus' concentration-mass relation.

	Masses are supplied in solar masses and converted internally to Colossus'
	``M_sun / h`` convention.
	"""
	try:
		from colossus.cosmology import cosmology
		from colossus.halo import concentration, mass_defs
	except ImportError as error:
		raise ImportError("nfw_concentration requires the optional 'colossus' package") from error
	cosmology.setCosmology("planck18")
	mass_vir, redshift = np.broadcast_arrays(mass_vir, redshift)
	mass_vir = mass_vir / cosmology.getCurrent().h
	result = np.empty(mass_vir.shape, dtype=float)
	for index in np.ndindex(mass_vir.shape):
		target_mass = mass_vir[index]
		z = redshift[index]

		def mass_difference(log_mass_200c):
			mass_200c = 10.0 ** log_mass_200c
			c200c = concentration.concentration(mass_200c, "200c", z, model=model)
			converted_mass, _, _ = mass_defs.changeMassDefinition(
				mass_200c, c200c, z, mdef_in="200c", mdef_out="vir"
			)
			return np.log10(converted_mass) - np.log10(target_mass)

		log_mass_200c = brentq(mass_difference, np.log10(target_mass) - 2.0, np.log10(target_mass) + 2.0)
		mass_200c = 10.0 ** log_mass_200c
		c200c = concentration.concentration(mass_200c, "200c", z, model=model)
		_, _, result[index] = mass_defs.changeMassDefinition(
			mass_200c, c200c, z, mdef_in="200c", mdef_out="vir"
		)
	return result.item() if result.ndim == 0 else result


def halo_concentration_mass(mass_vir, redshift, model: str = "diemer19"):
	"""Return halo concentration from the halo mass-redshift scaling relation."""
	return nfw_concentration(mass_vir, redshift, model)


def galaxy_effective_radius_mass(redshift, log_mass, galaxy_type: GalaxyType = "early", percentile: str = "median"):
	"""Return galaxy effective radius in kpc from the mass-redshift scaling relation."""
	return effective_radius_kpc(redshift, log_mass, galaxy_type, percentile)


def halo_to_stellar_mass(log_halo_mass, redshift):
	"""Return stellar mass in solar masses from the halo-to-stellar mass relation."""
	return np.power(10.0, smhm_log_stellar_mass(log_halo_mass, redshift))
