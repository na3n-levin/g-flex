from __future__ import annotations

# The copied glafic core is intentionally kept as top-level modules for compatibility.
# pyright: reportMissingImports=false

import json
import math
import sys
from functools import lru_cache
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import Planck18
from astropy.io import fits
from scipy.signal import fftconvolve

sys.path.insert(0, str(Path(__file__).resolve().with_name("core")))
from parser import parse_input_file

# Euclid VIS-like detector assumptions used for the grayscale I-band product.
A_EFF_CM2 = 1.006e4
PIVOT_ANGSTROM = 7200.0
BANDWIDTH_ANGSTROM = 3000.0
EXPOSURE_SECONDS = 565.0 * 4.0 + 100.0 * 2.0
READ_NOISE_ELECTRONS = 4.4
SKY_ABMAG_PER_ARCSEC2 = 21.0
DETECTOR_PIXEL_ARCSEC = 0.1
SIMULATION_BOUNDS = (-5.0, 5.0, -5.0, 5.0)
TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
TEMPLATE_FILES = {
    "elliptical": TEMPLATE_DIR / "CWW_E_ext.sed",
    "spiral": TEMPLATE_DIR / "CWW_Sbc_ext.sed",
    "late_spiral": TEMPLATE_DIR / "CWW_Scd_ext.sed",
    "irregular": TEMPLATE_DIR / "CWW_Im_ext.sed",
    "starburst": TEMPLATE_DIR / "KIN_SB1_ext.sed",
    "dusty_starburst": TEMPLATE_DIR / "KIN_SB2_ext.sed",
    "agn": TEMPLATE_DIR / "j0647agn_torus.fits",
}
TEMPLATE_ALIASES = {
    "old": "elliptical",
    "young": "starburst",
    "dusty": "dusty_starburst",
    "quiescent": "elliptical",
}
SUPPORTED_TEMPLATES = set(TEMPLATE_FILES) | set(TEMPLATE_ALIASES)
VIS_FILTER_CANDIDATES = (
    Path(__file__).resolve().parent / "filters" / "Euclid_VIS.vis.dat",
    Path(__file__).resolve().parent.parent / "forward_lensing_sim" / "filters" / "Euclid_VIS.vis.dat",
)


def _resolve(run_dir: Path, stem: str) -> Path:
    for candidate in (run_dir / f"_{stem}.fits", run_dir / f"{stem}.fits"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing {stem}.fits in {run_dir}")


def _component_maps(filename: Path) -> list[np.ndarray]:
    with fits.open(filename, memmap=True) as hdul:
        arrays = [np.asarray(hdu.data, dtype=float) for hdu in hdul if hdu.data is not None]
    if not arrays:
        raise ValueError(f"No image data found in {filename}")
    if len(arrays) == 1 and arrays[0].ndim == 3:
        cube = arrays[0]
        return [np.array(cube[index], dtype=float, copy=True) for index in range(max(1, cube.shape[0] - 1))]
    if len(arrays) > 1:
        arrays = arrays[:-1]
    return [np.array(array, dtype=float, copy=True) for array in arrays]


def _moffat_kernel(fwhm_pixels: float, beta: float = 3.0) -> np.ndarray:
    size = int(math.ceil(6.0 * fwhm_pixels))
    if size % 2 == 0:
        size += 1
    radius = size // 2
    y, x = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    alpha = fwhm_pixels / (2.0 * math.sqrt(2.0 ** (1.0 / beta) - 1.0))
    kernel = (beta - 1.0) / (math.pi * alpha**2) * (1.0 + (x * x + y * y) / alpha**2) ** (-beta)
    return kernel / kernel.sum()


def _bin(image: np.ndarray, source_pixel: float, target_pixel: float) -> np.ndarray:
    factor = int(round(target_pixel / source_pixel))
    if factor < 1 or not np.isclose(target_pixel / source_pixel, factor):
        raise ValueError("Simulation pixel size must divide the detector pixel size")
    height = image.shape[0] - image.shape[0] % factor
    width = image.shape[1] - image.shape[1] % factor
    if height != image.shape[0] or width != image.shape[1]:
        raise ValueError("Image shape must be divisible by the detector bin factor")
    trimmed = image[:height, :width]
    return trimmed.reshape(height // factor, factor, width // factor, factor).sum(axis=(1, 3))


def _vis_filter() -> tuple[np.ndarray, np.ndarray]:
    filter_path = next((path for path in VIS_FILTER_CANDIDATES if path.exists()), None)
    if filter_path is None:
        raise FileNotFoundError("Could not find the original Euclid VIS filter curve")
    wavelengths, transmission = np.loadtxt(filter_path, unpack=True)
    return np.asarray(wavelengths, dtype=float), np.asarray(transmission, dtype=float)


@lru_cache(maxsize=None)
def _load_template(template: str) -> tuple[np.ndarray, np.ndarray]:
    canonical_template = TEMPLATE_ALIASES.get(template, template)
    if canonical_template not in TEMPLATE_FILES:
        raise ValueError(f"Unsupported spectral template '{template}'")
    filename = TEMPLATE_FILES[canonical_template]
    if filename.suffix == ".fits":
        with fits.open(filename, memmap=True) as hdul:
            data = hdul[1].data
            template_wavelengths = np.asarray(data["wave"], dtype=float)
            template_flux = np.asarray(data["flux"], dtype=float)
    else:
        template_wavelengths, template_flux = np.loadtxt(filename, unpack=True)
    valid = np.isfinite(template_wavelengths) & np.isfinite(template_flux) & (template_flux >= 0.0)
    template_wavelengths = np.asarray(template_wavelengths[valid], dtype=float)
    template_flux = np.asarray(template_flux[valid], dtype=float)
    order = np.argsort(template_wavelengths)
    return template_wavelengths[order], template_flux[order]


def _template_spectrum(template: str, wavelengths: np.ndarray, redshift: float = 0.0) -> np.ndarray:
    if not np.isfinite(redshift) or redshift < 0.0:
        raise ValueError("Redshift must be finite and non-negative")
    template_wavelengths, template_flux = _load_template(template)
    return np.interp(wavelengths / (1.0 + redshift), template_wavelengths, template_flux, left=0.0, right=0.0)


def template_curve(template: str, redshift: float = 0.0, samples: int = 600) -> tuple[np.ndarray, np.ndarray]:
    if not np.isfinite(redshift) or redshift < 0.0:
        raise ValueError("Redshift must be finite and non-negative")
    template_wavelengths, template_flux = _load_template(template)
    sample_indices = np.linspace(0, len(template_wavelengths) - 1, min(samples, len(template_wavelengths)), dtype=int)
    return template_wavelengths[sample_indices] * (1.0 + redshift), template_flux[sample_indices]


def _template_gain(template: str, wavelengths: np.ndarray, transmission: np.ndarray, redshift: float = 0.0) -> float:
    spectrum = _template_spectrum(template, wavelengths, redshift)
    weighted = spectrum * transmission
    band_flux = np.trapezoid(weighted, wavelengths)
    if not np.isfinite(band_flux) or band_flux <= 0.0:
        raise ValueError(f"Spectral template '{template}' has no VIS throughput")
    mean_photon_wavelength = np.trapezoid(weighted * wavelengths, wavelengths) / band_flux
    h_cgs = 6.62607015e-27
    c_cgs = 2.99792458e10
    return mean_photon_wavelength * 1.0e-8 * A_EFF_CM2 / (h_cgs * c_cgs)


def _sky_flux_per_pixel(abmag: float, wavelengths: np.ndarray, transmission: np.ndarray) -> float:
    c_cm_s = 2.99792458e10
    wavelength_cm = wavelengths * 1.0e-8
    f_nu = 10.0 ** ((-abmag - 48.6) / 2.5)
    f_lambda = f_nu * c_cm_s / wavelength_cm**2 * 1.0e-8
    return float(np.trapezoid(f_lambda * transmission, wavelengths) * DETECTOR_PIXEL_ARCSEC**2)


def _add_detector_noise(signal: np.ndarray, gain: np.ndarray, seed: int, wavelengths: np.ndarray, transmission: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sky_flux = _sky_flux_per_pixel(SKY_ABMAG_PER_ARCSEC2, wavelengths, transmission)
    rate_image = np.clip(signal + sky_flux, 0.0, None)
    total_electrons = np.zeros_like(signal, dtype=float)
    total_exposure = 0.0
    for exposure_time, exposure_count in ((565.0, 4), (100.0, 2)):
        expected = rate_image * gain * exposure_time * exposure_count
        total_electrons += rng.poisson(np.clip(expected, 0.0, None))
        total_electrons += rng.normal(0.0, READ_NOISE_ELECTRONS * math.sqrt(exposure_count), size=signal.shape)
        total_exposure += exposure_time * exposure_count
    return np.divide(total_electrons, gain * total_exposure, out=np.zeros_like(total_electrons), where=gain > 0.0)


def _stretch(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=float)
    background = float(np.median(image))
    signal = np.clip(image - background, 0.0, None)
    positive = signal[signal > 0.0]
    scale = float(np.percentile(positive, 99.7)) if positive.size else 1.0
    scale = max(scale, np.finfo(float).tiny)
    return np.clip(np.arcsinh(signal / scale * 12.0) / np.arcsinh(12.0), 0.0, 1.0)


def _source_flux_map(image_map: np.ndarray, source_map: np.ndarray, luminosity: float, redshift: float) -> np.ndarray:
    total_weight = float(np.sum(np.clip(source_map, 0.0, None)))
    if total_weight <= 0.0:
        raise ValueError("Source FITS map has no positive flux to normalize")
    luminosity_distance_cm = Planck18.luminosity_distance(redshift).to_value("cm")
    total_flux = float(luminosity) / (4.0 * math.pi * luminosity_distance_cm**2)
    return np.clip(image_map, 0.0, None) / total_weight * total_flux


def _lens_flux_components(run, shape: tuple[int, int], pixel_size: float, metadata: dict) -> list[tuple[np.ndarray, str, float]]:
    components: list[tuple[np.ndarray, str, float]] = []
    x_coords = (np.arange(shape[1]) - (shape[1] - 1) / 2.0) * pixel_size
    y_coords = (np.arange(shape[0]) - (shape[0] - 1) / 2.0) * pixel_size
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)
    solar_luminosity = 3.828e33
    lens_metadata = metadata.get("lenses", [])
    for lens_index, lens in enumerate(run.lenses):
        if str(lens.kind).lower() != "sers" or len(lens.parameters) < 8:
            continue
        redshift, mass, x0, y0, ellipticity, position_angle, radius, index = [float(value) for value in lens.parameters[:8]]
        if lens_index >= len(lens_metadata):
            raise ValueError(f"Missing photometric metadata for lens component {lens_index + 1}")
        web_lens = lens_metadata[lens_index]
        if "massToLight" not in web_lens or not str(web_lens["massToLight"]).strip():
            raise ValueError(f"Missing mass-to-light ratio for luminous lens component {lens_index + 1}")
        mass_to_light = float(web_lens["massToLight"])
        if not np.isfinite(mass_to_light) or mass_to_light <= 0.0:
            raise ValueError(f"Mass-to-light ratio must be positive for luminous lens component {lens_index + 1}")
        distance_cm = Planck18.luminosity_distance(redshift).to_value("cm")
        total_flux = mass / mass_to_light * solar_luminosity / (4.0 * math.pi * distance_cm**2)
        axis_ratio = max(1.0 - ellipticity, 1.0e-3)
        angle = np.radians(position_angle)
        dx, dy = x_grid - x0, y_grid - y0
        xr = dx * np.cos(angle) + dy * np.sin(angle)
        yr = -dx * np.sin(angle) + dy * np.cos(angle)
        radius_grid = np.sqrt(xr * xr + (yr / axis_ratio) ** 2) + 1.0e-9
        b_n = 2.0 * index - 1.0 / 3.0 + 4.0 / (405.0 * index)
        profile = np.exp(-b_n * (radius_grid / radius) ** (1.0 / index))
        normalization = 2.0 * math.pi * axis_ratio * radius**2 * index * math.gamma(2.0 * index) / b_n ** (2.0 * index)
        components.append((total_flux * profile / normalization * pixel_size**2, str(web_lens["template"]), redshift))
    return components


def render_i_band(run_dir: str | Path, seed: int = 42, metadata: dict | None = None) -> Path:
    run_dir = Path(run_dir)
    if metadata is None:
        metadata_path = run_dir / "web_metadata.json"
        if not metadata_path.exists():
            raise ValueError(f"Missing photometric metadata: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    parsed = parse_input_file(run_dir / "run.input", resolve_only=True)
    run = parsed.template if hasattr(parsed, "template") and parsed.template is not None else parsed
    wavelengths, transmission = _vis_filter()
    image_maps = _component_maps(_resolve(run_dir, "image"))
    source_maps = _component_maps(_resolve(run_dir, "source"))
    flux_components: list[tuple[np.ndarray, str, float]] = []
    source_metadata = metadata.get("sources", [])
    extended_index = 0
    for source_index, source in enumerate(run.sources):
        if str(source.kind).lower() == "point":
            continue
        if extended_index >= len(image_maps) or extended_index >= len(source_maps):
            raise ValueError(f"Missing FITS plane for extended source component {source_index + 1}")
        image_map = image_maps[extended_index]
        source_map = source_maps[extended_index]
        if source_index >= len(source_metadata) or "luminosity" not in source_metadata[source_index]:
            raise ValueError(f"Missing luminosity for extended source component {source_index + 1}")
        luminosity = float(source_metadata[source_index]["luminosity"])
        if not np.isfinite(luminosity) or luminosity <= 0.0:
            raise ValueError(f"Luminosity must be positive for extended source component {source_index + 1}")
        template = str(source_metadata[source_index]["template"])
        if template not in SUPPORTED_TEMPLATES:
            raise ValueError(f"Unsupported spectral template '{template}' for source component {source_index + 1}")
        flux_components.append((_source_flux_map(image_map, source_map, luminosity, float(source.redshift)), template, float(source.redshift)))
        extended_index += 1
    pixel_size = float(run.cfg.get("pix_ext", 0.02))
    shape = image_maps[0].shape
    flux_components.extend(_lens_flux_components(run, shape, pixel_size, metadata))
    if not flux_components:
        raise ValueError("No bright source or lens components are available for I-band rendering")
    kernel = _moffat_kernel(0.18 / pixel_size)
    detector_components = [
        _bin(fftconvolve(component_map, kernel, mode="same"), pixel_size, DETECTOR_PIXEL_ARCSEC)
        for component_map, _, _ in flux_components
    ]
    detector_flux = np.sum(detector_components, axis=0)
    detector_gain_numerator = np.zeros_like(detector_flux, dtype=float)
    for detector_component, (_, template, redshift) in zip(detector_components, flux_components):
        detector_gain_numerator += detector_component * _template_gain(template, wavelengths, transmission, redshift)
    detector_gain = np.divide(detector_gain_numerator, detector_flux, out=np.zeros_like(detector_flux), where=detector_flux > 0.0)
    noisy = _add_detector_noise(detector_flux, detector_gain, seed, wavelengths, transmission)
    view = metadata.get("view", {})
    view_xmin = float(view.get("xmin", SIMULATION_BOUNDS[0]))
    view_xmax = float(view.get("xmax", SIMULATION_BOUNDS[1]))
    view_ymin = float(view.get("ymin", SIMULATION_BOUNDS[2]))
    view_ymax = float(view.get("ymax", SIMULATION_BOUNDS[3]))
    if not (SIMULATION_BOUNDS[0] <= view_xmin < view_xmax <= SIMULATION_BOUNDS[1] and SIMULATION_BOUNDS[2] <= view_ymin < view_ymax <= SIMULATION_BOUNDS[3]):
        raise ValueError("Requested view bounds must lie inside the fixed simulation frame")
    x0 = int(round((view_xmin - SIMULATION_BOUNDS[0]) / DETECTOR_PIXEL_ARCSEC))
    x1 = int(round((view_xmax - SIMULATION_BOUNDS[0]) / DETECTOR_PIXEL_ARCSEC))
    y0 = int(round((view_ymin - SIMULATION_BOUNDS[2]) / DETECTOR_PIXEL_ARCSEC))
    y1 = int(round((view_ymax - SIMULATION_BOUNDS[2]) / DETECTOR_PIXEL_ARCSEC))
    if y1 > noisy.shape[0] or x1 > noisy.shape[1] or x1 <= x0 or y1 <= y0:
        raise ValueError(f"Requested view {view_xmin}:{view_xmax}, {view_ymin}:{view_ymax} exceeds detector image {noisy.shape}")
    # FITS rows increase toward +y; flip for a screen image with +y upward.
    cropped = np.flipud(noisy[y0:y1, x0:x1])
    height, width = cropped.shape
    background = float(np.median(cropped))
    signal_values = np.clip(cropped - background, 0.0, None)
    positive = signal_values[signal_values > 0.0]
    scale = float(np.percentile(positive, 99.7)) if positive.size else 1.0
    scale = max(scale, np.finfo(float).tiny)
    normalized_values = signal_values / scale
    display = np.clip(np.arcsinh(normalized_values * 12.0) / np.arcsinh(12.0), 0.0, 1.0)
    output = run_dir / "ie_band.png"
    if output.exists():
        output.unlink()
    plt.imsave(output, display, cmap="gray", vmin=0.0, vmax=1.0)
    data_path = run_dir / "ie_band_data.json"
    if data_path.exists():
        data_path.unlink()
    data_path.write_text(json.dumps({"width": width, "height": height, "values": normalized_values.tolist()}), encoding="utf-8")
    return output
