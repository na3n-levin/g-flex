from __future__ import annotations

import json
import hashlib
import math
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np
from astropy.io import fits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates

ROOT = Path(__file__).resolve().parent
CORE = ROOT / "core"
STATIC = ROOT / "static"
OUT = ROOT / "runs"
PRESETS = ROOT / "presets"
CURVE_CACHE = OUT / ".curve_cache"
sys.path.insert(0, str(CORE))

from executor import execute_runs  # noqa: E402
from image_pipeline import SIMULATION_BOUNDS, _vis_filter, render_i_band, template_curve  # noqa: E402
from parser import parse_input_file  # noqa: E402
from ui import ConsoleReporter, LogLevel  # noqa: E402

SAFE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
SAFE_FUNCTION = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*$")

LENS_MODELS = [
    {"id": "sers", "label": "Sersic", "help": "elliptical mass profile", "count": 8},
    {"id": "anfw", "label": "Approximate NFW", "help": "fast dark matter halo", "count": 8},
    {"id": "gaupot", "label": "Gaussian lens potential", "help": "elliptical Gaussian potential", "count": 8},
    {"id": "sie", "label": "SIE", "help": "isothermal elliptical density", "count": 8},
    {"id": "pow", "label": "Elliptical power law", "help": "power-law elliptical density", "count": 8},
    {"id": "point", "label": "Point mass", "help": "dark compact lens", "count": 4},
]
SOURCE_MODELS = [
    {"id": "sersic", "label": "Sersic light", "help": "extended galaxy light", "count": 7},
    {"id": "gauss", "label": "Gaussian light", "help": "soft extended light", "count": 7},
]
FUNCTIONS = [
    {"id": "parameters.log10", "label": "Base-10 logarithm", "help": "Calculate log10 of a positive value", "args": ["value"], "argHelp": ["positive numeric value"]},
    {"id": "parameters.halo_concentration_mass", "label": "Halo concentration c_vir", "help": "NFW concentration from halo mass and redshift", "args": ["halo_mass", "redshift"], "argHelp": ["virial halo mass (solar masses)", "lens redshift"]},
    {"id": "parameters.galaxy_effective_radius_mass", "label": "Galaxy effective radius R_e", "help": "Galaxy size from redshift, stellar mass, and galaxy type", "args": ["redshift", "log10_stellar_mass", "galaxy_type"], "argHelp": ["galaxy redshift", "log10 stellar mass (solar masses)", "early or late"]},
    {"id": "parameters.halo_to_stellar_mass", "label": "Halo mass to stellar mass", "help": "Stellar mass predicted from halo mass", "args": ["log10_halo_mass", "redshift"], "argHelp": ["log10 halo mass (solar masses)", "galaxy redshift"]},
    {"id": "parameters.log10_kpc_to_arcsec", "label": "Physical size to angular size", "help": "Convert log10 physical size to arcseconds", "args": ["log10_kpc", "redshift"], "argHelp": ["log10 size (kpc)", "object redshift"]},
]
SPECTRAL_TEMPLATES = [
    {"id": "elliptical", "label": "Elliptical (CWW E)", "help": "empirical red galaxy spectrum"},
    {"id": "spiral", "label": "Spiral (CWW Sbc)", "help": "empirical spiral galaxy spectrum"},
    {"id": "late_spiral", "label": "Late spiral (CWW Scd)", "help": "empirical late-type galaxy spectrum"},
    {"id": "irregular", "label": "Irregular (CWW Im)", "help": "empirical irregular galaxy spectrum"},
    {"id": "starburst", "label": "Starburst (KIN SB1)", "help": "empirical starburst spectrum"},
    {"id": "dusty_starburst", "label": "Dusty starburst (KIN SB2)", "help": "empirical dusty starburst spectrum"},
    {"id": "agn", "label": "AGN (J0647 + torus)", "help": "observed AGN and torus spectrum"},
]
SUPPORTED_LENS_KINDS = {"sers", "anfw", "gaupot", "sie", "pow", "point"}
SUPPORTED_SOURCE_KINDS = {"sersic", "gauss"}
LUMINOUS_LENS_KINDS = {"sers"}
LEGACY_LENS_ALIASES = {"gauss": "gaupot"}
curve_processes: set[subprocess.Popen] = set()
curve_process_lock = threading.Lock()


def catalog() -> dict:
    return {"lensModels": LENS_MODELS, "sourceModels": SOURCE_MODELS, "functions": FUNCTIONS, "spectralTemplates": SPECTRAL_TEMPLATES}


def preset_path(preset_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", preset_id):
        raise ValueError("Invalid preset id")
    target = (PRESETS / f"{preset_id}.json").resolve()
    if PRESETS.resolve() not in target.parents:
        raise ValueError("Invalid preset path")
    return target


def list_presets() -> list[dict]:
    PRESETS.mkdir(exist_ok=True)
    presets = []
    for path in PRESETS.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            presets.append({"id": path.stem, "name": data["name"], "data": data["data"], "created": data.get("created", 0)})
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return sorted(presets, key=lambda item: item.get("created", 0), reverse=True)


def number(value: object, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"Missing numeric value for {field_name}")
    try:
        numeric = float(text)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid numeric value for {field_name}: {value!r}") from error
    if not math.isfinite(numeric):
        raise ValueError(f"Non-finite numeric value for {field_name}: {value!r}")
    return format(numeric, ".12g")


def parameter_token(value: object, field_name: str) -> str:
    text = str(value).strip()
    if not text or any(character.isspace() for character in text) or "\n" in text or "\r" in text:
        raise ValueError(f"Invalid value for {field_name}: {value!r}")
    return text


def mass_parameter_token(value: object, hubble: str, field_name: str) -> str:
    text = parameter_token(value, field_name)
    try:
        return number(float(text) * float(hubble), field_name)
    except ValueError:
        return f"({text})*{hubble}"


def normalize_lens_kind(value: object) -> str:
    kind = str(value).strip().lower()
    return LEGACY_LENS_ALIASES.get(kind, kind)


def view_bounds(config: dict) -> dict[str, float]:
    values = {name: float(number(config.get(name), name)) for name in ("xmin", "xmax", "ymin", "ymax")}
    if not (SIMULATION_BOUNDS[0] <= values["xmin"] < values["xmax"] <= SIMULATION_BOUNDS[1] and SIMULATION_BOUNDS[2] <= values["ymin"] < values["ymax"] <= SIMULATION_BOUNDS[3]):
        raise ValueError("View bounds must be ordered and lie inside the fixed -5 to 5 simulation frame")
    return values


def input_text(payload: dict) -> str:
    config = payload.get("config", {})
    view_bounds(config)
    variables = payload.get("variables", [])
    lenses = payload.get("lenses", [])
    sources = payload.get("sources", [])
    hubble = number(config.get("hubble"), "hubble")
    lines = [
        f"omega {number(config.get('omega'), 'omega')}",
        f"lambda {number(config.get('lambda'), 'lambda')}",
        f"weos {number(config.get('weos'), 'weos')}",
        f"hubble {hubble}",
        f"xmin {SIMULATION_BOUNDS[0]}",
        f"xmax {SIMULATION_BOUNDS[1]}",
        f"ymin {SIMULATION_BOUNDS[2]}",
        f"ymax {SIMULATION_BOUNDS[3]}",
        f"pix_ext {number(config.get('pix_ext'), 'pix_ext')}",
        f"pix_poi {number(config.get('pix_poi'), 'pix_poi')}",
    ]
    for variable in variables:
        name = str(variable.get("name", "")).strip()
        value = str(variable.get("value", "")).strip()
        if not SAFE_NAME.fullmatch(name) or not value or "\n" in value or "\r" in value:
            raise ValueError(f"Invalid variable: {name}")
        lines.append(f"${name}={value}")
    for index, function in enumerate(payload.get("functions", []), start=1):
        function_id = str(function.get("id", "")).strip()
        arguments = str(function.get("args", "")).strip()
        if not SAFE_FUNCTION.fullmatch(function_id) or not arguments or "\n" in arguments or "\r" in arguments:
            raise ValueError("Invalid function expression")
        lines.append(f"$fn_{index}=!{function_id}({arguments})")

    source_kinds = [str(item.get("kind", "sersic")).strip().lower() for item in sources]
    lines.append(f"startup {len(lenses)} {sum(kind != 'point' for kind in source_kinds)} {sum(kind == 'point' for kind in source_kinds)}")
    for lens in lenses:
        kind = normalize_lens_kind(lens.get("kind", "sers"))
        if kind not in SUPPORTED_LENS_KINDS:
            raise ValueError(f"Unsupported lens model '{kind}'. Supported models: sers, anfw, gaupot, sie, pow, point")
        params = [parameter_token(item, f"lens {kind} p{index + 1}") for index, item in enumerate(lens.get("parameters", []))]
        required_parameters = 4 if kind == "point" else 8
        valid_parameter_counts = {4, 8} if kind == "point" else {required_parameters}
        if len(params) not in valid_parameter_counts:
            raise ValueError(f"Lens model '{kind}' requires {required_parameters} parameters")
        if kind in {"sers", "anfw", "point"}:
            params[1] = mass_parameter_token(params[1], hubble, f"lens {kind} mass")
        if kind == "point":
            params = params[:4] + ["0"] * 4
        lines.append(f"lens {kind} {' '.join(params)}")
    for source in sources:
        kind = str(source.get("kind", "sersic")).strip().lower()
        if kind not in SUPPORTED_SOURCE_KINDS:
            raise ValueError(f"Unsupported source model '{kind}'. Supported models: sersic, gauss")
        params = [parameter_token(item, f"source {kind} p{index + 1}") for index, item in enumerate(source.get("parameters", []))]
        if len(params) != 8:
            raise ValueError(f"Extended source model '{kind}' requires 8 parameters")
        lines.append(f"extend {kind} {' '.join(params)}")
    # Keep glafic's own image products noiseless; detector noise is added by image_pipeline.py.
    lines.extend(["end_startup", "start_command", "writeimage 0 0", "writeimage_ori 0 0", "quit"])
    return "\n".join(lines) + "\n"


def critical_curves(run_dir: Path) -> dict[str, list[list[list[float]]]]:
    filename = run_dir / "_crit.dat"
    if not filename.exists() or filename.stat().st_size == 0:
        return {"critical": [], "caustic": []}
    try:
        data = np.loadtxt(filename, ndmin=2)
    except (OSError, ValueError):
        return {"critical": [], "caustic": []}
    if data.ndim != 2 or data.shape[1] != 8:
        return {"critical": [], "caustic": []}
    finite = np.isfinite(data).all(axis=1)
    return {
        "critical": data[finite][:, [0, 1, 4, 5]].reshape(-1, 2, 2).tolist(),
        "caustic": data[finite][:, [2, 3, 6, 7]].reshape(-1, 2, 2).tolist(),
    }


def curve_cache_key(payload: dict) -> str:
    config = payload.get("config", {})
    identity = {
        "profile_axis": "caustic-major-v3-abs-mu-range-0.5arcsec",
        "cosmology": {key: config.get(key) for key in ("omega", "lambda", "weos", "hubble")},
        "pix_ext": config.get("pix_ext"),
        "pix_poi": config.get("pix_poi"),
        "lenses": [{"kind": normalize_lens_kind(item.get("kind")), "parameters": item.get("parameters", [])} for item in payload.get("lenses", [])],
        "source_redshift": payload.get("sources", [{}])[0].get("parameters", [""])[0],
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return digest[:24]


def stop_curve_jobs() -> None:
    with curve_process_lock:
        processes = list(curve_processes)
    for process in processes:
        if process.poll() is None:
            process.terminate()


def publish_curve_result(run_dir: Path, curves: dict) -> None:
    pending_path = run_dir / "web_curves.json.tmp"
    pending_path.write_text(json.dumps(curves), encoding="utf-8")
    pending_path.replace(run_dir / "web_curves.json")


def caustic_major_axis(curves: dict) -> float:
    points = np.asarray(curves.get("caustic", []), dtype=float).reshape(-1, 2)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 2:
        return 0.0
    centered = points - points.mean(axis=0)
    _, _, vectors = np.linalg.svd(centered, full_matrices=False)
    direction = vectors[0]
    return float(math.atan2(direction[1], direction[0]))


def lens_info(run_dir: Path, curves: dict) -> dict:
    filename = run_dir / "_lens.fits"
    if not filename.exists():
        filename = run_dir / "lens.fits"
    with fits.open(filename, memmap=True) as hdul:
        data = next((np.asarray(hdu.data, dtype=float) for hdu in hdul if hdu.data is not None), None)
    if data is None or data.ndim != 3 or data.shape[0] < 7:
        raise ValueError("Unsupported writelens FITS layout")

    kappa = data[3]
    inverse_magnification = data[6]
    height, width = kappa.shape
    pixel_size = (SIMULATION_BOUNDS[1] - SIMULATION_BOUNDS[0]) / width
    x_coords = (np.arange(width) + 0.5) * pixel_size + SIMULATION_BOUNDS[0]
    y_coords = (np.arange(height) + 0.5) * pixel_size + SIMULATION_BOUNDS[2]
    angle = caustic_major_axis(curves)
    cosine, sine = math.cos(angle), math.sin(angle)
    x_limit = (SIMULATION_BOUNDS[1] - SIMULATION_BOUNDS[0]) / (2.0 * max(abs(cosine), 1.0e-12))
    y_limit = (SIMULATION_BOUNDS[3] - SIMULATION_BOUNDS[2]) / (2.0 * max(abs(sine), 1.0e-12))
    profile_limit = min(0.5, x_limit, y_limit)
    coordinates = np.linspace(-profile_limit, profile_limit, 401)
    x_samples = coordinates * cosine
    y_samples = coordinates * sine
    column_samples = (x_samples - SIMULATION_BOUNDS[0]) / pixel_size - 0.5
    row_samples = (y_samples - SIMULATION_BOUNDS[2]) / pixel_size - 0.5
    sample_coordinates = np.vstack((row_samples, column_samples))
    sampled_kappa = map_coordinates(kappa, sample_coordinates, order=1, mode="constant", cval=np.nan)
    sampled_inverse = map_coordinates(inverse_magnification, sample_coordinates, order=1, mode="constant", cval=np.nan)
    magnifications = np.abs(np.divide(1.0, sampled_inverse, out=np.full_like(sampled_inverse, np.nan), where=np.abs(sampled_inverse) > 1.0e-8))
    clean = lambda values: [float(value) if np.isfinite(value) else None for value in values]
    result = {"r": clean(coordinates), "kappa": clean(sampled_kappa), "magnification": clean(magnifications), "angle": math.degrees(angle)}
    write_lens_info_plot(run_dir, result)
    return result


def write_lens_info_plot(run_dir: Path, profile: dict) -> Path:
    radius = np.asarray(profile.get("r", []), dtype=float)
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    figure.suptitle(f"Profiles along caustic major axis ({float(profile.get('angle', 0.0)):.1f} deg)")
    plots = (("kappa", r"Convergence, $\kappa$", "#b53030"), ("magnification", r"Absolute magnification, $|\mu|$", "#26734d"))
    for axis, (key, title, color) in zip(axes, plots):
        values = np.asarray(profile.get(key, []), dtype=float)
        valid = np.isfinite(radius) & np.isfinite(values)
        axis.plot(radius[valid], values[valid], color=color, linewidth=1.8)
        axis.set_xlim(-0.5, 0.5)
        if key == "magnification":
            axis.set_yscale("log")
        axis.set_title(title)
        axis.set_xlabel("Coordinate along caustic major axis [arcsec]")
        axis.set_ylabel(r"$\kappa$" if key == "kappa" else r"$|\mu|$")
        axis.grid(True, color="#dfe6df", linewidth=0.8)
        axis.tick_params(direction="out")
    figure.savefig(run_dir / "lens_info.png", dpi=150, facecolor="white")
    plt.close(figure)
    return run_dir / "lens_info.png"


def empty_diagnostics(error: str | None = None) -> dict:
    result = {"critical": [], "caustic": [], "lensInfo": {"r": [], "kappa": [], "magnification": []}}
    if error:
        result["error"] = error
    return result


def generate_critical_curves(run_dir: Path, source_redshift: str, cache_path: Path) -> None:
    executable = str(ROOT / "glafic") if (ROOT / "glafic").exists() else shutil.which("glafic")
    if executable is None:
        publish_curve_result(run_dir, empty_diagnostics())
        return
    input_path = run_dir / "diagnostics.input"
    run_text = (run_dir / "run.input").read_text(encoding="utf-8")
    header = run_text.split("start_command", 1)[0]
    input_path.write_text(f"{header}start_command\nwritecrit {source_redshift}\nwritelens {source_redshift}\nquit\n", encoding="utf-8")
    process = None
    try:
        process = subprocess.Popen([executable, str(input_path)], cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        with curve_process_lock:
            curve_processes.add(process)
        try:
            returncode = process.wait(timeout=120)
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait()
            publish_curve_result(run_dir, empty_diagnostics("Critical-curve generation timed out after 120 seconds"))
            return
        if returncode == 0:
            curves = critical_curves(run_dir)
            curves["lensInfo"] = lens_info(run_dir, curves)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_tmp = cache_path.with_suffix(".tmp")
            cache_tmp.write_text(json.dumps(curves), encoding="utf-8")
            cache_tmp.replace(cache_path)
        else:
            error_output = process.stderr.read().decode("utf-8", errors="replace").strip() if process.stderr else ""
            curves = empty_diagnostics(error_output.splitlines()[-1] if error_output else f"glafic exited with status {returncode}")
        publish_curve_result(run_dir, curves)
    except (OSError, ValueError, IndexError) as error:
        publish_curve_result(run_dir, empty_diagnostics(str(error)))
    finally:
        if process is not None:
            with curve_process_lock:
                curve_processes.discard(process)
        input_path.unlink(missing_ok=True)


def validate_payload(payload: dict) -> None:
    if not isinstance(payload, dict):
        raise ValueError("Request must be a JSON object")
    if not payload.get("lenses"):
        raise ValueError("Add at least one lens component")
    if not payload.get("sources"):
        raise ValueError("Add at least one source component")
    view_bounds(payload.get("config", {}))
    for lens_index, lens in enumerate(payload.get("lenses", []), start=1):
        kind = normalize_lens_kind(lens.get("kind", ""))
        if kind not in SUPPORTED_LENS_KINDS:
            raise ValueError(f"Lens {lens_index} uses unsupported model '{kind}'")
        if kind in LUMINOUS_LENS_KINDS:
            if str(lens.get("template", "")).strip() not in {item["id"] for item in SPECTRAL_TEMPLATES}:
                raise ValueError(f"Lens {lens_index} requires a supported spectrum template")
            ratio = float(number(lens.get("massToLight"), f"lens {lens_index} massToLight"))
            if not math.isfinite(ratio) or ratio <= 0.0:
                raise ValueError(f"Lens {lens_index} mass-to-light ratio must be positive")
    for source_index, source in enumerate(payload.get("sources", []), start=1):
        kind = str(source.get("kind", "")).strip().lower()
        if kind not in SUPPORTED_SOURCE_KINDS:
            raise ValueError(f"Source {source_index} uses unsupported model '{kind}'")
        if str(source.get("template", "")).strip() not in {item["id"] for item in SPECTRAL_TEMPLATES}:
            raise ValueError(f"Source {source_index} requires a supported spectrum template")
        luminosity = float(number(source.get("luminosity"), f"source {source_index} luminosity"))
        if not math.isfinite(luminosity) or luminosity <= 0.0:
            raise ValueError(f"Source {source_index} luminosity must be positive")
    for function in payload.get("functions", []):
        if not SAFE_FUNCTION.fullmatch(str(function.get("id", ""))):
            raise ValueError("Unsupported function reference")


def recent_images(limit: int = 3) -> list[dict[str, str]]:
    candidates = []
    for image in OUT.glob("*/ie_band.png"):
        candidates.append((image.stat().st_mtime_ns, image))
    candidates.sort(key=lambda item: item[0], reverse=True)
    for _, image in candidates[limit:]:
        for old_image in (image, image.with_suffix(".jpg"), image.with_name("ie_band_data.json")):
            if old_image.exists():
                old_image.unlink()
    results = []
    for _, image in candidates[:limit]:
        result = {"name": image.parent.name, "image": "/" + str(image.relative_to(ROOT)).replace("\\", "/")}
        data_path = image.with_name("ie_band_data.json")
        if data_path.exists():
            result["data"] = "/" + str(data_path.relative_to(ROOT)).replace("\\", "/")
        lens_info_path = image.with_name("lens_info.png")
        if lens_info_path.exists():
            result["lensInfoImage"] = "/" + str(lens_info_path.relative_to(ROOT)).replace("\\", "/")
        curves_path = image.with_name("web_curves.json")
        if curves_path.exists():
            diagnostics = json.loads(curves_path.read_text(encoding="utf-8"))
            result["curves"] = diagnostics
            result["lensInfo"] = diagnostics.get("lensInfo", {"r": [], "kappa": [], "magnification": []})
        results.append(result)
    return results


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        return

    def send_json(self, value: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(value).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/catalog":
            self.send_json(catalog())
            return
        if path == "/api/presets":
            self.send_json({"presets": list_presets()})
            return
        if path.startswith("/api/spectrum/"):
            try:
                template = unquote(path.removeprefix("/api/spectrum/"))
                query = parse_qs(urlparse(self.path).query)
                redshift = float(query.get("z", ["0"])[0])
                if not math.isfinite(redshift) or redshift < 0.0:
                    raise ValueError("Redshift must be finite and non-negative")
                wavelengths, fluxes = template_curve(template, redshift)
                filter_wavelengths, transmission = _vis_filter()
                self.send_json({"template": template, "wavelengths": wavelengths.tolist(), "fluxes": fluxes.tolist(), "filterWavelengths": filter_wavelengths.tolist(), "transmission": transmission.tolist()})
            except (OSError, ValueError, TypeError) as error:
                self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
            return
        if path.startswith("/runs/"):
            target = (OUT / path.removeprefix("/runs/")).resolve()
            if OUT not in target.parents or not target.is_file() and target.name != "web_curves.json":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            if not target.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            body = target.read_bytes()
            content_type = "image/png" if target.suffix == ".png" else "application/json; charset=utf-8" if target.suffix == ".json" else "text/plain; charset=utf-8"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        relative = "index.html" if path == "/" else path.removeprefix("/")
        target = (STATIC / relative).resolve()
        if STATIC not in target.parents or not target.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content_type = "text/html; charset=utf-8" if target.suffix == ".html" else "text/css; charset=utf-8" if target.suffix == ".css" else "text/javascript; charset=utf-8"
        body = target.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/presets":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length))
                name = str(payload.get("name", "")).strip()
                if not name or not isinstance(payload.get("data"), dict):
                    raise ValueError("Preset name and data are required")
                PRESETS.mkdir(exist_ok=True)
                preset_id = f"preset_{time.time_ns()}"
                path = preset_path(preset_id)
                path.write_text(json.dumps({"name": name, "data": payload["data"], "created": time.time_ns()}, indent=2), encoding="utf-8")
                self.send_json({"preset": {"id": preset_id, "name": name, "data": payload["data"]}})
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
                self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
            return
        if path not in {"/api/preview", "/api/run"}:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            validate_payload(payload)
            text = input_text(payload)
            if path == "/api/preview":
                with tempfile.TemporaryDirectory() as directory:
                    source = Path(directory) / "web.input"
                    source.write_text(text, encoding="utf-8")
                    parsed = parse_input_file(source, resolve_only=True)
                    self.send_json({"input": parsed.to_input_text(), "runs": len(getattr(parsed, "runs", [parsed]))})
                return
            stop_curve_jobs()
            OUT.mkdir(exist_ok=True)
            with tempfile.TemporaryDirectory() as directory:
                source = Path(directory) / "web.input"
                source.write_text(text, encoding="utf-8")
                parsed = parse_input_file(source)
                results = execute_runs(parsed, workspace_root=ROOT, out_root=OUT, fallback_name=f"web_run_{time.time_ns()}", reporter=ConsoleReporter(LogLevel.QUIET), max_workers=1)
            rendered = []
            for result in results:
                metadata = {"lenses": payload.get("lenses", []), "sources": payload.get("sources", []), "view": view_bounds(payload.get("config", {}))}
                (result.output_dir / "web_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
                image = render_i_band(result.output_dir, seed=int(payload.get("seed", 42)), metadata=metadata)
                (result.output_dir / "web_curves.json").unlink(missing_ok=True)
                output = str(result.output_dir.relative_to(ROOT)).replace("\\", "/")
                source_redshift = parameter_token(payload["sources"][0]["parameters"][0], "source redshift")
                cache_path = CURVE_CACHE / f"{curve_cache_key(payload)}.json"
                cached_diagnostics = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else None
                if isinstance(cached_diagnostics, dict) and "lensInfo" in cached_diagnostics:
                    shutil.copyfile(cache_path, result.output_dir / "web_curves.json")
                    curves = cached_diagnostics
                    write_lens_info_plot(result.output_dir, curves["lensInfo"])
                    curves_pending = False
                else:
                    threading.Thread(target=generate_critical_curves, args=(result.output_dir, source_redshift, cache_path), daemon=True).start()
                    curves = empty_diagnostics()
                    curves_pending = True
                rendered.append({"name": result.name, "output": output, "image": "/" + str(image.relative_to(ROOT)).replace("\\", "/"), "curves": curves, "lensInfo": curves.get("lensInfo", {"r": [], "kappa": [], "magnification": []}), "lensInfoImage": "/" + str((result.output_dir / "lens_info.png").relative_to(ROOT)).replace("\\", "/") if not curves_pending else None, "diagnosticsPending": curves_pending})
            images = recent_images()
            for entry in images:
                matching = next((run for run in rendered if run["name"] == entry["name"]), None)
                if matching is not None:
                    entry["diagnosticsPending"] = matching["diagnosticsPending"]
                    entry["lensInfoImage"] = matching["lensInfoImage"]
                    if matching["diagnosticsPending"]:
                        entry["diagnosticsUrl"] = f"/{matching['output']}/web_curves.json"
            self.send_json({"input": text, "results": rendered, "images": images})
        except Exception as error:
            self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)

    def do_DELETE(self) -> None:
        path = urlparse(self.path).path
        if not path.startswith("/api/presets/"):
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            preset_path(path.removeprefix("/api/presets/")).unlink()
            self.send_json({"deleted": True})
        except FileNotFoundError:
            self.send_error(HTTPStatus.NOT_FOUND)
        except (OSError, ValueError) as error:
            self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    recent_images()
    print(f"Lensing simulator at http://127.0.0.1:{port}")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
