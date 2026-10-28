# Lensing simulator website

This project is a browser-based front end for building and running gravitational lensing simulations with `glafic`, then producing a Euclid-like VIS I-band preview. It is designed for interactive exploration: you can assemble lens mass models and source light models, adjust parameters, preview the generated `glafic` input, and run the simulation to produce a noisy synthetic image.

The website does not replace the physics engine. It generates the input files, runs the simulation, and converts the resulting image products into a detector-like observational preview. Most of the scientific model logic still lives in the `glafic` pipeline and the Python rendering layer under `gflex/`.

## Features

- Graphical editing for lens and source components
- Parameter functions and variables with a small scripting interface
- Preset save/load support
- `glafic` input preview
- Simulation execution through the bundled `glafic` binary
- Synthetic I-band image generation with detector effects
- Lens diagnostic products: critical curves, caustics, and radial profiles
- Past-results gallery with side-by-side comparison and residual images
- Single-run and CPU-parallel batch execution with live progress
- Gaussian potential (`gaupot`) fields follow glafic's manual: `z_lens`, `z_source/fid`, `x`, `y`, `e_p`, `theta_e`, `sigma`, `kappa_0`

## Quick start

Compile glafic on your system. For guidance, refer to the glafic2 github repository. Copy the compiled executable into gflex/

From the project directory:

```powershell
pip install -r requirements.txt
python server.py
```

Then open:

```text
http://127.0.0.1:8765
```

The web app is a standard-library HTTP server with no frontend build step.

## How the app works

### 1. Build a lensing scene

The UI lets you add supported lens and source components to a model canvas. Supported lens kinds include:

- Sersic
- Approximate NFW
- Gaussian lens potential (`gaupot`)
- SIE
- Elliptical power law
- Point mass

Supported source/light kinds include:

- Sersic light
- Gaussian light

Each component has parameter fields that match the `glafic` input conventions. Values may reference variables defined in the Functions tab using `$name`, and NumPy functions are available as `np`. All defined variables should be preceded by `$` (e.g. $lens_z). The default scaling laws can be accessed in the 'Functions' tab by clicking on them or writing out the full function name. Note that default functions have to be preceded by `!`. For more examples, load the example preset.

The app stores UI metadata separately from the input file. For example, `web_metadata.json` contains photometric information like luminosity, spectral template choice, and lens mass-to-light ratios; this metadata is not sent directly to `glafic`.

### 2. Preview the generated input

The Preview input action validates the assembled configuration and prints a generated `run.input` file in the format expected by `glafic`.

The generated input follows the usual `glafic` input file pattern:

- standard primary cosmology and simulation parameters
- a counted `startup` block
- per-component parameter lists
- `end_startup`
- `start_command`
- image-writing commands
- `quit`

### 3. Run the simulation

Choose Single run or Batch run beside the Run button. For Batch run, enter a batch name
and the number of runs (between 1 and 10,000). Batch names may contain letters, numbers,
spaces, hyphens, underscores, and periods, and must start with a letter or number.
The backend creates one independent run directory per sample and executes the `glafic`
jobs concurrently, bounded by the system CPU count. The Functions tab can define one
sampled variable per assignment, for example:

```text
$halo_mass = !parameters.sample_normal(12.0, 0.15)
$x_offset = !parameters.sample_uniform(-0.5, 0.5)
```

The distribution helpers include `sample_normal`, `sample_uniform`, `sample_lognormal`,
`sample_exponential`, and `sample_choice`. Each batch sample is resolved independently,
and later arithmetic expressions can reference sampled variables. The UI displays the
latest completed image, recent image history, progress, and diagnostic status while the
batch is running. Critical-curve and lens-information diagnostics are computed for each
run after its image is rendered; the UI polls unfinished diagnostics and enables the
critical-curve overlay when they are ready.

When the batch finishes, results are stored under
`runs/batch/<batch-name>/<run-name>/`. Each run directory contains its own resolved
`run.input`, `header.json`, raw glafic products, physical-flux FITS products, image
preview, and diagnostics. A failed batch reports the backend error in the output panel;
completed runs remain available in the batch directory.

When you run the simulation, the backend executes the `glafic` binary in the project run
directory. The resulting image products are then read back and processed into a
grayscale I-band preview.

The latest image is shown in the main viewer, with a short recent history available below it.
The `Past results` tab lists up to 50 completed results retained in `runs/`. Select any
two results with matching image dimensions to view both images side by side and inspect
a signed residual image (result A minus result B).

### 4. Produce the I-band observational preview

The final rendered preview is not a raw `glafic` output. It is a Python post-processing pipeline that:

- reads the emitted lens/source maps
- converts component flux into physical values using luminosity and cosmology
- applies a VIS-like spectral throughput
- convolves the image with a Moffat PSF
- bins to detector pixels
- adds sky background, Poisson noise, and read noise
- crops to the requested view and normalizes for display

The final display output is stored as `ie_band.png`, and the underlying normalized data
are saved in `ie_band_data.json`. The same linear detector values are also written to
`ie_band_linear.fits` without the display asinh stretch. In the web app, you can adjust
the display stretch by clicking the fullscreen button and adjusting the asinh stretch
bar. Once the critical curves are computed, you can access the critical curves in the
image and the other lens information (for example caustics, convergence, and
magnification) in the `Lens Info` tab.

## Typical workflow

1. Drag lens and source cards into the scene.
2. Set model parameters and redshifts.
3. Define helper variables or scaling rules in the Functions tab.
4. Use Preview input to inspect the generated `run.input`.
5. Run the simulation to generate the image products.
6. Inspect the I-band preview and lens diagnostics.
7. Open Past results to compare two completed simulations and inspect their residual.
8. Save and reload presets using the Presets tab.

## Output and file layout

The project uses a few important folders and files:

- `server.py` — HTTP server and request handling
- `core/` — parsing and simulation setup helpers
- `static/` — frontend assets
- `templates/` — spectral templates used for photometry
- `presets/` — saved user presets
- `runs/` — per-run output directories

Each run directory usually contains files such as:

- `run.input` — `glafic` input file
- `image.fits` / `source.fits` — emitted simulation maps
- `ie_band.png` — grayscale I-band preview
- `ie_band_data.json` — normalized detector-plane values used for display
- `ie_band_linear.fits` — linear normalized detector-plane values without display stretch
- `header.json` — resolved configuration, variables, lenses, and sources for this run
- `lens_image.fits` — unstretched physical-flux lensed image plus source-light FITS product
- `lensed_image_only.fits` — unstretched physical-flux lensed image component without source light
- `lensing_galaxy_light.fits` — unstretched physical-flux source-light FITS product
- `web_curves.json` — critical curves, caustics, and lens-information diagnostics
- `lens_info.png` — radial lens-information plot when diagnostics are complete

The FITS values are scaled by `1e30` for numerical readability, with
`BUNIT = 1e30 erg s-1 cm-2 pixel-1`. The lensed and unlensed source maps are
normalized using the configured source luminosity and luminosity distance before
being written; no display stretch, clipping, or normalization is applied.

## Appendix: technical details

This appendix summarizes the actual code-level assumptions used in the rendering pipeline.

### Detector and sky model

The I-band rendering pipeline uses Euclid VIS-like assumptions defined in `image_pipeline.py`:

- `A_EFF_CM2 = 1.006e4` (effective light-gathering area)
- `PIVOT_ANGSTROM = 7200.0` (pivot wavelength of Euclid I-band)
- `BANDWIDTH_ANGSTROM = 3000.0` (approximate band width)
- `EXPOSURE_SECONDS = 565.0 * 4.0 + 100.0 * 2.0 = 2460.0 s` (EWS exposures)
- `READ_NOISE_ELECTRONS = 4.4`
- `SKY_ABMAG_PER_ARCSEC2 = 21.0` (fiducial sky background)
- `DETECTOR_PIXEL_ARCSEC = 0.1` (I-band pixel size)
- `SIMULATION_BOUNDS = (-5.0, 5.0, -5.0, 5.0)`

### PSF and detector sampling

The rendered I-band image is produced as follows:

- source and lens component maps are combined into flux maps
- a Moffat PSF kernel is created via `_moffat_kernel(...)`
- each flux component is convolved with the kernel using `scipy.signal.fftconvolve(..., mode="same")`
- the result is binned to detector pixels at 0.1 arcsec

The PSF kernel is a Moffat profile. The code uses a default `beta = 3.0` and a width derived from `0.18 / pixel_size` arcseconds.

### Noise model

Detector noise is added in `_add_detector_noise(...)`:

- shot noise from the expected electrons is sampled with a Poisson distribution
- read noise is added with a Gaussian distribution of 4.4 electrons
- sky photons are included using the model in `_sky_flux_per_pixel(...)`
- the final image is normalized to electrons per unit gain and exposure

This means the rendered preview is intentionally a noisy image, not a clean `glafic` output map.

### Photometric templates and SEDs

Bright galaxy and AGN components use spectral templates from the included `templates/` directory. Supported templates include:

- elliptical
- spiral
- late_spiral
- irregular
- starburst
- dusty_starburst
- AGN (`j0647agn_torus.fits`)

These are shifted by redshift and convolved with the VIS throughput curve for the filter-integrated photon gain used in the detector model.

### Lens diagnostics

The website can generate lens-information diagnostics by running `glafic` in diagnostic mode and reading the output FITS products. These include:

- critical curves
- caustics
- convergence and magnification profiles along the caustic major axis

The corresponding plots are generated and stored on disk for visualization in the Lens Info tab.

### Important caveat

The code intentionally keeps the underlying `glafic` products mostly noiseless. The noise, PSF blur, and detector binning are added afterward in the Python rendering layer to create a more realistic I-band observation preview.

## Notes for developers

- The backend is intentionally simple and uses the Python standard library HTTP server.
- There is no frontend build process; static files are served directly.
- Scientific input generation is handled in Python, while `glafic` does the actual gravitational lensing calculation.
- Most of the image realism comes from the post-processing code in `image_pipeline.py`, not from the raw `glafic` output itself.

## Related files

- `server.py`
- `image_pipeline.py`
- `core/parser.py`
- `core/runs.py`
- `templates/`
- `static/`
