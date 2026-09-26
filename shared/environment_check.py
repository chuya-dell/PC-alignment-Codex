"""Read local source data without modification and verify local output."""
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import tempfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    output = ROOT / "data/results/environment_check"
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output / "matplotlib"))
    packages = {
        "numpy": "numpy", "opencv-python": "cv2", "pandas": "pandas",
        "scipy": "scipy", "scikit-image": "skimage", "matplotlib": "matplotlib",
        "tqdm": "tqdm", "roifile": "roifile",
    }
    versions = {}
    for distribution, module in packages.items():
        importlib.import_module(module)
        versions[distribution] = importlib.metadata.version(distribution)
    from shared.registration import load_image_unicode
    import numpy as np
    config = json.loads((ROOT / "data/raw/local_environment.json").read_text(encoding="utf-8"))
    image_path = Path(config["sample_image"])
    before = hashlib.sha256(image_path.read_bytes()).hexdigest()
    pixels = load_image_unicode(str(image_path))
    if pixels is None or pixels.ndim != 2 or not np.isfinite(pixels).all():
        raise RuntimeError("Image decoding failed")
    with tempfile.TemporaryDirectory(prefix="write_check_", dir=output) as temporary:
        saved = Path(temporary) / "pixels.npy"
        np.save(saved, pixels)
        if not np.array_equal(np.load(saved), pixels):
            raise RuntimeError("Output round trip failed")
    after = hashlib.sha256(image_path.read_bytes()).hexdigest()
    if before != after:
        raise RuntimeError("Source image changed during check")
    spectrum = Path(config["sample_spectrum"])
    spectrum_bytes = spectrum.read_bytes()
    if not spectrum_bytes:
        raise RuntimeError("Spectrum file is empty")
    manifest = json.loads((ROOT / "data/raw/target_manifest.local.json").read_text(encoding="utf-8"))
    missing = [entry["root"] for entry in manifest if not Path(entry["root"]).is_dir()]
    if missing:
        raise RuntimeError(f"Missing data directories: {missing}")
    report = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version, "executable": sys.executable,
        "packages": versions, "image_path": str(image_path),
        "image_shape": list(pixels.shape), "loaded_dtype": str(pixels.dtype),
        "image_sha256": before, "source_image_unchanged": True,
        "output_round_trip": "passed", "spectrum_path": str(spectrum),
        "spectrum_bytes": len(spectrum_bytes),
        "spectrum_sha256": hashlib.sha256(spectrum_bytes).hexdigest(),
        "spectrum_check": "binary read only; format interpretation not tested",
        "manifest_directories_checked": len(manifest),
        "scope": "Environment smoke test only; no scientific analysis validation",
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
