r"""
Lab 6: Stereo 3D reconstruction from StereoPi SBS video.

Input video example:
    D:\CVLabs\Lab6\practic\chess\1943352024.mp4

What it does:
1) Reads a side-by-side stereo video and splits it into left/right frames.
2) Optionally calibrates the stereo pair from chessboard frames in the same video.
3) Rectifies stereo frames when calibration succeeds.
4) Builds a disparity/depth visualization by classic OpenCV StereoSGBM.
5) Builds a neural stereo disparity visualization from the left/right pair using PSMNet.
6) Displays both mapped videos simultaneously and prints simple comparison metrics.

Install:
    pip install opencv-contrib-python numpy torch torchvision tqdm

First run example:
    python lab6_depth_video.py --video "C:\Users\Kysya\Desktop\учёба\Магистратура\CV\CV-labs\CV6\1943352024.mp4" --calibrate

If chessboard detection fails, set the inner-corner count, for example:
    python lab6_depth_video.py --video "...mp4" --calibrate --pattern-cols 9 --pattern-rows 6

Then reuse calibration:
    python lab6_depth_video.py --video "...mp4" --calib-file stereo_calib.npz
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List

import cv2 as cv
import numpy as np


@dataclass
class StereoCalibration:
    K1: np.ndarray
    D1: np.ndarray
    K2: np.ndarray
    D2: np.ndarray
    R: np.ndarray
    T: np.ndarray
    image_size: Tuple[int, int]
    map1x: Optional[np.ndarray] = None
    map1y: Optional[np.ndarray] = None
    map2x: Optional[np.ndarray] = None
    map2y: Optional[np.ndarray] = None
    Q: Optional[np.ndarray] = None


def split_sbs(frame: np.ndarray, swap: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Split a side-by-side stereo frame into left and right images."""
    h, w = frame.shape[:2]
    mid = w // 2
    left, right = frame[:, :mid].copy(), frame[:, mid:].copy()
    if swap:
        left, right = right, left
    return left, right


def ensure_even_disparities(value: int) -> int:
    """OpenCV requires numDisparities to be divisible by 16."""
    return max(16, int(np.ceil(value / 16.0)) * 16)


def normalize_to_u8(x: np.ndarray, invalid_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Robust normalization to 8-bit for visualization."""
    arr = x.astype(np.float32).copy()
    if invalid_mask is not None:
        arr[invalid_mask] = np.nan
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros(arr.shape[:2], dtype=np.uint8)
    lo, hi = np.nanpercentile(arr[finite], [2, 98])
    if hi <= lo:
        hi = lo + 1.0
    out = np.clip((arr - lo) * 255.0 / (hi - lo), 0, 255)
    out[~finite] = 0
    return out.astype(np.uint8)


def colorize(gray_u8: np.ndarray, cmap: int = cv.COLORMAP_MAGMA) -> np.ndarray:
    return cv.applyColorMap(gray_u8, cmap)


def make_sgbm(width: int, block_size: int = 5, max_disp_fraction: float = 0.30) -> cv.StereoSGBM:
    # For 640 px half-frame, 192 disparities is a reasonable starting point.
    num_disp = ensure_even_disparities(width * max_disp_fraction)
    block_size = block_size if block_size % 2 == 1 else block_size + 1
    channels = 1
    return cv.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=8 * channels * block_size ** 2,
        P2=32 * channels * block_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=8,
        speckleWindowSize=100,
        speckleRange=32,
        preFilterCap=63,
        mode=cv.STEREO_SGBM_MODE_SGBM_3WAY,
    )


def compute_sgbm_depth_vis(
    left_bgr: np.ndarray,
    right_bgr: np.ndarray,
    stereo: cv.StereoSGBM,
    focal_px: Optional[float] = None,
    baseline_m: float = 0.17,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Return disparity, colored visualization, and metric depth if focal is known."""
    gray_l = cv.cvtColor(left_bgr, cv.COLOR_BGR2GRAY)
    gray_r = cv.cvtColor(right_bgr, cv.COLOR_BGR2GRAY)

    disparity_raw = stereo.compute(gray_l, gray_r).astype(np.float32) / 16.0
    valid = disparity_raw > 1.0

    # For display: larger disparity means closer, therefore brighter/warmer.
    disp_u8 = normalize_to_u8(disparity_raw, invalid_mask=~valid)
    disp_color = colorize(disp_u8)

    depth_m = None
    if focal_px is not None:
        depth_m = np.full_like(disparity_raw, np.nan, dtype=np.float32)
        depth_m[valid] = (float(focal_px) * float(baseline_m)) / disparity_raw[valid]

    return disparity_raw, disp_color, depth_m


class NeuralStereoPSMNet:
    """PSMNet wrapper: predicts disparity from a rectified left/right stereo pair."""

    DEFAULT_REPO_URL = "https://github.com/JiaRenChang/PSMNet.git"
    DEFAULT_GDRIVE_ID = "1NDKrWHkwgMKtDwynXVU12emK3G5d5kkp"

    def __init__(self, repo_path: str = "PSMNet", weights_path: str = "psmnet_sceneflow_torch18.tar", device: Optional[str] = None, maxdisp: int = 192, model_name: str = "stackhourglass"):
        import sys
        import torch
        import torch.nn.functional as F
        import torchvision.transforms as transforms
        from PIL import Image
        # Original PSMNet code has hard-coded .cuda() calls inside forward().
        # If the installed PyTorch has no CUDA support, make .cuda() a safe no-op
        # so the model can run on CPU. This is slower, but works for lab/demo use.
        if not torch.cuda.is_available():
            if not getattr(torch.Tensor, "_psmnet_cuda_patched", False):
                def _tensor_cuda_noop(self, device=None, non_blocking=False, memory_format=None):
                    return self
                def _module_cuda_noop(self, device=None):
                    return self
                torch.Tensor.cuda = _tensor_cuda_noop
                torch.nn.Module.cuda = _module_cuda_noop
                torch.Tensor._psmnet_cuda_patched = True

        self.torch = torch
        self.F = F
        self.Image = Image
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.maxdisp = int(maxdisp)
        repo = Path(repo_path).expanduser().resolve()
        weights = Path(weights_path).expanduser().resolve()
        if not repo.exists():
            raise FileNotFoundError(f"PSMNet repo not found: {repo}\nRun: git clone {self.DEFAULT_REPO_URL} {repo}")
        if not weights.exists():
            raise FileNotFoundError(f"PSMNet weights not found: {weights}\nRun with --setup-psmnet or download a .tar checkpoint from the official PSMNet README.")
        sys.path.insert(0, str(repo))
        try:
            from models import stackhourglass, basic
        except Exception as e:
            raise ImportError("Cannot import PSMNet models. Check --psmnet-repo path.") from e
        if model_name == "stackhourglass":
            model = stackhourglass(self.maxdisp)
        elif model_name == "basic":
            model = basic(self.maxdisp)
        else:
            raise ValueError("--psmnet-model must be 'stackhourglass' or 'basic'")
        state = torch.load(str(weights), map_location=self.device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        fixed = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(fixed, strict=False)
        self.model = model.to(self.device).eval()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @staticmethod
    def setup(repo_path: str, weights_path: str, download_weights: bool = True) -> None:
        import subprocess, sys
        repo = Path(repo_path).expanduser().resolve()
        weights = Path(weights_path).expanduser().resolve()
        if not repo.exists():
            subprocess.check_call(["git", "clone", NeuralStereoPSMNet.DEFAULT_REPO_URL, str(repo)])
        else:
            print(f"[setup] Repo already exists: {repo}")
        if download_weights and not weights.exists():
            try:
                import gdown
            except Exception:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
                import gdown
            weights.parent.mkdir(parents=True, exist_ok=True)
            url = f"https://drive.google.com/uc?id={NeuralStereoPSMNet.DEFAULT_GDRIVE_ID}"
            gdown.download(url, str(weights), quiet=False)
        elif weights.exists():
            print(f"[setup] Weights already exist: {weights}")

    def _to_tensor(self, bgr: np.ndarray):
        rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)
        return self.transform(self.Image.fromarray(rgb))

    def predict_disparity(self, left_bgr: np.ndarray, right_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        torch, F = self.torch, self.F
        img_l = self._to_tensor(left_bgr)
        img_r = self._to_tensor(right_bgr)
        _, h, w = img_l.shape
        top_pad = (16 - h % 16) % 16
        right_pad = (16 - w % 16) % 16
        img_l = F.pad(img_l, (0, right_pad, top_pad, 0)).unsqueeze(0).to(self.device)
        img_r = F.pad(img_r, (0, right_pad, top_pad, 0)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            disp = self.model(img_l, img_r)
            if isinstance(disp, (list, tuple)):
                disp = disp[-1]
            disp = torch.squeeze(disp).detach().cpu().numpy().astype(np.float32)
        if top_pad > 0:
            disp = disp[top_pad:, :]
        if right_pad > 0:
            disp = disp[:, :-right_pad]
        valid = np.isfinite(disp) & (disp > 0.1) & (disp < self.maxdisp)
        return disp, colorize(normalize_to_u8(disp, invalid_mask=~valid), cv.COLORMAP_VIRIDIS)

def calibrate_from_video(
    video_path: str,
    pattern_size: Tuple[int, int],
    square_size_m: float,
    max_pairs: int,
    frame_step: int,
    swap: bool,
    save_file: str,
) -> StereoCalibration:
    """Stereo calibration from chessboard frames in an SBS video."""
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    cols, rows = pattern_size
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_size_m

    objpoints: List[np.ndarray] = []
    imgpoints_l: List[np.ndarray] = []
    imgpoints_r: List[np.ndarray] = []
    image_size: Optional[Tuple[int, int]] = None

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 40, 1e-3)
    idx = 0
    found = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % frame_step != 0:
            idx += 1
            continue
        left, right = split_sbs(frame, swap=swap)
        if image_size is None:
            image_size = (left.shape[1], left.shape[0])

        gl = cv.cvtColor(left, cv.COLOR_BGR2GRAY)
        gr = cv.cvtColor(right, cv.COLOR_BGR2GRAY)
        ok_l, corners_l = cv.findChessboardCorners(gl, pattern_size, None)
        ok_r, corners_r = cv.findChessboardCorners(gr, pattern_size, None)

        if ok_l and ok_r:
            corners_l = cv.cornerSubPix(gl, corners_l, (11, 11), (-1, -1), criteria)
            corners_r = cv.cornerSubPix(gr, corners_r, (11, 11), (-1, -1), criteria)
            objpoints.append(objp.copy())
            imgpoints_l.append(corners_l)
            imgpoints_r.append(corners_r)
            found += 1
            print(f"Calibration pair {found}/{max_pairs} found at frame {idx}")
            if found >= max_pairs:
                break
        idx += 1

    cap.release()

    if image_size is None or len(objpoints) < 8:
        raise RuntimeError(
            "Too few stereo chessboard detections. Try --pattern-cols/--pattern-rows, "
            "lower --frame-step, or use a video where the whole chessboard is visible in both halves."
        )

    _, K1, D1, K2, D2, R, T, E, F = cv.stereoCalibrate(
        objpoints,
        imgpoints_l,
        imgpoints_r,
        None,
        None,
        None,
        None,
        image_size,
        criteria=criteria,
        flags=cv.CALIB_FIX_INTRINSIC if False else 0,
    )

    calib = StereoCalibration(K1=K1, D1=D1, K2=K2, D2=D2, R=R, T=T, image_size=image_size)
    prepare_rectification(calib)
    np.savez(save_file, K1=K1, D1=D1, K2=K2, D2=D2, R=R, T=T, image_size=np.array(image_size), Q=calib.Q)
    print(f"Saved calibration to {save_file}")
    return calib


def load_calibration(path: str) -> StereoCalibration:
    data = np.load(path)
    image_size = tuple(int(x) for x in data["image_size"])
    calib = StereoCalibration(
        K1=data["K1"], D1=data["D1"], K2=data["K2"], D2=data["D2"],
        R=data["R"], T=data["T"], image_size=image_size,
    )
    prepare_rectification(calib)
    return calib


def prepare_rectification(calib: StereoCalibration) -> None:
    R1, R2, P1, P2, Q, _, _ = cv.stereoRectify(
        calib.K1, calib.D1, calib.K2, calib.D2, calib.image_size, calib.R, calib.T, alpha=0
    )
    calib.map1x, calib.map1y = cv.initUndistortRectifyMap(
        calib.K1, calib.D1, R1, P1, calib.image_size, cv.CV_32FC1
    )
    calib.map2x, calib.map2y = cv.initUndistortRectifyMap(
        calib.K2, calib.D2, R2, P2, calib.image_size, cv.CV_32FC1
    )
    calib.Q = Q


def rectify_pair(left: np.ndarray, right: np.ndarray, calib: Optional[StereoCalibration]) -> Tuple[np.ndarray, np.ndarray]:
    if calib is None or calib.map1x is None or calib.map2x is None:
        return left, right
    left_r = cv.remap(left, calib.map1x, calib.map1y, cv.INTER_LINEAR)
    right_r = cv.remap(right, calib.map2x, calib.map2y, cv.INTER_LINEAR)
    return left_r, right_r


def compare_maps(sgbm_disp: np.ndarray, nn_rel: np.ndarray) -> Tuple[float, float]:
    """Compare after robust min-max normalization. Both are relative closeness-like maps."""
    valid = np.isfinite(sgbm_disp) & (sgbm_disp > 1)
    if valid.sum() < 100:
        return float("nan"), float("nan")
    a = normalize_to_u8(sgbm_disp, invalid_mask=~valid).astype(np.float32) / 255.0
    b = normalize_to_u8(nn_rel).astype(np.float32) / 255.0
    av = a[valid].reshape(-1)
    bv = b[valid].reshape(-1)
    mae = float(np.mean(np.abs(av - bv)))
    if np.std(av) < 1e-6 or np.std(bv) < 1e-6:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(av, bv)[0, 1])
    return mae, corr


def draw_text(img: np.ndarray, text: str, org=(12, 28)) -> np.ndarray:
    out = img.copy()
    cv.putText(out, text, org, cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv.LINE_AA)
    cv.putText(out, text, org, cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1, cv.LINE_AA)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to side-by-side stereo video")
    parser.add_argument("--calibrate", action="store_true", help="Calibrate from the same chessboard video")
    parser.add_argument("--calib-file", default="stereo_calib.npz", help="Calibration npz file")
    parser.add_argument("--pattern-cols", type=int, default=9, help="Number of inner chessboard corners along columns")
    parser.add_argument("--pattern-rows", type=int, default=6, help="Number of inner chessboard corners along rows")
    parser.add_argument("--square-mm", type=float, default=35.0, help="Chessboard square size in mm")
    parser.add_argument("--baseline-cm", type=float, default=17.0, help="Stereo baseline in cm")
    parser.add_argument("--swap", action="store_true", help="Swap left/right halves; readme command used -3dswap")
    parser.add_argument("--frame-step", type=int, default=15, help="Use every Nth frame during calibration/processing")
    parser.add_argument("--max-calib-pairs", type=int, default=35)
    parser.add_argument("--no-neural", action="store_true", help="Disable neural stereo and show only SGBM")
    parser.add_argument("--psmnet-repo", default="PSMNet", help="Path to cloned PSMNet repository")
    parser.add_argument("--psmnet-weights", default="psmnet_sceneflow_torch18.tar", help="Path to PSMNet .tar checkpoint")
    parser.add_argument("--psmnet-model", default="stackhourglass", choices=["stackhourglass", "basic"], help="PSMNet model architecture")
    parser.add_argument("--psmnet-maxdisp", type=int, default=192, help="Maximum disparity used by PSMNet")
    parser.add_argument("--setup-psmnet", action="store_true", help="Clone official PSMNet repo and download default weights with gdown, then exit")
    parser.add_argument("--no-download-weights", action="store_true", help="During --setup-psmnet, clone repo but do not download weights")
    parser.add_argument("--out", default="depth_compare.mp4", help="Optional output video path")
    args = parser.parse_args()

    calib: Optional[StereoCalibration] = None
    if args.setup_psmnet:
        NeuralStereoPSMNet.setup(args.psmnet_repo, args.psmnet_weights, download_weights=not args.no_download_weights)
        print("PSMNet setup finished.")
        return

    if args.calibrate:
        calib = calibrate_from_video(
            args.video,
            (args.pattern_cols, args.pattern_rows),
            args.square_mm / 1000.0,
            args.max_calib_pairs,
            args.frame_step,
            args.swap,
            args.calib_file,
        )
    elif os.path.exists(args.calib_file):
        calib = load_calibration(args.calib_file)
        print(f"Loaded calibration from {args.calib_file}")
    else:
        print("No calibration file found: SGBM will run without rectification and metric depth will be approximate/disabled.")

    cap = cv.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(args.video)

    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Cannot read first video frame")
    left0, right0 = split_sbs(frame, swap=args.swap)
    left0, right0 = rectify_pair(left0, right0, calib)
    h, w = left0.shape[:2]
    stereo = make_sgbm(w)

    focal_px = None
    if calib is not None:
        focal_px = float(calib.K1[0, 0])
    baseline_m = args.baseline_cm / 100.0

    neural = None
    if not args.no_neural:
        neural = NeuralStereoPSMNet(
            repo_path=args.psmnet_repo,
            weights_path=args.psmnet_weights,
            maxdisp=args.psmnet_maxdisp,
            model_name=args.psmnet_model,
        )
        print(f"Loaded PSMNet on {neural.device}")

    fps = cap.get(cv.CAP_PROP_FPS) or 30.0
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    writer = cv.VideoWriter(args.out, fourcc, max(1.0, fps / max(1, args.frame_step)), (w * 2, h))

    cap.set(cv.CAP_PROP_POS_FRAMES, 0)
    frame_idx = 0
    metrics_mae, metrics_corr = [], []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % args.frame_step != 0:
            frame_idx += 1
            continue

        left, right = split_sbs(frame, swap=args.swap)
        left, right = rectify_pair(left, right, calib)
        disp, sgbm_color, depth_m = compute_sgbm_depth_vis(left, right, stereo, focal_px, baseline_m)
        sgbm_panel = draw_text(sgbm_color, "Classic SGBM: disparity / relative closeness")

        if neural is not None:
            nn_disp, nn_color = neural.predict_disparity(left, right)
            mae, corr = compare_maps(disp, nn_disp)
            if np.isfinite(mae): metrics_mae.append(mae)
            if np.isfinite(corr): metrics_corr.append(corr)
            label = f"Neural PSMNet: disparity | MAE={mae:.3f}, corr={corr:.3f}"
            nn_panel = draw_text(nn_color, label)
        else:
            nn_panel = draw_text(left, "Neural stereo disabled")

        combined = np.hstack([sgbm_panel, nn_panel])
        cv.imshow("Depth comparison: SGBM vs PSMNet", combined)
        writer.write(combined)

        key = cv.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
        frame_idx += 1

    cap.release()
    writer.release()
    cv.destroyAllWindows()

    if metrics_mae:
        print(f"Average normalized MAE: {np.mean(metrics_mae):.4f}")
    if metrics_corr:
        print(f"Average Pearson correlation: {np.mean(metrics_corr):.4f}")
    print(f"Saved comparison video: {args.out}")


if __name__ == "__main__":
    main()