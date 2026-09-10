from __future__ import annotations

import argparse
import queue
import threading
from pathlib import Path
from typing import Optional, Sequence

import torch

from src.codec.frame_decoder import (
    BitstreamFrameSource,
    DecodedFrame,
    DecoderModels,
    configure_decode_runtime,
    load_decoder_models,
    resolve_decode_device,
)
from src.utils.common import str2bool
from src.utils.stream_helper import NalType


def discover_intra_images(folder: str | Path, recursive: bool = False) -> list[Path]:
    folder = Path(folder)
    candidates = folder.rglob("*") if recursive else folder.iterdir()
    return sorted(
        path for path in candidates
        if path.is_file() and path.suffix.lower() == ".dcvci"
    )


def _frame_photo(frame: DecodedFrame, max_width: int, max_height: int):
    from PIL import Image, ImageTk

    image = Image.fromarray(frame.rgb)
    ratio = min(max_width / frame.width, max_height / frame.height, 1.0)
    if ratio < 1.0:
        size = (max(1, int(frame.width * ratio)), max(1, int(frame.height * ratio)))
        resample = getattr(Image, "Resampling", Image).BILINEAR
        image = image.resize(size, resample)
    return ImageTk.PhotoImage(image)


class DecodeWorker:
    def __init__(self, source: BitstreamFrameSource):
        self.source = source
        self.requests: queue.Queue = queue.Queue()
        self.results: queue.Queue = queue.Queue()
        self.thread = threading.Thread(target=self._run, name="dcvc-viewer-decode", daemon=True)
        self.thread.start()

    def request(self, request_id: int, frame_index: int):
        self.clear_pending()
        self.requests.put(("frame", request_id, frame_index))

    def close(self):
        self.clear_pending()
        self.requests.put(("close", None, None))

    def clear_pending(self):
        while True:
            try:
                self.requests.get_nowait()
            except queue.Empty:
                return

    def _run(self):
        while True:
            kind, request_id, frame_index = self.requests.get()
            if kind == "close":
                self.source.close()
                return
            try:
                frame = self.source.seek(frame_index, output_format="rgb")
                self.results.put(("frame", request_id, frame))
            except Exception as exc:  # pylint: disable=broad-exception-caught
                self.results.put(("error", request_id, str(exc)))


class ImageGalleryWorker:
    def __init__(self, models: DecoderModels):
        self.models = models
        self.requests: queue.Queue = queue.Queue()
        self.results: queue.Queue = queue.Queue()
        self.thread = threading.Thread(target=self._run, name="dcvc-image-viewer-decode", daemon=True)
        self.thread.start()

    def request(self, request_id: int, image_path: Path):
        self.clear_pending()
        self.requests.put(("image", request_id, image_path))

    def close(self):
        self.clear_pending()
        self.requests.put(("close", None, None))

    def clear_pending(self):
        while True:
            try:
                self.requests.get_nowait()
            except queue.Empty:
                return

    def _run(self):
        while True:
            kind, request_id, image_path = self.requests.get()
            if kind == "close":
                return
            source = None
            try:
                source = BitstreamFrameSource(image_path, self.models)
                if source.index.frame_count != 1 or source.index.frames[0].nal_type != NalType.NAL_I:
                    raise ValueError(".dcvci must contain exactly one I-frame")
                frame = source.seek(0, output_format="rgb")
                self.results.put(("image", request_id, (image_path, frame)))
            except Exception as exc:  # pylint: disable=broad-exception-caught
                self.results.put(("error", request_id, str(exc)))
            finally:
                if source is not None:
                    source.close()


class ViewerApp:
    def __init__(self, root, source: BitstreamFrameSource, fps: float, start_frame: int,
                 max_width: int, max_height: int, runtime_label: str):
        import tkinter as tk

        self.tk = tk
        self.root = root
        self.source = source
        self.worker = DecodeWorker(source)
        self.fps = max(0.1, fps)
        self.max_width = max_width
        self.max_height = max_height
        self.runtime_label = runtime_label
        self.current_frame = max(0, min(start_frame, source.index.frame_count - 1))
        self.request_id = 0
        self.playing = False
        self.photo = None
        self.play_after_id = None

        root.title(f"DCVC-RT Viewer - {source.index.path.name}")
        root.protocol("WM_DELETE_WINDOW", self.close)

        self.image_label = tk.Label(root, bg="black")
        self.image_label.pack(fill=tk.BOTH, expand=True)

        controls = tk.Frame(root)
        controls.pack(fill=tk.X, padx=8, pady=8)

        self.play_button = tk.Button(controls, text="Play", width=8, command=self.toggle_play)
        self.play_button.pack(side=tk.LEFT, padx=(0, 6))

        tk.Button(controls, text="Prev", width=8, command=self.previous_frame).pack(side=tk.LEFT, padx=(0, 6))
        tk.Button(controls, text="Next", width=8, command=self.next_frame).pack(side=tk.LEFT, padx=(0, 10))

        self.scale = tk.Scale(
            controls,
            from_=0,
            to=max(0, source.index.frame_count - 1),
            orient=tk.HORIZONTAL,
            showvalue=False,
        )
        self.scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.scale.bind("<ButtonRelease-1>", self.on_slider_release)

        self.frame_label = tk.Label(controls, width=18, anchor="e")
        self.frame_label.pack(side=tk.LEFT, padx=(10, 0))

        meta = tk.Frame(root)
        meta.pack(fill=tk.X, padx=8, pady=(0, 8))
        self.status_label = tk.Label(meta, anchor="w")
        self.status_label.pack(fill=tk.X)

        self.set_status("Loading frame...")
        self.request_frame(self.current_frame)
        self.root.after(30, self.poll_results)

    def request_frame(self, frame_index: int):
        if self.source.index.frame_count <= 0:
            self.set_status("No frames in bitstream")
            return
        frame_index = max(0, min(frame_index, self.source.index.frame_count - 1))
        self.current_frame = frame_index
        self.scale.set(frame_index)
        self.update_frame_label()
        self.request_id += 1
        self.worker.request(self.request_id, frame_index)

    def poll_results(self):
        while True:
            try:
                kind, request_id, payload = self.worker.results.get_nowait()
            except queue.Empty:
                break
            if request_id != self.request_id:
                continue
            if kind == "error":
                self.playing = False
                self.play_button.configure(text="Play")
                self.set_status(payload)
                continue
            self.display_frame(payload)
        self.root.after(30, self.poll_results)

    def display_frame(self, frame: DecodedFrame):
        self.photo = _frame_photo(frame, self.max_width, self.max_height)
        self.image_label.configure(image=self.photo)
        self.current_frame = frame.index
        self.scale.set(frame.index)
        self.update_frame_label()
        self.set_status(
            f"{self.source.index.path.name} | {frame.width}x{frame.height} | "
            f"{frame.frame_type}-frame | QP {frame.qp} | {self.runtime_label}"
        )

        if self.playing:
            self.schedule_next_play_frame()

    def toggle_play(self):
        self.playing = not self.playing
        self.play_button.configure(text="Pause" if self.playing else "Play")
        if self.playing:
            self.schedule_next_play_frame()
        elif self.play_after_id is not None:
            self.root.after_cancel(self.play_after_id)
            self.play_after_id = None

    def schedule_next_play_frame(self):
        if self.play_after_id is not None:
            self.root.after_cancel(self.play_after_id)
        delay_ms = max(1, int(1000 / self.fps))
        self.play_after_id = self.root.after(delay_ms, self.next_play_frame)

    def next_play_frame(self):
        self.play_after_id = None
        if not self.playing:
            return
        if self.current_frame + 1 >= self.source.index.frame_count:
            self.playing = False
            self.play_button.configure(text="Play")
            return
        self.request_frame(self.current_frame + 1)

    def previous_frame(self):
        self.playing = False
        self.play_button.configure(text="Play")
        self.request_frame(self.current_frame - 1)

    def next_frame(self):
        self.playing = False
        self.play_button.configure(text="Play")
        self.request_frame(self.current_frame + 1)

    def on_slider_release(self, _event):
        self.playing = False
        self.play_button.configure(text="Play")
        self.request_frame(int(self.scale.get()))

    def update_frame_label(self):
        total = self.source.index.frame_count
        self.frame_label.configure(text=f"{self.current_frame + 1}/{total}")

    def set_status(self, text: str):
        self.status_label.configure(text=text)

    def close(self):
        if self.play_after_id is not None:
            self.root.after_cancel(self.play_after_id)
        self.worker.close()
        self.root.destroy()


class ImageGalleryApp:
    def __init__(self, root, image_paths: Sequence[Path], models: DecoderModels, start_image: int,
                 max_width: int, max_height: int, runtime_label: str):
        import tkinter as tk

        self.tk = tk
        self.root = root
        self.image_paths = list(image_paths)
        self.worker = ImageGalleryWorker(models)
        self.max_width = max_width
        self.max_height = max_height
        self.runtime_label = runtime_label
        self.current_image = max(0, min(start_image, len(self.image_paths) - 1))
        self.request_id = 0
        self.photo = None

        root.title("DCVC-RT Intra Image Viewer")
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.bind("<Left>", lambda _event: self.previous_image())
        root.bind("<Right>", lambda _event: self.next_image())
        root.bind("<Home>", lambda _event: self.request_image(0))
        root.bind("<End>", lambda _event: self.request_image(len(self.image_paths) - 1))

        self.image_label = tk.Label(root, bg="black")
        self.image_label.pack(fill=tk.BOTH, expand=True)

        controls = tk.Frame(root)
        controls.pack(fill=tk.X, padx=8, pady=8)
        tk.Button(controls, text="Prev Image", width=10, command=self.previous_image).pack(
            side=tk.LEFT, padx=(0, 6)
        )
        tk.Button(controls, text="Next Image", width=10, command=self.next_image).pack(
            side=tk.LEFT, padx=(0, 10)
        )
        self.scale = tk.Scale(
            controls,
            from_=0,
            to=max(0, len(self.image_paths) - 1),
            orient=tk.HORIZONTAL,
            showvalue=False,
        )
        self.scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.scale.bind("<ButtonRelease-1>", self.on_slider_release)
        self.image_count_label = tk.Label(controls, width=18, anchor="e")
        self.image_count_label.pack(side=tk.LEFT, padx=(10, 0))

        meta = tk.Frame(root)
        meta.pack(fill=tk.X, padx=8, pady=(0, 8))
        self.status_label = tk.Label(meta, anchor="w")
        self.status_label.pack(fill=tk.X)

        self.set_status("Loading image...")
        self.request_image(self.current_image)
        self.root.after(30, self.poll_results)

    def request_image(self, image_index: int):
        image_index = max(0, min(image_index, len(self.image_paths) - 1))
        self.current_image = image_index
        self.scale.set(image_index)
        self.update_image_count()
        self.request_id += 1
        self.set_status(f"Loading {self.image_paths[image_index].name}...")
        self.worker.request(self.request_id, self.image_paths[image_index])

    def poll_results(self):
        while True:
            try:
                kind, request_id, payload = self.worker.results.get_nowait()
            except queue.Empty:
                break
            if request_id != self.request_id:
                continue
            if kind == "error":
                self.set_status(payload)
                continue
            image_path, frame = payload
            self.display_image(image_path, frame)
        self.root.after(30, self.poll_results)

    def display_image(self, image_path: Path, frame: DecodedFrame):
        self.photo = _frame_photo(frame, self.max_width, self.max_height)
        self.image_label.configure(image=self.photo)
        self.set_status(
            f"{image_path.name} | {frame.width}x{frame.height} | "
            f"I-frame | QP {frame.qp} | {self.runtime_label}"
        )

    def previous_image(self):
        self.request_image(self.current_image - 1)

    def next_image(self):
        self.request_image(self.current_image + 1)

    def on_slider_release(self, _event):
        self.request_image(int(self.scale.get()))

    def update_image_count(self):
        self.image_count_label.configure(
            text=f"{self.current_image + 1}/{len(self.image_paths)} images"
        )

    def set_status(self, text: str):
        self.status_label.configure(text=text)

    def close(self):
        self.worker.close()
        self.root.destroy()


def configure_parser(parser: argparse.ArgumentParser):
    parser.add_argument(
        "bin_path",
        help="DCVC-RT .bin video, .dcvci image, or folder of .dcvci images to view.",
    )
    parser.add_argument("--model_path_i", type=str, default="./checkpoints/cvpr2025_image.pth.tar")
    parser.add_argument("--model_path_p", type=str, default="./checkpoints/cvpr2025_video.pth.tar")
    parser.add_argument("--cuda", type=str2bool, default=True)
    parser.add_argument("--cuda_idx", type=int, default=None)
    parser.add_argument("--force_zero_thres", type=float, default=None)
    parser.add_argument(
        "--runtime",
        choices=("auto", "int16", "float"),
        default="auto",
        help=(
            "Decoder arithmetic. Use int16 for INT16 archives; it fails instead of silently "
            "falling back when the native INT16 runtime is unavailable."
        ),
    )
    parser.add_argument("--recursive", action="store_true", help="Find .dcvci images recursively.")
    parser.add_argument("--fps", type=float, default=30.0, help="Playback FPS.")
    parser.add_argument(
        "--start_frame",
        type=int,
        default=0,
        help="Initial video frame or gallery image index, zero-based.",
    )
    parser.add_argument("--max_width", type=int, default=1280, help="Maximum display width.")
    parser.add_argument("--max_height", type=int, default=720, help="Maximum display height.")


def run(args) -> int:
    bin_path = Path(args.bin_path)
    if not bin_path.exists():
        print(f"Bitstream or image folder not found: {bin_path}")
        return 2
    if bin_path.is_file() and bin_path.suffix.lower() not in {".bin", ".dcvci"}:
        print(f"Unsupported viewer input: {bin_path}")
        return 2
    if bin_path.is_dir():
        image_paths = discover_intra_images(bin_path, args.recursive)
        if not image_paths:
            print(f"No .dcvci images found in {bin_path}")
            return 0
    else:
        image_paths = [bin_path] if bin_path.suffix.lower() == ".dcvci" else []

    try:
        import tkinter as tk
        import PIL.ImageTk  # noqa: F401
    except Exception as exc:  # pylint: disable=broad-exception-caught
        print(f"Viewer requires tkinter and Pillow ImageTk support: {exc}")
        return 2

    if args.cuda and not torch.cuda.is_available():
        print("Warning: --cuda specified but no devices found. Running on CPU.")

    try:
        runtime_label = configure_decode_runtime(args.runtime, args.cuda)
    except ValueError as exc:
        print(f"Error: {exc}")
        return 2
    print(f"Decoder runtime: {runtime_label}")
    if args.runtime == "auto" and "float" in runtime_label.lower():
        print(
            "Warning: auto selected the float decoder. Bitstreams do not identify their "
            "arithmetic runtime; use --runtime int16 for an INT16 archive."
        )

    try:
        root = tk.Tk()
        root.withdraw()
    except Exception as exc:  # pylint: disable=broad-exception-caught
        print(f"Unable to open the viewer window: {exc}")
        return 2

    device = resolve_decode_device(args.cuda, args.cuda_idx)
    models = load_decoder_models(
        args.model_path_i,
        args.model_path_p,
        device,
        args.force_zero_thres,
        load_p_model=not image_paths,
    )
    if image_paths:
        ImageGalleryApp(
            root,
            image_paths,
            models,
            args.start_frame,
            args.max_width,
            args.max_height,
            runtime_label,
        )
    else:
        source = BitstreamFrameSource(bin_path, models)
        if source.index.frame_count == 0:
            print(f"No decodable frames found in {bin_path}")
            source.close()
            root.destroy()
            return 1
        ViewerApp(
            root,
            source,
            args.fps,
            args.start_frame,
            args.max_width,
            args.max_height,
            runtime_label,
        )
    root.deiconify()
    root.mainloop()
    return 0
