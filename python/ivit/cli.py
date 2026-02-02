"""
iVIT-SDK Command Line Interface.

Usage:
    ivit info              Show system and SDK information
    ivit devices           List available inference devices
    ivit benchmark         Run performance benchmark
    ivit predict           Run inference on image/video
    ivit convert           Convert model format
    ivit export            Export model with optimizations
    ivit serve             Start inference server

Examples:
    $ ivit info
    $ ivit devices
    $ ivit benchmark models/yolov8n.onnx --device cuda:0
    $ ivit predict models/yolov8n.onnx image.jpg --conf 0.5
    $ ivit convert models/yolov8n.onnx -f openvino -o models/
    $ ivit serve models/yolov8n.onnx --port 8080
"""

import argparse
import sys
import time
from pathlib import Path


def cmd_info(args):
    """Show system and SDK information."""
    import ivit
    import platform

    print("=" * 60)
    print("iVIT-SDK System Information")
    print("=" * 60)
    print()

    # SDK Info
    print("SDK:")
    print(f"  Version:        {ivit.__version__}")
    print(f"  C++ Bindings:   {'Available' if ivit.is_cpp_available() else 'Not available'}")
    print()

    # System Info
    print("System:")
    print(f"  Platform:       {platform.system()} {platform.release()}")
    print(f"  Architecture:   {platform.machine()}")
    print(f"  Python:         {platform.python_version()}")
    print()

    # Backend availability
    print("Backends:")
    try:
        import openvino
        print(f"  OpenVINO:       {openvino.__version__}")
    except ImportError:
        print("  OpenVINO:       Not installed")

    try:
        import tensorrt
        print(f"  TensorRT:       {tensorrt.__version__}")
    except ImportError:
        print("  TensorRT:       Not installed")

    print()

    # Devices summary
    print("Devices:")
    device_list = ivit.list_devices()
    for d in device_list:
        print(f"  [{d.id}] {d.name} ({d.backend})")

    print()
    print("=" * 60)


def cmd_devices(args):
    """List available inference devices."""
    import ivit

    device_list = ivit.list_devices()

    if args.json:
        import json
        data = [
            {
                "id": d.id,
                "name": d.name,
                "type": d.type,
                "backend": d.backend,
                "available": d.available,
            }
            for d in device_list
        ]
        print(json.dumps(data, indent=2))
        return

    print()
    print("Available Devices:")
    print("-" * 60)
    print(f"{'ID':<12} {'Name':<30} {'Backend':<12} {'Type':<8}")
    print("-" * 60)

    for d in device_list:
        status = "OK" if d.available else "N/A"
        print(f"{d.id:<12} {d.name[:28]:<30} {d.backend:<12} {d.type:<8}")

    print("-" * 60)
    print(f"Total: {len(device_list)} device(s)")
    print()

    # Show best device
    best = ivit.devices.best()
    print(f"Best device (performance): {best.id}")
    best_eff = ivit.devices.best("efficiency")
    print(f"Best device (efficiency):  {best_eff.id}")
    print()


def cmd_benchmark(args):
    """Run performance benchmark."""
    import ivit
    import numpy as np

    model_path = args.model
    device = args.device
    iterations = args.iterations
    warmup = args.warmup

    print()
    print("=" * 60)
    print("iVIT-SDK Benchmark")
    print("=" * 60)
    print()
    print(f"Model:      {model_path}")
    print(f"Device:     {device}")
    print(f"Iterations: {iterations}")
    print(f"Warmup:     {warmup}")
    print()

    # Load model
    print("Loading model...", end=" ", flush=True)
    start = time.time()
    model = ivit.load(model_path, device=device)
    load_time = time.time() - start
    print(f"Done ({load_time:.2f}s)")
    print()

    # Get input shape
    input_shape = model.input_info[0]["shape"]
    print(f"Input shape: {input_shape}")
    print()

    # Create dummy input
    if len(input_shape) == 4:
        _, c, h, w = input_shape
        dummy_image = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    else:
        dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    # Warmup
    print(f"Warming up ({warmup} iterations)...", end=" ", flush=True)
    for _ in range(warmup):
        model.predict(dummy_image)
    print("Done")
    print()

    # Benchmark
    print(f"Benchmarking ({iterations} iterations)...")
    latencies = []

    for i in range(iterations):
        start = time.perf_counter()
        results = model.predict(dummy_image)
        latency = (time.perf_counter() - start) * 1000
        latencies.append(latency)

        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{iterations} completed")

    print()

    # Statistics
    latencies = np.array(latencies)
    print("-" * 40)
    print("Results:")
    print("-" * 40)
    print(f"  Mean:     {np.mean(latencies):.2f} ms")
    print(f"  Std:      {np.std(latencies):.2f} ms")
    print(f"  Min:      {np.min(latencies):.2f} ms")
    print(f"  Max:      {np.max(latencies):.2f} ms")
    print(f"  P50:      {np.percentile(latencies, 50):.2f} ms")
    print(f"  P90:      {np.percentile(latencies, 90):.2f} ms")
    print(f"  P99:      {np.percentile(latencies, 99):.2f} ms")
    print(f"  FPS:      {1000 / np.mean(latencies):.1f}")
    print("-" * 40)
    print()


def cmd_predict(args):
    """Run inference on image/video."""
    import ivit

    model_path = args.model
    source = args.source
    device = args.device
    conf = args.conf
    iou = args.iou
    save = args.save
    show = args.show

    print()
    print(f"Model:  {model_path}")
    print(f"Source: {source}")
    print(f"Device: {device}")
    print()

    # Load model
    print("Loading model...", end=" ", flush=True)
    model = ivit.load(model_path, device=device)
    print("Done")
    print()

    # Check if source is video
    source_path = Path(source)
    is_video = source_path.suffix.lower() in ('.mp4', '.avi', '.mov', '.mkv')
    is_camera = source.isdigit()

    if is_video or is_camera:
        # Stream inference
        src = int(source) if is_camera else source
        print(f"Streaming inference...")
        print("-" * 40)

        frame_count = 0
        for results in model.stream(src, conf=conf, iou=iou, show=show, save=save):
            frame_count += 1
            print(f"Frame {frame_count}: {len(results)} detections, {results.inference_time_ms:.1f}ms")

        print("-" * 40)
        print(f"Total frames: {frame_count}")
    else:
        # Image inference
        print("Running inference...")
        results = model.predict(source, conf=conf, iou=iou)

        print()
        print(f"Inference time: {results.inference_time_ms:.2f} ms")
        print(f"Detections: {len(results)}")
        print()

        for i, det in enumerate(results.detections):
            print(f"  [{i}] {det.label}: {det.confidence:.2%} @ [{det.bbox.x1:.0f}, {det.bbox.y1:.0f}, {det.bbox.x2:.0f}, {det.bbox.y2:.0f}]")

        if save:
            output_path = source_path.stem + "_result" + source_path.suffix
            results.save(output_path)
            print()
            print(f"Saved to: {output_path}")

        if show:
            results.show()

    print()


def cmd_convert(args):
    """Convert model to different format."""
    import shutil

    model_path = args.model
    output_format = args.format.lower()
    output_dir = Path(args.output) if args.output else Path(".")
    precision = args.precision

    print()
    print("=" * 60)
    print("iVIT-SDK Model Converter")
    print("=" * 60)
    print()
    print(f"Input:    {model_path}")
    print(f"Format:   {output_format}")
    print(f"Output:   {output_dir}")
    print(f"Precision: {precision}")
    print()

    # Verify input exists
    input_path = Path(model_path)
    if not input_path.exists():
        print(f"Error: Model not found: {model_path}")
        sys.exit(1)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert based on format
    if output_format == "openvino":
        _convert_to_openvino(input_path, output_dir, precision)
    elif output_format == "tensorrt":
        _convert_to_tensorrt(input_path, output_dir, precision)
    elif output_format == "onnx":
        _convert_to_onnx(input_path, output_dir)
    else:
        print(f"Error: Unsupported format: {output_format}")
        print("Supported formats: openvino, tensorrt, onnx")
        sys.exit(1)

    print()
    print("=" * 60)


def _convert_to_openvino(input_path: Path, output_dir: Path, precision: str):
    """Convert to OpenVINO IR format."""
    import subprocess
    import shutil

    model_name = input_path.stem
    output_xml = output_dir / f"{model_name}.xml"
    output_bin = output_dir / f"{model_name}.bin"

    # Strategy 1: Try C++ binding (works with APT-installed OpenVINO)
    try:
        from ivit._ivit_core import convert_model

        print("Converting to OpenVINO IR format...")
        if precision.lower() == "fp16":
            print("  Compressing to FP16...")

        convert_model(str(input_path), str(output_xml), "cpu", precision.lower())

        print(f"  Created: {output_xml}")
        print(f"  Created: {output_bin}")
        print()
        print("Conversion complete!")
        return
    except ImportError:
        pass

    # Strategy 2: Try ovc command-line tool (from APT or pip openvino-dev)
    ovc_path = shutil.which("ovc")
    if ovc_path:
        print("Converting to OpenVINO IR format (using ovc)...")

        cmd = [ovc_path, str(input_path), "--output_model", str(output_xml)]
        if precision.lower() == "fp16":
            print("  Compressing to FP16...")
            cmd.append("--compress_to_fp16")

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error: ovc conversion failed:\n{result.stderr}")
            sys.exit(1)

        print(f"  Created: {output_xml}")
        print(f"  Created: {output_bin}")
        print()
        print("Conversion complete!")
        return

    # No conversion tool available
    print("Error: OpenVINO conversion tools not found.")
    print("Install one of the following:")
    print("  pip install -e .                          # Build C++ binding")
    print("  sudo apt install openvino-tools           # APT ovc tool")
    sys.exit(1)


def _convert_to_tensorrt(input_path: Path, output_dir: Path, precision: str):
    """Convert to TensorRT engine."""
    model_name = input_path.stem
    output_engine = output_dir / f"{model_name}.engine"

    # Strategy 1: Try C++ binding
    try:
        from ivit._ivit_core import convert_model

        print("Converting to TensorRT engine...")
        print(f"  Precision: {precision}")

        convert_model(str(input_path), str(output_engine), "cuda:0", precision.lower())

        print(f"  Created: {output_engine}")
        print()
        print("Conversion complete!")
        return
    except ImportError:
        pass

    # Strategy 2: Fall back to Python tensorrt
    try:
        import tensorrt as trt
    except ImportError:
        print("Error: TensorRT not installed.")
        print("TensorRT requires NVIDIA GPU and proper installation.")
        sys.exit(1)

    print("Converting to TensorRT engine...")
    print(f"  Precision: {precision}")

    # TensorRT logger
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    # Build engine
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Parse ONNX
    print("  Parsing ONNX model...")
    with open(input_path, 'rb') as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(f"  Error: {parser.get_error(error)}")
            sys.exit(1)

    # Build config
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

    if precision.lower() == "fp16":
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("  FP16 enabled")
        else:
            print("  Warning: FP16 not supported on this platform")

    if precision.lower() == "int8":
        if builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            print("  INT8 enabled (requires calibration)")
        else:
            print("  Warning: INT8 not supported on this platform")

    # Build engine
    print("  Building engine (this may take a while)...")
    engine_bytes = builder.build_serialized_network(network, config)

    if engine_bytes is None:
        print("Error: Failed to build engine")
        sys.exit(1)

    with open(output_engine, 'wb') as f:
        f.write(engine_bytes)

    print(f"  Created: {output_engine}")
    print()
    print("Conversion complete!")


def _convert_to_onnx(input_path: Path, output_dir: Path):
    """Convert PyTorch model to ONNX."""
    if input_path.suffix.lower() == '.onnx':
        # Already ONNX, just copy
        import shutil
        output_path = output_dir / input_path.name
        shutil.copy(input_path, output_path)
        print(f"  Copied to: {output_path}")
        return

    if input_path.suffix.lower() == '.pt':
        try:
            from ultralytics import YOLO
            print("Converting PyTorch model to ONNX...")
            model = YOLO(str(input_path))
            model.export(format="onnx")
            # Move to output dir
            onnx_path = input_path.with_suffix('.onnx')
            if onnx_path.exists():
                import shutil
                output_path = output_dir / onnx_path.name
                shutil.move(str(onnx_path), str(output_path))
                print(f"  Created: {output_path}")
        except ImportError:
            print("Error: Ultralytics required for .pt conversion")
            print("Install with: pip install ultralytics")
            sys.exit(1)
    else:
        print(f"Error: Unsupported input format: {input_path.suffix}")
        sys.exit(1)

    print()
    print("Conversion complete!")


def cmd_export(args):
    """Export model with optimizations for target platform."""
    import ivit

    model_path = args.model
    output_path = args.output
    target = args.target
    precision = args.precision

    print()
    print("=" * 60)
    print("iVIT-SDK Model Export")
    print("=" * 60)
    print()
    print(f"Input:    {model_path}")
    print(f"Output:   {output_path}")
    print(f"Target:   {target}")
    print(f"Precision: {precision}")
    print()

    # Load model first to validate
    print("Loading model...", end=" ", flush=True)
    model = ivit.load(model_path)
    print("Done")
    print()

    # Determine output format based on target
    target_formats = {
        "intel_cpu": "openvino",
        "intel_gpu": "openvino",
        "intel_npu": "openvino",
        "nvidia_gpu": "tensorrt",
        "cpu": "onnx",
        "auto": "onnx",
    }

    output_format = target_formats.get(target, "onnx")
    print(f"Export format: {output_format}")

    # Convert
    input_path = Path(model_path)
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    if output_format == "openvino":
        _convert_to_openvino(input_path, output_dir, precision)
    elif output_format == "tensorrt":
        _convert_to_tensorrt(input_path, output_dir, precision)
    else:
        # Just copy ONNX
        import shutil
        shutil.copy(model_path, output_path)
        print(f"Exported to: {output_path}")

    print()
    print("=" * 60)


def cmd_serve(args):
    """Start inference server."""
    import ivit
    import json
    from http.server import HTTPServer, BaseHTTPRequestHandler
    import io
    import base64

    model_path = args.model
    device = args.device
    port = args.port
    host = args.host

    print()
    print("=" * 60)
    print("iVIT-SDK Inference Server")
    print("=" * 60)
    print()
    print(f"Model:  {model_path}")
    print(f"Device: {device}")
    print(f"Host:   {host}")
    print(f"Port:   {port}")
    print()

    # Load model
    print("Loading model...", end=" ", flush=True)
    model = ivit.load(model_path, device=device)
    print("Done")
    print()

    # Create request handler
    class InferenceHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/health":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "ok"}).encode())
            elif self.path == "/info":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                info = {
                    "model": model.name,
                    "device": model.device,
                    "task": str(model.task),
                    "input_shape": model.input_info[0]["shape"] if model.input_info else [],
                }
                self.wfile.write(json.dumps(info).encode())
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            if self.path == "/predict":
                try:
                    import numpy as np
                    import cv2

                    content_length = int(self.headers['Content-Length'])
                    post_data = self.rfile.read(content_length)

                    # Try to parse as JSON with base64 image
                    try:
                        data = json.loads(post_data)
                        if "image" in data:
                            # Base64 encoded image
                            img_data = base64.b64decode(data["image"])
                            nparr = np.frombuffer(img_data, np.uint8)
                            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        else:
                            raise ValueError("No image in request")
                    except json.JSONDecodeError:
                        # Raw image bytes
                        nparr = np.frombuffer(post_data, np.uint8)
                        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                    if image is None:
                        raise ValueError("Failed to decode image")

                    # Run inference
                    conf = float(self.headers.get('X-Conf-Threshold', 0.5))
                    results = model.predict(image, conf=conf)

                    # Format response
                    response = {
                        "inference_time_ms": results.inference_time_ms,
                        "detections": [
                            {
                                "label": d.label,
                                "confidence": float(d.confidence),
                                "bbox": [d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2],
                            }
                            for d in results.detections
                        ]
                    }

                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps(response).encode())

                except Exception as e:
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": str(e)}).encode())
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, format, *args):
            print(f"[{self.log_date_time_string()}] {format % args}")

    # Start server
    server = HTTPServer((host, port), InferenceHandler)
    print(f"Server started at http://{host}:{port}")
    print()
    print("Endpoints:")
    print(f"  GET  /health  - Health check")
    print(f"  GET  /info    - Model information")
    print(f"  POST /predict - Run inference (send image as raw bytes or JSON with base64)")
    print()
    print("Press Ctrl+C to stop")
    print("-" * 60)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
        print("Server stopped")
        server.shutdown()


def cmd_zoo(args):
    """Model Zoo commands."""
    import ivit

    if args.zoo_command == "list":
        task = args.task
        from ivit.zoo.registry import print_models
        print_models(task)
    elif args.zoo_command == "search":
        results = ivit.zoo.search(args.query)
        if results:
            print(f"\nFound {len(results)} model(s):")
            for name in results:
                info = ivit.zoo.get_model_info(name)
                print(f"  {name:<20} {info.task:<12} {info.description[:40]}")
        else:
            print(f"No models found for: {args.query}")
    elif args.zoo_command == "info":
        try:
            info = ivit.zoo.get_model_info(args.name)
            print()
            print(f"Model: {info.name}")
            print(f"Task: {info.task}")
            print(f"Description: {info.description}")
            print(f"Input size: {info.input_size}")
            print(f"Classes: {info.num_classes}")
            print(f"Formats: {', '.join(info.formats)}")
            print(f"Source: {info.source}")
            if info.metrics:
                print(f"Metrics: {info.metrics}")
            if info.tags:
                print(f"Tags: {', '.join(info.tags)}")
            print()
        except KeyError as e:
            print(f"Error: {e}")
    elif args.zoo_command == "download":
        print(f"Downloading {args.name}...")
        path = ivit.zoo.download(args.name, format=args.format)
        print(f"Downloaded to: {path}")


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        prog="ivit",
        description="iVIT-SDK Command Line Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ivit info                                    Show system info
  ivit devices                                 List devices
  ivit devices --json                          List devices as JSON
  ivit benchmark model.onnx -d cuda:0          Benchmark on GPU
  ivit predict model.onnx image.jpg            Run inference
  ivit predict model.onnx video.mp4 --save     Process video
  ivit convert model.onnx -f openvino          Convert to OpenVINO
  ivit convert model.onnx -f tensorrt -p fp16  Convert to TensorRT FP16
  ivit export model.onnx -o out.xml -t intel_gpu  Export for Intel GPU
  ivit serve model.onnx --port 8080            Start inference server
  ivit zoo list                                List Model Zoo models
  ivit zoo search yolo                         Search models
  ivit zoo download yolov8n                    Download model
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # info command
    info_parser = subparsers.add_parser("info", help="Show system information")
    info_parser.set_defaults(func=cmd_info)

    # devices command
    devices_parser = subparsers.add_parser("devices", help="List available devices")
    devices_parser.add_argument("--json", action="store_true", help="Output as JSON")
    devices_parser.set_defaults(func=cmd_devices)

    # benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Run performance benchmark")
    bench_parser.add_argument("model", help="Model path")
    bench_parser.add_argument("-d", "--device", default="auto", help="Device (default: auto)")
    bench_parser.add_argument("-n", "--iterations", type=int, default=100, help="Iterations (default: 100)")
    bench_parser.add_argument("-w", "--warmup", type=int, default=10, help="Warmup iterations (default: 10)")
    bench_parser.set_defaults(func=cmd_benchmark)

    # predict command
    predict_parser = subparsers.add_parser("predict", help="Run inference")
    predict_parser.add_argument("model", help="Model path")
    predict_parser.add_argument("source", help="Image/video path or camera index")
    predict_parser.add_argument("-d", "--device", default="auto", help="Device (default: auto)")
    predict_parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold (default: 0.5)")
    predict_parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold (default: 0.45)")
    predict_parser.add_argument("--save", action="store_true", help="Save results")
    predict_parser.add_argument("--show", action="store_true", help="Display results")
    predict_parser.set_defaults(func=cmd_predict)

    # convert command
    convert_parser = subparsers.add_parser("convert", help="Convert model format")
    convert_parser.add_argument("model", help="Input model path")
    convert_parser.add_argument("-f", "--format", required=True, choices=["openvino", "tensorrt", "onnx"],
                                help="Output format")
    convert_parser.add_argument("-o", "--output", default=".", help="Output directory (default: .)")
    convert_parser.add_argument("-p", "--precision", default="fp32", choices=["fp32", "fp16", "int8"],
                                help="Precision (default: fp32)")
    convert_parser.set_defaults(func=cmd_convert)

    # export command
    export_parser = subparsers.add_parser("export", help="Export model for target platform")
    export_parser.add_argument("model", help="Input model path")
    export_parser.add_argument("-o", "--output", required=True, help="Output model path")
    export_parser.add_argument("-t", "--target", default="auto",
                               choices=["auto", "cpu", "intel_cpu", "intel_gpu", "intel_npu", "nvidia_gpu"],
                               help="Target platform (default: auto)")
    export_parser.add_argument("-p", "--precision", default="fp32", choices=["fp32", "fp16", "int8"],
                               help="Precision (default: fp32)")
    export_parser.set_defaults(func=cmd_export)

    # serve command
    serve_parser = subparsers.add_parser("serve", help="Start inference server")
    serve_parser.add_argument("model", help="Model path")
    serve_parser.add_argument("-d", "--device", default="auto", help="Device (default: auto)")
    serve_parser.add_argument("--host", default="127.0.0.1",
                               help="Host (default: 127.0.0.1, use 0.0.0.0 for public access)")
    serve_parser.add_argument("--port", type=int, default=8080, help="Port (default: 8080)")
    serve_parser.set_defaults(func=cmd_serve)

    # zoo command
    zoo_parser = subparsers.add_parser("zoo", help="Model Zoo operations")
    zoo_subparsers = zoo_parser.add_subparsers(dest="zoo_command", help="Zoo commands")

    zoo_list_parser = zoo_subparsers.add_parser("list", help="List available models")
    zoo_list_parser.add_argument("-t", "--task", choices=["detect", "classify", "segment", "pose"],
                                 help="Filter by task")

    zoo_search_parser = zoo_subparsers.add_parser("search", help="Search models")
    zoo_search_parser.add_argument("query", help="Search query")

    zoo_info_parser = zoo_subparsers.add_parser("info", help="Show model info")
    zoo_info_parser.add_argument("name", help="Model name")

    zoo_download_parser = zoo_subparsers.add_parser("download", help="Download model")
    zoo_download_parser.add_argument("name", help="Model name")
    zoo_download_parser.add_argument("-f", "--format", default="onnx", help="Format (default: onnx)")

    zoo_parser.set_defaults(func=cmd_zoo)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
