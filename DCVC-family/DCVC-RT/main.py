#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys

from src.cli.decode_workflow import configure_parser as configure_decode_parser
from src.cli.decode_workflow import run as run_decode
from src.cli.cleanup_workflow import configure_parser as configure_cleanup_parser
from src.cli.cleanup_workflow import run as run_cleanup
from src.cli.encode_workflow import configure_parser as configure_encode_parser
from src.cli.encode_workflow import run as run_encode
from src.cli.stream_workflow import configure_parser as configure_stream_parser
from src.cli.stream_workflow import run as run_stream
from src.cli.viewer import configure_parser as configure_view_parser
from src.cli.viewer import run as run_view


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DCVC-RT unified encode/decode entrypoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    from src.cli.pareto_search import configure_parser as configure_pareto, run as run_pareto
    pareto_parser = subparsers.add_parser('pareto-search', help='Search INT16 size/PSNR Pareto configurations')
    configure_pareto(pareto_parser)
    pareto_parser.set_defaults(handler=run_pareto)

    encode_parser = subparsers.add_parser(
        "encode",
        help="Encode videos and optional thumbnails into DCVC-RT bitstreams",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    configure_encode_parser(encode_parser)
    encode_parser.set_defaults(handler=run_encode)

    stream_parser = subparsers.add_parser(
        "stream",
        help="Resolve remote media with yt-dlp and encode it without storing source containers",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    configure_stream_parser(stream_parser)
    stream_parser.set_defaults(handler=run_stream)

    decode_parser = subparsers.add_parser(
        "decode",
        help="Decode DCVC-RT video bitstreams to YUV and intra images to PNG",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    configure_decode_parser(decode_parser)
    decode_parser.set_defaults(handler=run_decode)

    view_parser = subparsers.add_parser(
        "view",
        help="View a DCVC-RT video, image, or image folder without writing decoded files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    configure_view_parser(view_parser)
    view_parser.set_defaults(handler=run_view)

    cleanup_parser = subparsers.add_parser(
        "cleanup",
        help="Validate completed channel artifacts and safely remove exact inventoried source paths",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    configure_cleanup_parser(cleanup_parser)
    cleanup_parser.set_defaults(handler=run_cleanup)

    from src.cli.training_workflow import (
        configure_train_parser, configure_data_parser, configure_export_parser,
        run_train, run_data, run_export,
    )
    for command, help_text, configure, handler in (
        ("train", "Train or fine-tune configurable DCVC-RT models", configure_train_parser, run_train),
        ("prepare-training-data", "Index source-disjoint datasets and cache original video frames", configure_data_parser, run_data),
        ("export-model", "Export models and reproducible prepared INT16 state", configure_export_parser, run_export),
    ):
        subparser = subparsers.add_parser(command, help=help_text)
        configure(subparser)
        subparser.set_defaults(handler=handler)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args) or 0)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
