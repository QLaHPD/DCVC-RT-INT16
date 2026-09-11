"""Keep optional training dependencies out of normal encode/decode imports."""


def configure_train_parser(parser):
    parser.add_argument("--config", required=True, help="Versioned YAML or JSON training configuration")
    parser.add_argument("--resume", help="Resume an optimizer-boundary training checkpoint")
    parser.add_argument("--check-config", action="store_true", help="Validate and print configuration without training")


def run_train(args):
    from src.training.config import load_config
    config = load_config(args.config)
    if args.check_config:
        import json
        print(json.dumps(config.to_dict(), indent=2, sort_keys=True))
        return 0
    from src.training.trainer import train
    return train(config, args.resume)


def configure_data_parser(parser):
    parser.add_argument("--config", required=True, help="Training config describing source media and cache paths")


def run_data(args):
    from datetime import datetime, timezone
    from src.training.config import load_config
    from src.training.data import prepare_data
    prepare_data(load_config(args.config), emit=lambda text: print(
        f"{datetime.now(timezone.utc).isoformat()} {text}", flush=True))
    return 0


def configure_export_parser(parser):
    parser.add_argument("--checkpoint", required=True, help="Training or metadata-bearing model checkpoint")
    parser.add_argument("--output", required=True, help="New directory for checkpoints, INT16 state and manifest")


def run_export(args):
    from src.training.export import export_checkpoint
    manifest = export_checkpoint(args.checkpoint, args.output)
    print(f"Exported {', '.join(manifest['models'])} model(s) to {args.output}")
    return 0
