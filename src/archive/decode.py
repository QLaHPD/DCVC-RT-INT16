"""RT-style folder decode, keeping UF runtime and identity checks for every file."""
from copy import copy
from pathlib import Path
from src.archive.storage import event


def run_batch(args):
    from src.archive.workflow import run_decode
    if args.input_folder:
        base = Path(args.input_folder).resolve()
        paths = base.rglob('*') if args.recursive else base.iterdir()
        inputs = sorted(p for p in paths if p.is_file() and p.suffix.lower() in ('.bin','.dcvci')
                        and not any(part.startswith('.') for part in p.relative_to(base).parts))
    else:
        inputs = [Path(args.input).absolute()]
        base = inputs[0].parent
    jobs = []
    for path in inputs:
        image = path.suffix.lower()=='.dcvci'
        if args.input_kind == 'images' and not image or args.input_kind == 'videos' and image: continue
        options = copy(args)
        options.input,options.input_folder,options.output_folder = str(path),None,None
        target = Path(args.output_folder)/path.relative_to(base).with_suffix('.png' if image else '.yuv')
        options.output_file = str(target)
        jobs.append(options)
    from src.archive.workflow import execute_jobs
    options = copy(args)
    options.procs = getattr(args,'worker',1)
    options.shared_instance = None
    options.cuda_idx = args.cuda_idx if isinstance(args.cuda_idx,list) else [args.cuda_idx]
    tasks = [dict(source=job.input,final=job.output_file,options=job) for job in jobs]
    outcomes,_ = execute_jobs(options,tasks,None,None,worker_entry=_decode_entry)
    return int('failed' in outcomes)


def _decode_entry(writer,job,_config,_paths,device,_instance,event_queue=None):
    from src.archive.workflow import run_decode
    from src.archive.dashboard import install
    install(event_queue)
    options = job['options']
    options.cuda_idx = 0 if device=='cpu' else device
    try:
        result = run_decode(options)
        writer.send('encoded' if result==0 else 'failed')
    except Exception as exc:
        event('decode_failed',source=options.input,error=str(exc))
        writer.send('failed')
    finally:
        writer.close()
