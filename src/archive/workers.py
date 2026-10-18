"""Long-lived INT16 encoders: one checked model pair per worker process."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
import multiprocessing.connection
import os
from pathlib import Path
import signal
import time

from src.archive.storage import event, sha256


def source_fingerprint(source):
    value = Path(source).stat()
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def prepare_local_job(job):
    """Overlap bounded CPU-only input work with an already-running encoder."""
    if job.get('remote'):
        return job
    try:
        if Path(job['final']).exists() or (job.get('legacy') and Path(job['legacy']).exists()):
            return job
        from src.archive.media import probe
        source = Path(job['source'])
        before = source_fingerprint(source)
        digest = sha256(source)
        media = probe(source)
        if source_fingerprint(source) == before:
            return {**job, 'prepared_source': dict(fingerprint=before, sha256=digest,
                    media=media, identity=dict(size=before[2], mtime_ns=before[3]))}
    except Exception:
        # The owning worker reports errors through its normal lease/error path.
        pass
    return job


class PreparedJobs:
    """Prepare at most one extra input per encoder, without loading any models."""
    def __init__(self, jobs, workers):
        self.jobs = iter(jobs)
        self.capacity = workers
        self.pending = deque()
        self.executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='uf-prepare')
        self._fill()

    def _fill(self):
        while len(self.pending) < self.capacity:
            job = next(self.jobs, None)
            if job is None:
                break
            self.pending.append(self.executor.submit(prepare_local_job, job))

    def __iter__(self):
        return self

    def __next__(self):
        if not self.pending:
            raise StopIteration
        job = self.pending.popleft().result()
        self._fill()
        return job

    def close(self):
        for future in self.pending:
            future.cancel()
        self.executor.shutdown(wait=True, cancel_futures=True)


class WorkerCodec:
    """Load lazily, so a run containing only completed archives loads no models."""
    def __init__(self, config, paths, device):
        self.config, self.paths, self.device = config, paths, device
        self.codec = None

    def __call__(self):
        if self.codec is None:
            from src.archive.workflow import make_codec
            started = time.monotonic()
            self.codec = make_codec(self.config, self.paths, self.device)
            event('codec_initialized', worker_pid=os.getpid(), device=self.device,
                  load_seconds=time.monotonic()-started,
                  message='Models loaded; this worker will reuse them for subsequent videos')
        return self.codec

    def reset(self):
        if self.codec is not None:
            self.codec.reset_video()


def _encode_worker(connection, config, paths, device, instance, events):
    from src.archive.dashboard import install
    from src.archive.workflow import encode_job
    install(events)

    def terminate(_signal, _frame):
        # Unwind readers, RAM staging and ownership leases before exiting.
        raise KeyboardInterrupt()

    signal.signal(signal.SIGTERM, terminate)
    provider = WorkerCodec(config, paths, device)
    try:
        while True:
            try:
                job = connection.recv()
            except EOFError:
                break
            if job is None:
                break
            try:
                status = encode_job(job, config, paths, device, instance,
                                    codec_provider=provider)
                provider.reset()
            except Exception as exc:
                event('worker_failed', source=job['source'], worker_pid=os.getpid(),
                      error=f'{type(exc).__name__}: {exc}')
                status = 'failed'
            connection.send(status)
            # A failed CUDA operation may poison the context. Replace this
            # worker on failure; successful videos keep their checked weights.
            if status == 'failed':
                break
    finally:
        connection.close()


class PersistentEncoder:
    def __init__(self, context, config, paths, device, instance, events):
        self.device = device
        self.job = None
        self.connection, child = context.Pipe()
        self.process = context.Process(target=_encode_worker,
                                       args=(child, config, paths, device, instance, events),
                                       name='uf-encoder')
        try:
            self.process.start()
        except BaseException:
            self.connection.close()
            raise
        finally:
            child.close()

    def assign(self, job):
        self.job = job
        self.connection.send(job)

    def receive(self):
        try:
            status = self.connection.recv() if self.connection.poll() else 'failed'
        except (EOFError, OSError):
            status = 'failed'
        if self.process.exitcode:
            status = 'failed'
        if status not in ('encoded', 'resumed', 'busy', 'failed'):
            status = 'failed'
        if status == 'failed':
            event('worker_failed', source=self.job['source'], worker_pid=self.process.pid,
                  exitcode=self.process.exitcode,
                  message='Replacing failed encoder worker before the next video')
        self.job = None
        return status

    def close(self, abort=False):
        try:
            if self.process.is_alive():
                if abort:
                    self.process.terminate()
                else:
                    try:
                        self.connection.send(None)
                    except (BrokenPipeError, EOFError, OSError):
                        pass
            self.process.join(timeout=10)
            if self.process.is_alive():
                self.process.kill()
                self.process.join()
        finally:
            self.connection.close()
            self.process.close()


def execute_encode_jobs(args, jobs, config, paths, context, devices, events, job_result=None):
    from src.int16.scheduling import EncodeGates
    slots = [devices[index % len(devices)] for index in range(args.procs)]
    gates = EncodeGates(config, slots)
    remaining = PreparedJobs(jobs, len(slots))
    workers, outcomes, failed_channels = [], [], set()

    def assign(worker):
        job = next(remaining, None)
        if job is None:
            return False
        worker.assign(gates.job(job, worker.device))
        return True

    try:
        for device in slots:
            # Avoid creating idle processes when there are fewer jobs than slots.
            job = next(remaining, None)
            if job is None:
                break
            worker = PersistentEncoder(context, config, paths, device, args.shared_instance, events)
            workers.append(worker)
            worker.assign(gates.job(job, device))
        while workers:
            handles = [handle for worker in workers
                       for handle in (worker.connection, worker.process.sentinel)]
            ready = multiprocessing.connection.wait(handles, timeout=.1)
            for worker in list(workers):
                if worker.connection not in ready and worker.process.sentinel not in ready:
                    continue
                job, device = worker.job, worker.device
                status = worker.receive()
                outcomes.append(status)
                if job_result is not None:
                    job_result(job, status)
                if status in ('failed', 'busy'):
                    failed_channels.add(str(Path(job['final']).parent))
                if status == 'failed' or not worker.process.is_alive():
                    worker.close(abort=status == 'failed')
                    workers.remove(worker)
                    job = next(remaining, None)
                    if job is None:
                        continue
                    worker = PersistentEncoder(context, config, paths, device, args.shared_instance, events)
                    workers.append(worker)
                    worker.assign(gates.job(job, device))
                elif not assign(worker):
                    worker.close()
                    workers.remove(worker)
    finally:
        for worker in workers:
            worker.close(abort=True)
        remaining.close()
        gates.close()
    return outcomes, failed_channels
