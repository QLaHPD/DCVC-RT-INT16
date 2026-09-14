"""Bounded streaming lookahead and one-time input-buffer calibration."""
from contextlib import contextmanager
import math
import multiprocessing
from multiprocessing.reduction import DupFd
import os
from pathlib import Path
import queue
import signal
import subprocess
import tempfile
import threading
import time

from src.archive.storage import event

_PENDING = 1000000
_PREFIX = 'UF_DOWNLOAD:'
_TEMPLATE = ('download:' + _PREFIX + '%(progress.fragment_index)s:%(progress.fragment_count)s:'
             '%(progress.downloaded_bytes)s:%(progress.total_bytes)s:'
             '%(progress.total_bytes_estimate)s:%(progress.status)s')


class BufferPolicy:
    def __init__(self, context):
        self.lock = context.RLock()
        self.frames = context.Value('i', 8, lock=False)
        self.measured = context.Value('b', False, lock=False)
        self.owner = context.Value('i', 0, lock=False)

    def claim(self):
        with self.lock:
            if not self.measured.value and not self.owner.value:
                self.owner.value = os.getpid()

    def capacity(self):
        with self.lock:
            return self.frames.value

    def observe(self, frames, fps):
        if frames <= 0 or fps <= 0 or not math.isfinite(fps):
            return False
        elapsed = frames / fps
        with self.lock:
            if self.measured.value or self.owner.value != os.getpid() or elapsed < 5:
                return False
            self.frames.value = max(1, math.ceil(fps * 2))
            self.measured.value = True
            event('stream_buffer_calibrated', fps=fps, sample_seconds=elapsed,
                  buffer_frames=self.frames.value)
            return True

    def release(self, pid=None):
        with self.lock:
            if self.owner.value == (pid or os.getpid()):
                self.owner.value = 0


def near_download_end(line):
    if not line.startswith(_PREFIX):
        return False
    parts = line[len(_PREFIX):].strip().split(':')
    if len(parts) != 6:
        return False
    def number(value):
        try:
            result = float(value)
            return result if math.isfinite(result) and result >= 0 else None
        except ValueError:
            return None
    index, count, downloaded, total, estimate = map(number, parts[:5])
    if parts[5] == 'finished':
        return True
    if index is not None and count is not None and count > 0:
        return count - index <= 20
    size = total or estimate
    return bool(size and downloaded is not None and downloaded >= size * .95)


class PreparedDownload:
    """An unread OS pipe starts extraction/network work without saving a video."""
    def __init__(self, remote, context):
        from src.archive.streaming import cookie_copy, command
        self.cookies = cookie_copy(remote.get('cookies'))
        self.cookie_path = self.cookies.__enter__()
        self.errors = tempfile.TemporaryFile()
        self.status = context.Value('i', _PENDING)
        self.near_end = context.Event()
        self.process = None
        self.thread = None
        self.closed = False
        try:
            cmd = command(remote['binary'], self.cookie_path) + [
                '--progress', '--newline', '--progress-delta', '0.2',
                '--progress-template', _TEMPLATE, '--no-playlist',
                '-f', str(remote['video_format']), '-o', '-', '--', remote['url']]
            self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                            start_new_session=True)
            self.thread = threading.Thread(target=self._monitor, name='uf-download-progress', daemon=True)
            self.thread.start()
        except BaseException:
            self.close()
            raise

    def _monitor(self):
        try:
            for raw in self.process.stderr:
                line = raw.decode(errors='replace').strip()
                if line.startswith(_PREFIX):
                    if near_download_end(line):
                        self.near_end.set()
                else:
                    self.errors.write(raw)
            self.errors.flush()
            self.status.value = self.process.wait()
        except Exception as exc:
            self.errors.write(str(exc).encode())
            self.errors.flush()
            self.status.value = 1
        finally:
            self.near_end.set()

    def descriptor(self):
        # Unwrapped and closed at worker entry/exit, even for resumed/busy jobs.
        return dict(stream=DupFd(self.process.stdout.fileno()),
                    errors=DupFd(self.errors.fileno()), status=self.status)

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.process is not None:
            if self.process.poll() is None:
                try:
                    os.killpg(self.process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(self.process.pid, signal.SIGKILL)
                    self.process.wait()
            if self.thread is not None and self.thread.ident is not None:
                self.thread.join(timeout=5)
            self.process.stdout.close()
            self.process.stderr.close()
        self.errors.close()
        self.cookies.__exit__(None, None, None)


def open_descriptor(descriptor):
    opened = []
    try:
        for name in ('stream', 'errors'):
            descriptor[name] = os.fdopen(descriptor[name].detach(), 'rb')
            opened.append(descriptor[name])
    except BaseException:
        for stream in opened:
            stream.close()
        raise


def close_descriptor(descriptor):
    for name in ('stream', 'errors'):
        stream = descriptor[name]
        if hasattr(stream, 'detach') and not hasattr(stream, 'read'):
            os.close(stream.detach())
        else:
            stream.close()


@contextmanager
def prepared_pipe(descriptor):
    stream = descriptor['stream']
    yield stream
    while stream.read(1024 * 1024):
        pass
    deadline = time.monotonic() + 15
    while descriptor['status'].value == _PENDING:
        if time.monotonic() >= deadline:
            raise RuntimeError('yt-dlp did not report completion after stream EOF')
        time.sleep(.02)
    if descriptor['status'].value:
        errors = descriptor['errors']
        errors.seek(0)
        raise RuntimeError('yt-dlp stream failed: ' + errors.read().decode(errors='replace')[-2500:])


class AheadJobs:
    """Resolve at most one queued job plus one in-flight metadata request."""
    def __init__(self, jobs, stop):
        self.jobs, self.stop = jobs, stop
        self.queue = queue.Queue(maxsize=1)
        self.thread = threading.Thread(target=self._run, name='uf-next-video', daemon=True)
        self.thread.start()

    def _put(self, item):
        while not self.stop.is_set():
            try:
                self.queue.put(item, timeout=.1)
                return
            except queue.Full:
                pass

    def _run(self):
        try:
            for job in self.jobs:
                if self.stop.is_set():
                    break
                self._put(('job', job))
            self._put(('end', None))
        except BaseException as exc:
            if not self.stop.is_set():
                self._put(('error', exc))

    def close(self):
        self.stop.set()
        self.thread.join(timeout=10)
        if self.thread.is_alive():
            raise RuntimeError('Streaming metadata producer did not stop')


def needs_download(job):
    return not (Path(job['final']).exists() or (job.get('legacy') and Path(job['legacy']).exists()))


def execute_stream_jobs(args, config, paths, binary, counts):
    from src.archive.streaming import stream_jobs
    from src.archive.workflow import _job_entry
    from src.archive.dashboard import event_queue
    context = multiprocessing.get_context('spawn')
    policy = BufferPolicy(context)
    stop = threading.Event()
    producer = AheadJobs(stream_jobs(args, binary, counts, cancel_event=stop), stop)
    devices = ['cpu'] if args.device == 'cpu' else (args.cuda_idx or [0])
    free = [devices[i % len(devices)] for i in range(args.procs)]
    active, outcomes, failed = [], [], set()
    pending = warm = None
    exhausted = False
    try:
        while not exhausted or pending is not None or active:
            if pending is None and not exhausted:
                try:
                    kind, payload = producer.queue.get_nowait()
                    if kind == 'end':
                        exhausted = True
                    elif kind == 'error':
                        raise payload
                    else:
                        pending = payload
                except queue.Empty:
                    pass
            # One additional downloader may warm while the GPU slots are full.
            nearing = any(item['near'].is_set() for item in active)
            if pending is not None and warm is None and (free or nearing) and needs_download(pending):
                warm = PreparedDownload(pending['remote'], context)
                event('stream_download_started' if free else 'stream_next_prefetched', source=pending['source'])
            if pending is not None and free:
                job, prepared = pending, warm
                pending = warm = None
                near = prepared.near_end if prepared else context.Event()
                job.update(buffer_policy=policy, stream_near_end=near)
                if prepared:
                    job['prepared_download'] = prepared.descriptor()
                device = free.pop(0)
                reader, writer = context.Pipe(duplex=False)
                process = context.Process(target=_job_entry, args=(writer,job,config,paths,device,args.shared_instance),
                                          kwargs={'event_queue':event_queue()})
                try:
                    process.start()
                except BaseException:
                    reader.close(); writer.close()
                    if prepared: prepared.close()
                    raise
                writer.close()
                active.append(dict(process=process, reader=reader, job=job, device=device, prepared=prepared, near=near))
            for item in list(active):
                process = item['process']
                if process.is_alive():
                    continue
                process.join()
                try:
                    status = item['reader'].recv() if item['reader'].poll() else 'failed'
                except EOFError:
                    status = 'failed'
                if process.exitcode:
                    status = 'failed'
                    event('worker_failed', source=item['job']['source'], exitcode=process.exitcode)
                outcomes.append(status)
                if status in ('busy','failed'):
                    failed.add(str(Path(item['job']['final']).parent))
                policy.release(process.pid)
                item['reader'].close()
                if item['prepared']: item['prepared'].close()
                active.remove(item); free.append(item['device'])
            stop.wait(.02)
    finally:
        stop.set()
        for item in active:
            process = item['process']
            if process.is_alive(): process.terminate()
            if item['prepared']: item['prepared'].close()
            process.join(timeout=10)
            if process.is_alive(): process.kill(); process.join()
            item['reader'].close()
        if warm: warm.close()
        producer.close()
    return outcomes, failed
