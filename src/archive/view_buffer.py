"""Bounded YUV history and decode-ahead, with cancellable cache-miss seeks."""

import threading


class BufferedDecodeWorker:
    def __init__(self, source, capacity=512, history=512):
        if type(capacity) is not int or capacity < 1:
            raise ValueError('Decode-ahead capacity must be a positive integer')
        if type(history) is not int or history < 0:
            raise ValueError('Decode history must be a nonnegative integer')
        self.source, self.capacity = source, capacity
        self.history_capacity = history
        self.condition = threading.Condition()
        self.frames = {}
        self.request_id = None
        self.cursor = self.delivered = None
        self.generation = 0
        self.error = None
        self.eof = self.closed = False
        self.thread = threading.Thread(target=self._run, name='uf-view-prefetch', daemon=True)
        self.thread.start()

    def request(self, request_id, frame_index):
        with self.condition:
            if frame_index not in self.frames:
                self.generation += 1
                self.frames.clear()
            self.request_id, self.delivered = request_id, None
            self._move(frame_index)
            self.error = None
            self.condition.notify_all()

    def _move(self, frame_index):
        previous = self.frames.get(self.cursor)
        if previous is not None and self.cursor != frame_index:
            # Tk's PhotoImage owns the displayed pixels; retain only compact YUV.
            previous.rgb = None
        self.cursor = frame_index
        low, high = frame_index - self.history_capacity, frame_index + self.capacity
        for index in list(self.frames):
            if index < low or index > high:
                del self.frames[index]
        self.eof = self.source.index.frame_count - 1 in self.frames

    def _work_index(self):
        if self.cursor is None or self.error:
            return None
        if self.cursor not in self.frames:
            return self.cursor
        index = max(self.frames) + 1
        if index <= min(self.source.index.frame_count - 1, self.cursor + self.capacity):
            return index
        return None

    def status(self, request_id):
        with self.condition:
            if request_id != self.request_id:
                return dict(size=0, history=0, eof=False, error=None)
            return dict(size=sum(index > self.cursor for index in self.frames),
                        history=sum(index < self.cursor for index in self.frames),
                        eof=self.eof, error=self.error)

    def pop_due(self, request_id, frame_index):
        """Consume only frames whose presentation time has arrived."""
        with self.condition:
            if request_id != self.request_id:
                return None
            due = [index for index in self.frames if self.cursor <= index <= frame_index]
            if not due or max(due) == self.delivered:
                return None
            index = max(due)
            self._move(index)
            self.delivered = index
            self.condition.notify_all()
            return self.frames[index]

    def _cancelled(self, generation, index):
        with self.condition:
            return (self.closed or generation != self.generation or
                    not self.cursor - self.history_capacity <= index <= self.cursor + self.capacity)

    def _run(self):
        try:
            while True:
                with self.condition:
                    self.condition.wait_for(lambda: self.closed or
                                            self._work_index() is not None)
                    if self.closed:
                        return
                    generation, index = self.generation, self._work_index()
                try:
                    frame = self.source.seek(index, output_format='yuv420p',
                                             cancel=lambda: self._cancelled(generation, index))
                except Exception as exc:
                    with self.condition:
                        if not self._cancelled(generation, index):
                            self.error = f'{type(exc).__name__}: {exc}'
                            self.condition.notify_all()
                    continue
                with self.condition:
                    if self._cancelled(generation, index):
                        continue
                    self.frames[index] = frame
                    self.eof = self.source.index.frame_count - 1 in self.frames
                    self.condition.notify_all()
        finally:
            self.source.close()

    def close(self):
        with self.condition:
            self.closed = True
            self.frames.clear()
            self.condition.notify_all()
        # Finish the current native operation and release CUDA on its owning
        # thread before Python exits. FrameSource checks cancellation per packet.
        self.thread.join()
