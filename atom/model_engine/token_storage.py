# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import array
import operator
from itertools import chain


class TokenStorage:
    """A borrowed, read-only prompt followed by an owned int32 tail."""

    __slots__ = ("_prompt", "prompt_handle", "tail")

    @classmethod
    def borrow_prompt(cls, handle):
        storage = cls.__new__(cls)
        storage.prompt_handle = handle.acquire()
        storage._prompt = None
        storage.tail = array.array("i")
        try:
            view = storage.prompt_handle.view
            if (
                view.ndim != 1
                or not view.c_contiguous
                or view.itemsize != 4
                or view.format != "i"
                or not view.readonly
            ):
                raise ValueError("prompt must be a read-only contiguous int32 view")
            storage._prompt = view[:]
        except BaseException:
            storage.release()
            raise
        return storage

    def __len__(self):
        return len(self._prompt) + len(self.tail)

    def __iter__(self):
        return chain(self._prompt, self.tail)

    def token_at(self, index):
        index = operator.index(index)
        length = len(self)
        if index < 0:
            index += length
        if not 0 <= index < length:
            raise IndexError("token index out of range")
        prompt_length = len(self._prompt)
        if index < prompt_length:
            return self._prompt[index]
        return self.tail[index - prompt_length]

    def __getitem__(self, key):
        if not isinstance(key, slice):
            return self.token_at(key)
        start, end, step = key.indices(len(self))
        if step != 1:
            return array.array("i", (self.token_at(i) for i in range(start, end, step)))
        return self.owned_slice(start, end)

    def _tail_key(self, key):
        prompt_length = len(self._prompt)
        if isinstance(key, slice):
            start, end, step = key.indices(len(self))
            indices = range(start, end, step)
            if start < prompt_length or (indices and indices[-1] < prompt_length):
                raise TypeError("prompt tokens are read-only")
            # Preserve the negative-step sentinel after translating to tail offsets.
            stop = end - prompt_length
            if step < 0 and stop < 0:
                stop = None
            return slice(start - prompt_length, stop, step)
        index = operator.index(key)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError("token index out of range")
        if index < prompt_length:
            raise TypeError("prompt tokens are read-only")
        return index - prompt_length

    def __setitem__(self, key, value):
        tail_key = self._tail_key(key)
        if isinstance(tail_key, slice) and not isinstance(value, array.array):
            value = array.array("i", value)
        self.tail[tail_key] = value

    def __delitem__(self, key):
        del self.tail[self._tail_key(key)]

    def append(self, token):
        self.tail.append(token)

    append_generated = append

    def extend(self, tokens):
        self.tail.extend(tokens)

    def pop(self, index=-1):
        return self.tail.pop(self._tail_key(index))

    def prompt_view(self):
        return self._prompt[:]

    def spans(self, start=0, end=None):
        start, end, _ = slice(start, end).indices(len(self))
        if start >= end:
            return ()
        prompt_length = len(self._prompt)
        spans = []
        if start < prompt_length:
            spans.append(self._prompt[start : min(end, prompt_length)])
        if end > prompt_length:
            spans.append(
                memoryview(self.tail)[
                    max(start - prompt_length, 0) : end - prompt_length
                ]
            )
        return tuple(spans)

    def copy_into(self, destination, start=0, end=None):
        """Append to an int32 array or fill a writable buffer with this window."""
        spans = self.spans(start, end)
        try:
            if isinstance(destination, array.array):
                if destination.typecode != "i":
                    raise TypeError("token destination must be an int32 array")
                for span in spans:
                    destination.frombytes(span.cast("B"))
            else:
                with memoryview(destination).cast("B") as target:
                    offset = 0
                    for span in spans:
                        size = span.nbytes
                        target[offset : offset + size] = span.cast("B")
                        offset += size
        finally:
            for span in spans:
                span.release()

    def owned_slice(self, start=0, end=None):
        tokens = array.array("i")
        self.copy_into(tokens, start, end)
        return tokens

    def range_equals(self, start, expected):
        if start < 0:
            start += len(self)
        if start < 0 or start + len(expected) > len(self):
            return False
        return all(
            self.token_at(start + i) == token for i, token in enumerate(expected)
        )

    def overwrite_tail(self, offset, tokens):
        tokens = array.array("i", tokens)
        if offset < 0 or offset + len(tokens) > len(self.tail):
            raise IndexError("tail overwrite out of range")
        self.tail[offset : offset + len(tokens)] = tokens

    def truncate_tail(self, length):
        if not 0 <= length <= len(self.tail):
            raise ValueError("tail length out of range")
        del self.tail[length:]

    def completion_view_or_snapshot(self, final_length):
        return self.tail[:final_length]

    def release(self):
        if self._prompt is not None:
            self._prompt.release()
            self._prompt = None
        handle, self.prompt_handle = self.prompt_handle, None
        if handle is not None:
            handle.release()

    def __del__(self):
        if getattr(self, "prompt_handle", None) is not None:
            self.release()
