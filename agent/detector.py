"""streaming pattern detectors to extract tool usage from a stream of tokens"""

import typing
import dataclasses


@dataclasses.dataclass
class Detector:
    """observes a generated stream of words until wake word is detected
    and then calls on_wake with the accumulated suffix until it returns
    false"""
    iterator: typing.AsyncIterator[str]
    wake_word: str
    on_wake: typing.Callable[[str], bool]

    # when we're not awake, this is a ring buffer searching for the wake
    # word. when we are, this is the suffix we're accumulating for on_wake
    word_buffer: str = ""
    is_awake: bool = False

    def __aiter__(self):
        return self
    
    async def __anext__(self):
        item = await anext(self.iterator)

        self.word_buffer += item
        if self.is_awake:
            if not self.on_wake(self.word_buffer):
                self.is_awake = False
                self.word_buffer = ""
        else:
            if self.wake_word in self.word_buffer:
                suffix = self.word_buffer.split(self.wake_word)[1]
                if suffix:
                    if self.on_wake(suffix):
                        self.word_buffer = suffix
                        self.is_awake = True
                else:
                    self.is_awake = True
                    self.word_buffer = ""
            else:
                # only keep the last len(wake_word) characters
                self.word_buffer = self.word_buffer[-len(self.wake_word):]
        return item
