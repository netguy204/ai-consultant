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


class Tokenizer:
    """tokenizes a stream of characters into words"""
    value: str
    offset: int
    state: str

    def __init__(self, value: str):
        self.value = value
        self.offset = 0
        self.state = ""
    
    def __str__(self):
        wh = " " * self.offset
        return f"{self.value}\n{wh}^"
    
    def __repr__(self):
        return f"<Tokenizer {self.value!r}, {self.offset!r}, {self.state}>"
    
    def looking_at(self, v):
        return self.value[self.offset:].startswith(v)
    
    def consume(self, v):
        if self.looking_at(v):
            self.offset += len(v)
            return v
        return None
    
    def consume_whitespace(self):
        while not self.at_end() and self.value[self.offset].isspace():
            self.offset += 1
    
    def take(self, n: int):
        result = self.value[self.offset:self.offset + n]
        self.offset += n
        return result
    
    def peek(self, n: int):
        return self.value[self.offset:self.offset + n]
    
    def rewind(self, n: int):
        self.offset -= n
    
    def at_end(self):
        return self.offset >= len(self.value)
    
    def rest(self) -> str:
        return self.value[self.offset:]

@dataclasses.dataclass
class CodeBlock:
    language: str|None
    code: str

class JSONMDParser:
    """json parser that yields the json entity it is looking at if
    it completes validly"""

    def consume_string(self, tokenizer: Tokenizer) -> str|None:
        if tokenizer.consume('"') is None:
            return None
        result = ""
        while not tokenizer.at_end():
            char = tokenizer.take(1)
            if char == '"':
                return result
            elif char == "\\":
                char = tokenizer.take(1)
                if char == "n":
                    result += "\n"
                elif char == "t":
                    result += "\t"
                else:
                    result += char
            else:
                result += char
        return None
    
    def consume_number(self, tokenizer: Tokenizer) -> float|None:
        number = ""
        first_decimal = False
        first_e = False
        while not tokenizer.at_end():
            char = tokenizer.take(1)
            if char.isdigit():
                number += char
            elif first_decimal is False and char == ".":
                number += char
                first_decimal = True
            elif first_e is False and char in ("e", "E"):
                number += char
                first_e = True
                first_decimal = True  # decimal no longer allowed
            elif char in ("+", "-") and number and number[-1] in ("e", "E"):
                number += char
            else:
                tokenizer.rewind(1)
                break
        try:
            return float(number)
        except ValueError:
            return None

    def consume_object(self, tokenizer: Tokenizer) -> dict|None:
        if tokenizer.consume("{") is None:
            return None
        result = {}
        first_field = True
        while not tokenizer.at_end():
            tokenizer.consume_whitespace()
            if tokenizer.looking_at("}"):
                tokenizer.consume("}")
                return result

            if first_field:
                first_field = False
            else:
                if tokenizer.consume(",") is None:
                    tokenizer.state = "expected comma or end of object"
                    return None
                tokenizer.consume_whitespace()

            key = self.consume_string(tokenizer)
            if key is None:
                tokenizer.state = "expected string key"
                return None
            tokenizer.consume_whitespace()
            if tokenizer.consume(":") is None:
                tokenizer.state = "expected colon after key"
                return None
            tokenizer.consume_whitespace()
            value = self.consume_value(tokenizer)
            if value is None:
                tokenizer.state = "expected value after colon"
                return None
            result[key] = value
            tokenizer.consume_whitespace()

        tokenizer.state = "unexpected end of input"
        return None
    
    def consume_array(self, tokenizer: Tokenizer) -> list|None:
        if tokenizer.consume("[") is None:
            return None
        result = []
        first_value = True
        while not tokenizer.at_end():
            tokenizer.consume_whitespace()
            if tokenizer.looking_at("]"):
                tokenizer.consume("]")
                return result

            if first_value:
                first_value = False
            else:
                if tokenizer.consume(",") is None:
                    tokenizer.state = "expected comma or end of array"
                    return None
                tokenizer.consume_whitespace()

            value = self.consume_value(tokenizer)
            if value is None:
                tokenizer.state = "expected value in array"
                return None
            result.append(value)
            tokenizer.consume_whitespace()

        tokenizer.state = "unexpected end of input"
        return None

    def consume_value(self, tokenizer: Tokenizer) -> typing.Any:
        if tokenizer.looking_at("{"):
            return self.consume_object(tokenizer)
        elif tokenizer.looking_at('"'):
            return self.consume_string(tokenizer)
        elif tokenizer.looking_at("-") or tokenizer.looking_at("+") or tokenizer.peek(1).isdigit():
            return self.consume_number(tokenizer)
        elif tokenizer.looking_at("["):
            return self.consume_array(tokenizer)
        return None
    
    def consume_code_block(self, tokenizer: Tokenizer) -> CodeBlock|None:
        if tokenizer.consume("```") is None:
            return None
        # consume to EOL
        language = ""
        while not tokenizer.at_end():
            char = tokenizer.take(1)
            if char == "\n":
                break
            language += char
        
        # normalize missing language
        language.strip()
        if not language:
            language = None
    
        # consume code block
        code = ""
        found_end = False
        while not tokenizer.at_end():
            if tokenizer.consume("```") is not None:
                found_end = True
                break
            code += tokenizer.take(1)

        if not found_end:
            return None

        return CodeBlock(language=language, code=code)
    
    def consume_extended_value(self, tokenizer: Tokenizer) -> typing.Any:
        if tokenizer.looking_at("```"):
            return self.consume_code_block(tokenizer)
        return self.consume_value(tokenizer)
    
    def parse(self, value: str) -> typing.Any:
        tokenizer = Tokenizer(value)
        return self.consume_value(tokenizer)
    
    def scan(self, text: str) -> typing.Generator[typing.Any, typing.Any, None]:
        """scan document for valid json objects and emit them"""
        tokenizer = Tokenizer(text)
        while not tokenizer.at_end():
            tokenizer.consume_whitespace()
            value = self.consume_extended_value(tokenizer)
            if value is None:
                tokenizer.take(1)
            else:
                yield value
