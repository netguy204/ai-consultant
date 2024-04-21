import re

from agent.detector import Detector, JSONMDParser, Tokenizer, CodeBlock


async def demo_stream(seq):
    """demo stream"""
    for item in seq:
        yield item


async def test_basic_detector():
    """test detector"""
    matches = []

    def on_wake(suffix):
        """on wake"""
        matches.append(suffix)
        return True

    detector = Detector(
        demo_stream(["a", "b", "c", "d", "e", "f"]),
        "c",
        on_wake,
    )

    complete = ""
    async for item in detector:
        complete += item
    
    assert complete == "abcdef"
    assert matches == ["d", "de", "def"]


async def test_detector_no_match():
    """test detector"""
    matches = []

    def on_wake(suffix):
        """on wake"""
        matches.append(suffix)
        return True

    detector = Detector(
        demo_stream(["a", "b", "c", "d", "e", "f"]),
        "z",
        on_wake,
    )

    complete = ""
    async for item in detector:
        complete += item
    
    assert complete == "abcdef"
    assert matches == []


async def test_detector_start_stop_wake():
    """prove that on_wake can return false and the detector will wake it again
    on the next match"""
    matches = []

    def on_wake(suffix):
        """on wake"""
        matches.append(suffix)
        return False

    detector = Detector(
        demo_stream(["a", "b", "c", "d", "e", "c", "f"]),
        "c",
        on_wake,
    )

    complete = ""
    async for item in detector:
        complete += item
    
    assert complete == "abcdecf"
    assert matches == ["d", "f"]


example_output = """
Certainly, I'd love to help you with that.

Here's the code for your specification.

#### pyproject.toml
```toml
[tool.poetry]
name = "my-project"
version = "0.1.0"
description = "My project description"
```

and here's the code for your main.py file.

#### main.py
```python
print("Hello, world!")
```

I hope that helps
"""


async def test_detector_example():
    """prove that a code block detector can successfully
    extract code blocks from a stream of text no matter
    how its split into chunks"""

    matches = []
    # should capture code blocks between ``` markers while ignoring the
    # language tag after the first marker
    matcher = re.compile(r"```.*?\n(.*?)```", re.DOTALL)

    def on_wake(suffix):
        """extract the code between the ``` markers and add it to matches"""
        match = matcher.search(suffix)
        if match:
            matches.append(match.group(1))
            return False
        return True

    for chunk_size in range(2, 10):
        matches = []
        # partition the example output into chunks of size chunk_size
        partitioning = [
            example_output[i:i+chunk_size]
            for i in range(0, len(example_output), chunk_size)
        ]

        detector = Detector(
            demo_stream(partitioning),
            "####",
            on_wake,
        )

        complete = ""
        async for item in detector:
            complete += item
        print(matches)
        assert complete == example_output
        assert matches == [
            "[tool.poetry]\nname = \"my-project\"\nversion = \"0.1.0\"\ndescription = \"My project description\"\n",
            "print(\"Hello, world!\")\n"
        ]


def test_parse_string():
    """we can extract a string if we're looking at one"""
    parser = JSONMDParser()
    assert parser.consume_string(Tokenizer('"hello"')) == "hello"
    assert parser.consume_string(Tokenizer('"hello" 123')) == "hello"
    assert parser.consume_string(Tokenizer('123 "hello"')) is None
    assert parser.consume_string(Tokenizer('"\\\"hello\\\""')) == "\"hello\""


def test_parse_number():
    """we can extract a number if we're looking at one"""
    parser = JSONMDParser()
    assert parser.consume_number(Tokenizer("123")) == 123
    assert parser.consume_number(Tokenizer("123.45")) == 123.45
    assert parser.consume_number(Tokenizer("123e4")) == 123e4
    assert parser.consume_number(Tokenizer("123e-4")) == 123e-4
    assert parser.consume_number(Tokenizer("123 abc")) == 123
    assert parser.consume_number(Tokenizer("123.45.")) == 123.45


def test_parse_object():
    """we can extract any fully valid json object we are looking at"""
    parser = JSONMDParser()
    assert parser.consume_object(Tokenizer("{}")) == {}
    assert parser.consume_object(Tokenizer('{')) is None
    assert parser.consume_object(Tokenizer('{"foo": 1')) is None
    assert parser.consume_object(Tokenizer('{"a": 1}')) == {"a": 1}
    assert parser.consume_object(Tokenizer('{"a": 1}}12')) == {"a": 1}
    assert parser.consume_object(Tokenizer('{"a": 1, "b": "hello"}')) == {"a": 1, "b": "hello"}
    assert parser.consume_object(Tokenizer('{"a": 1, }')) is None


def test_parse_array():
    """we can extract any fully valid array we are looking at"""
    parser = JSONMDParser()
    assert parser.consume_array(Tokenizer("[]")) == []
    assert parser.consume_array(Tokenizer("[")) is None
    assert parser.consume_array(Tokenizer("[1")) is None
    assert parser.consume_array(Tokenizer("[1]")) == [1]
    assert parser.consume_array(Tokenizer("[1, 2]")) == [1, 2]
    assert parser.consume_array(Tokenizer("[1, 2,]")) is None
    assert parser.consume_array(Tokenizer('[1, 2, "3"]')) == [1, 2, "3"]


def test_complex():
    """we can extract a complex json object"""
    parser = JSONMDParser()
    assert parser.parse('{"a": 1, "b": [1, 2, 3], "c": {"d": "hello"}}') == {"a": 1, "b": [1, 2, 3], "c": {"d": "hello"}}
    assert parser.parse('{"a": 1, "b": [1, 2, 3], "c": {"d": "hello"}') is None
    assert parser.parse('{"a": 1, "b": [1, 2, 3], "c": {"d": "hello"') is None
    assert parser.parse('[{"key": "foo"}]') == [{"key": "foo"}]
    assert parser.parse('[{"key": "foo"}') is None


def test_scan():
    doc = """
    I've been thinking about ones
    {"a": 1}
    And occasionally twos
    {"b": 2}

    But never threes
    {"c": 3}

    But here's a 4.

```javascript
console.log("and some code")
```
    """

    parser = JSONMDParser()
    assert list(parser.scan(doc)) == [{"a": 1}, {"b": 2}, {"c": 3}, 4, CodeBlock(language="javascript", code='console.log("and some code")\n')]
