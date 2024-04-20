from agent.detector import Detector


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

```pyproject.toml
[tool.poetry]
name = "my-project"
version = "0.1.0"
description = "My project description"
```

and here's the code for your main.py file.

```main.py
print("Hello, world!")
```

I hope that helps
"""

async def test_detector_example():
    """prove that a code block detector can successfully
    extract code blocks from a stream of text no matter
    how its split into chunks"""

    matches = []

    def on_wake(suffix):
        """on wake"""
        if not matches:
            # start the first match
            matches.append('')
        if "```" in suffix:
            before, _, _ = suffix.partition("```")
            matches[-1] = before
            # start the next match
            matches.append('')
            return False
        matches[-1] = suffix
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
            "```",
            on_wake,
        )

        complete = ""
        async for item in detector:
            complete += item
        
        assert complete == example_output
        assert matches == [
            "pyproject.toml\n[tool.poetry]\nname = \"my-project\"\nversion = \"0.1.0\"\ndescription = \"My project description\"\n",
            "main.py\nprint(\"Hello, world!\")\n",
            ""
        ]