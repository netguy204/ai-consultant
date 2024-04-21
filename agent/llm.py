"""provide completions via vertex ai"""
import os
import typing
import json
import re

import anthropic
import dotenv

from . import detector

def write_file(path: str, content: str, base):
    """write content to a file"""
    path = os.path.join(base, path)
    basepath, _ = os.path.split(path)
    if basepath and not os.path.exists(basepath):
        os.makedirs(basepath)
    with open(path, "w") as file:
        file.write(content)


def cat_file(path, base):
    """read a file's content"""
    path = os.path.join(base, path)
    with open(path) as file:
        return file.read()


def cat_files_of_type(suffix, base):
    """read all files with a given suffix"""
    result = ""
    for root, _, files in os.walk(base):
        for file in files:
            if file.endswith(suffix):
                rel_path = os.path.relpath(os.path.join(root, file), base)
                result += f"#### {rel_path}\n"
                result += "```\n"
                result += cat_file(rel_path, base)
                result += "\n```\n\n"
    return result


def file_metadata(path: str) -> str:
    """returns the line count if the file is text, binary otherwise"""
    try:
        with open(path) as file:
            return f"{len(file.readlines())} lines"
    except UnicodeDecodeError:
        return "binary"


def ls_tree(base: str):
    """recursively depict the directory hierarchy as an outline
    including line counts for all files"""
    result = ""
    for root, _, files in os.walk(base):
        rel_path = os.path.relpath(root, base)
        result += f"### {rel_path}\n"
        for file in files:
            full_path = os.path.join(root, file)
            result += f"* {file} ({file_metadata(full_path)})\n"
        result += "\n"
    return result


def run_tests(base: str) -> str:
    """run tests in the tests/ directory"""
    import subprocess
    result = subprocess.run(["poetry", "run", "pytest"],
                            cwd=base,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode == 0:
        return "all tests passed"
    return f"pytests failed with {result.returncode}: {result.stderr.decode() + result.stdout.decode()}"

def run_poetry(args: list[str], base: str) -> str:
    """run a poetry command"""
    import subprocess
    result = subprocess.run(["poetry", *args],
                            cwd=base,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode == 0:
        return result.stdout.decode()
    return f"poetry failed with {result.returncode}: {result.stderr.decode() + result.stdout.decode()}"

class ChatSession:
    transcript: list[dict]
    client: anthropic.Anthropic
    system: str
    base: str

    def __init__(self, client: anthropic.Anthropic, system: str, base: str):
        self.transcript = []
        self.system = system
        self.client = client
        self.base = base

    async def send_message_async(self, message: str) -> typing.AsyncGenerator[str, None]:
        """send a message to the model and yield the response
        as it comes in"""
        self.transcript.append({"role": "user", "content": message})
        with self.client.messages.stream(
            model="claude-3-opus-20240229",
            max_tokens=4096,
            messages=self.transcript,
            system=self.system,
        ) as stream:
            completion = ""
            for text in stream.text_stream:
                completion += text
                yield text
        self.transcript.append({"role": "assistant", "content": completion})
        with open(os.path.join(self.base, "transcript.json"), "w") as file:
            json.dump(self.transcript, file)


class StatefulChat:
    """a chat session with a generative model that can invoke tools"""
    chat: ChatSession
    base_path: str

    def __init__(self, system_prompt: str, base_path: str):
        self.base_path = base_path
        dotenv.load_dotenv()
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.chat = ChatSession(client, system_prompt, self.base_path)
    
    def evaluate_tools(self, message: str) -> str|None:
        """evaluate any tools in the message and return the result"""
        parser = detector.JSONMDParser()
        # pathish = re.compile(r"([\w/]+\.\w+)")
        
        active_code_block = None

        def inner_write_file(**args):
            if active_code_block is None:
                raise ValueError("there was code block immediately before this write_file command. I should try emitting the code block and then the write_file command")
            return write_file(**args, content=active_code_block.code)
        
        tools = {
            "write_file": lambda **args: inner_write_file(**args, base=self.base_path),
            "cat_file": lambda **args: cat_file(**args, base=self.base_path),
            "cat_files_of_type": lambda **args: cat_files_of_type(**args, base=self.base_path),
            "ls_tree": lambda **args: ls_tree(**args, base=self.base_path),
            "check_tests": lambda **args: run_tests(**args, base=self.base_path),
            "poetry": lambda **args: run_poetry(**args, base=self.base_path),
        }

        observations = []
        for element in parser.scan(message):
            if isinstance(element, detector.CodeBlock):
                if element.language == 'json':
                    # gemini can't be trusted to not put commands in
                    # code blocks blocks, so we'll parse them here
                    try:
                        element = json.loads(element.code)
                    except json.JSONDecodeError as exc:
                        observations.append(f"{element.code}: failed to parse as json: {exc}")
                        continue
                else:
                    if active_code_block is not None:
                        active_code_block.code += element.code
                    else:
                        # gemini also insists on skipping write_file and
                        # putting the file name at the beginning of the code block
                        # so try to accomodate that
                        # first_line = element.code.split("\n")[0]
                        # match = pathish.match(first_line)
                        # if match:
                        #     write_file(path=match.group(1), content=element.code, base=self.base_path)
                        # else:
                        active_code_block = element

            if not isinstance(element, dict):
                continue

            if "command" not in element:
                observations.append(json.dumps(element) + " is bare json without a command key")
                continue

            name = element["command"]
            element.pop("command")
            if name not in tools:
                observations.append(f"unknown tool {name}")
                continue

            try:
                result = tools[name](**element)
                observations.append(f"invoked {name} with {element} and got {result}")
                active_code_block = None
            except Exception as exc:
                observations.append(f"failed to invoke {name} with {element}: {exc}")
        
        if active_code_block is not None:
            observations.append("the final code block was not folloed by a write_file command")

        return "\n".join(f"OBSERVATION: {obs}\n" for obs in observations)


    async def interact(self, prompt: str) -> typing.Generator[str,str,None]:
        """send the next interaction to the model and yield the response.
        automatically invoke any tools and reprompt as necessary"""
        followups = [lambda: self.chat.send_message_async(prompt)]
        
        while followups:
            responses = followups.pop(0)()
            full_completion = ""
            async for text in responses:
                full_completion += text
                yield text
            yield "\n"
            tool_output = self.evaluate_tools(full_completion)
            if tool_output:
                followups.append(lambda: self.chat.send_message_async(tool_output))


async def main():
    """main function"""
    import sys
    chat = StatefulChat("")
    async for chunk in chat.interact("please use the provided tools to carefully analyze agent/llm.py and create complete and functional unit tests for it in the tests/agent directory"):
        sys.stdout.write(chunk)
        sys.stdout.flush()

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())