"""provide completions via vertex ai"""
import os
import typing

import vertexai
from vertexai.generative_models import GenerativeModel, Part, Tool, FunctionDeclaration, ChatSession
import vertexai.preview.generative_models as generative_models


PROJECT_ID = "expeng-k8s-prototype"
LOCATION = "us-central1"
vertexai.init(project=PROJECT_ID, location=LOCATION)


generation_config = {
    "max_output_tokens": 8192,
    "temperature": 1,
    "top_p": 0.95,
}

safety_settings = {
    generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
}


def write_file(path: str, content: str, base: str='.'):
    """write content to a file"""
    path = os.path.join(base, path)
    basepath, _ = os.path.split(path)
    if basepath and not os.path.exists(basepath):
        os.makedirs(basepath)
    with open(path, "w") as file:
        file.write(content)


def cat_file(path, base='.'):
    """read a file's content"""
    path = os.path.join(base, path)
    with open(path) as file:
        return file.read()


def cat_files_of_type(suffix, base='.'):
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


def ls_tree(base='.'):
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


def invoke_tool(name, *, fn: typing.Callable, args: dict):
    """invoke a tool function with the given arguments"""
    try:
        return Part.from_function_response(
            name,
            {"content": fn(**args)}
        )
    except Exception as exc:
        return Part.from_function_response(
            name,
            {"error": str(exc)}
        )

class StatefulChat:
    chat: ChatSession

    def __init__(self, system_prompt: str):
        write_file_tool = FunctionDeclaration(
            name="write_file",
            description="fully replace the contents of a file in the project",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "the path to the file to replace inside the project"
                    }, "content": {
                        "type": "string",
                        "description": "the entire new contents for the file"
                    }
                },
                "required": ["path", "content"]
            },
        )
        cat_file_tool = FunctionDeclaration(
            name="cat_file",
            description="read the contents of a file in the project",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "the path to the file to read inside the project"
                    }
                },
                "required": ["path"]
            },
        )
        cat_files_of_type_tool = FunctionDeclaration(
            name="cat_files_of_type",
            description="read all files with a given suffix",
            parameters={
                "type": "object",
                "properties": {
                    "suffix": {
                        "type": "string",
                        "description": "the suffix of the files to read inside the project"
                    }
                },
                "required": ["suffix"]
            },
        )
        ls_tree_tool = FunctionDeclaration(
            name="ls_tree",
            description="recursively depict the directory hierarchy as an outline",
            parameters={
                "type": "object",
                "properties": {},
            },
        )
        tools = [write_file_tool, cat_file_tool, cat_files_of_type_tool,
                 ls_tree_tool]
        project_tools = Tool(function_declarations=tools)
        model = GenerativeModel(
            "gemini-1.5-pro-preview-0409",
            tools=[project_tools],
            system_instruction=system_prompt,
            safety_settings=safety_settings,
        )
        self.chat = model.start_chat()

    async def interact(self, prompt: str) -> typing.Generator[str,str,None]:
        """send the next interaction to the model and yield the response.
        automatically invoke any tools and reprompt as necessary"""
        followups = [lambda: self.chat.send_message_async(prompt, stream=True)]
        while followups:
            responses = await (followups.pop(0)())
            async for response in responses:
                # print(response)
                candidate = response.candidates[0]
                for tool in candidate.function_calls:
                    if tool.name == "write_file":
                        response = invoke_tool("write_file",
                                               fn=write_file,
                                               args={"path": tool.args["path"],
                                                     "content": tool.args["content"]})
                        followups.append(lambda: self.chat.send_message_async(response, stream=True))
                    elif tool.name == "cat_file":
                        response = invoke_tool("cat_file",
                                               fn=cat_file,
                                               args={"path": tool.args["path"]})
                        followups.append(lambda: self.chat.send_message_async(response, stream=True))
                    elif tool.name == "cat_files_of_type":
                        response = invoke_tool("cat_files_of_type",
                                               fn=cat_files_of_type,
                                               args={"suffix": tool.args["suffix"]})
                        followups.append(lambda: self.chat.send_message_async(response, stream=True))
                    elif tool.name == "ls_tree":
                        response = invoke_tool("ls_tree",
                                               fn=ls_tree,
                                               args={})
                        # print(response)
                        followups.append(lambda: self.chat.send_message_async(response, stream=True))
                
                if not candidate.function_calls:
                    yield candidate.text


async def main():
    """main function"""
    import sys
    chat = StatefulChat("")
    async for chunk in chat.interact("please create unit tests for agent/llm.py"):
        sys.stdout.write(chunk)
        sys.stdout.flush()

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())