"""console based chat application"""

import sys
import asyncio
import argparse
import os
import re

import agent.llm as llm
from agent.detector import Detector

async def main2():
    """main function"""
    args = argparse.ArgumentParser()
    args.add_argument("outdir")
    args.add_argument("--prompt", default="consultant.prompt")
    args = args.parse_args()

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    matcher = re.compile(r"```.*?\n(.*?)```", re.DOTALL)

    def on_wake(suffix):
        """extract the code between the ``` markers and write it to disk"""
        match = matcher.search(suffix)
        if match:
            # the first line of the suffix should be the file name
            filename = suffix.split("\n")[0]
            path = os.path.join(args.outdir, filename)

            # ensure the destination directory exists
            subdir, _ = os.path.split(path)
            if not os.path.exists(subdir):
                os.makedirs(subdir)

            with open(path, "w") as file:
                file.write(match.group(1))

            return False
        return True
    
    system = open(args.prompt).read()
    transcript = []
    while True:
        prompt = input("> ")
        if not prompt:
            break
        transcript.append({"role": "client", "content": prompt})
        prefix = system + "\n\n" + "\n".join([
            f"{item['role']}: {item['content']}"
            for item in transcript
            if item["role"] != "system"
        ])

        detector = Detector(
            llm.agenerate(prefix),
            "####",
            on_wake,
        )
        completion = ""
        async for chunk in detector:
            completion += chunk
            sys.stdout.write(chunk)
            sys.stdout.flush()
        transcript.append({"role": "consultant", "content": completion})


async def main():
    """main function"""
    args = argparse.ArgumentParser()
    args.add_argument("outdir")
    args.add_argument("--prompt", default="consultant.prompt")
    args = args.parse_args()

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    
    chat = llm.StatefulChat(
        system_prompt=open(args.prompt).read(),
        base_path=args.outdir,
    )
    while True:
        prompt = input("> ")
        if not prompt:
            break
        async for chunk in chat.interact(prompt):
            sys.stdout.write(chunk)
            sys.stdout.flush()

if __name__ == '__main__':
    asyncio.run(main())
