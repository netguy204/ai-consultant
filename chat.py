"""console based chat application"""

import sys
import asyncio

import agent.llm as llm


async def main():
    """main function"""
    transcript = []
    while True:
        prompt = input("> ")
        if not prompt:
            break
        transcript.append({"role": "user", "content": prompt})
        prefix = "\n".join([
            f"{item['role']}: {item['content']}"
            for item in transcript
        ])
        completion = ""
        async for chunk in llm.agenerate(prefix):
            completion += chunk
            sys.stdout.write(chunk)
            sys.stdout.flush()
        transcript.append({"role": "agent", "content": completion})

if __name__ == '__main__':
    asyncio.run(main())
