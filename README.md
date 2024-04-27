AI Consultant
==============================

This is a software engineer consultant that given a description of an idea will create a plan and then attempt to implement and test it.

## Installation

1. Clone the repo
2. `poetry shell` # activates virtual environment
3. `poetry install` # install's dependencies
4. `cd isolation_env/python`
5. `docker build -t python_isolation` # create the docker image for the shell the assistant is allowed to use
6. `cd ../..`
7. `echo ANTHROPIC_API_KEY=..." > .env` # set the environment variable for the anthropic api key

## Execution
`python chat.py ${project_directory}`

This will start the chatbot to work on the project in project_directory

## Issues

The bot frequently forgets the rules for writing source files. It should emit a markdown block and then the write_file command but sometimes it emits the write_file command first. Usually it will figure out its mistake after a few surprises.

## Code

* `agent/llm.py` - Interacts with anthropic and provides the tool execution
* `agent/detector.py` - Parsers that find the tool invocations in the LLM output
* `agent/shell.py` - Provides the isolated execution environment to the agent (via docker)
* `consultant.prompt` - The prompt that creates the consultant behavior and describes the tool use
