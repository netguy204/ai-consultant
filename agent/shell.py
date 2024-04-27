"""tools for interacting with an isolated shell"""
import dataclasses
import re
import os

import pexpect

@dataclasses.dataclass
class ShellEnvironment:
    """a shell environment"""
    image: str
    sentinal: str
    init: str

python_isolation = ShellEnvironment(
    image="python_isolation",
    sentinal="MySentinalPrompt>",
    init="poetry shell && poetry install",
)

IMAGE="python_isolation"#"python:3.11.9-slim-bullseye"
SENTINAL="MySentinalPrompt>"
DOCKER="/usr/local/bin/docker"

@dataclasses.dataclass
class ShellResponse:
    """a response from a shell command"""
    output: str
    return_code: int


class Shell:
    """a shell session"""
    shell: pexpect.spawn
    env: ShellEnvironment

    def __init__(self, cwd: str, env: ShellEnvironment):
        self.env = env
        cwd = os.path.abspath(cwd)
        self.shell = pexpect.spawn(f"{DOCKER} run -v {cwd}:/app -it {env.image} /bin/bash -l")
        result = self.shell.expect([":/#", pexpect.EOF, pexpect.TIMEOUT], timeout=120)
        if result == 1:
            output = self.shell.before.decode()
            raise Exception(f"no shell returned: {output}")
        if result == 2:
            output = self.shell.before.decode()
            raise Exception(f"timeout waiting for shell: {output}")
        self.shell.sendline(f"export PS1='{SENTINAL}'")
        self.shell.expect(SENTINAL)
        self.shell.expect(SENTINAL)

        self.run("cd /app")
        #self.run("poetry shell")
        self.run("poetry install")

    def _run_echo(self, line: str, timeout) -> str:
        ansi_escape = re.compile(r'''
            \x1B  # ESC
            (?:   # 7-bit C1 Fe (except CSI)
                [@-Z\\-_]
            |     # or [ for CSI, followed by a control sequence
                \[
                [0-?]*  # Parameter bytes
                [ -/]*  # Intermediate bytes
                [@-~]   # Final byte
            )
        ''', re.VERBOSE)
        self.shell.sendline(line)
        self.shell.expect(SENTINAL, timeout=timeout)

        unescaped = ansi_escape.sub('', self.shell.before.decode())

        output = unescaped.split('\r\n')[1:]
        
        return ("\n".join(output)).replace('\r', '')

    def run(self, line: str, timeout=-1) -> ShellResponse:
        """send a line to the shell"""
        output = self._run_echo(line, timeout)
        code = self._run_echo("echo $?", -1)
        return ShellResponse(output, int(code))

    def close(self):
        self.shell.terminate(force=True)
        self.shell.wait()
        self.shell.close()
