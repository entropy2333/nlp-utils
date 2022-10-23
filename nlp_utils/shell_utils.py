from subprocess import PIPE, Popen


class ShellFactory:
    """
    Factory class for creating shell utilities.

    >>> from nlp_utils.shell_utils import ShellFactory
    >>> cmd = 'echo "Hello World!"'
    >>> print(execute_command(cmd))
    """

    @staticmethod
    def get_current_dir():
        return "pwd"

    @staticmethod
    def format_date():
        return "date +%Y%m%d_%H%M%S"

    @staticmethod
    def find_process_by_pid(pid):
        return f"ps -p {pid}"

    @staticmethod
    def find_process_by_name(name):
        return f"ps -ef | grep {name}"

    @staticmethod
    def kill_process_by_pid(pid):
        return f"kill -9 {pid}"

    @staticmethod
    def kill_process_by_name(name):
        return f"killall {name}"


def execute_command(command):
    """
    Execute a command in a shell and return the output.
    """
    process = Popen(command, shell=True, stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    return stdout.decode("utf-8").strip()
