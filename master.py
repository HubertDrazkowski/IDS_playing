import pretty_errors

from rich.traceback import install
install()

# Example that will cause an error
def this_fails():
    x = 1 / 0

this_fails()