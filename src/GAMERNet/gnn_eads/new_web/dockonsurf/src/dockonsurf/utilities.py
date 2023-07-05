import logging

logger = logging.getLogger('DockOnSurf')


def tail(f, lines=20):
    """Returns the specified last number of lines of a file.

    @param f: The file to retrieve the last lines from.
    @param lines: The number of lines to be retrieved.
    @return str: The last number of lines
    """
    total_lines_wanted = lines

    block_size = 1024
    f.seek(0, 2)
    block_end_byte = f.tell()
    lines_to_go = total_lines_wanted
    block_number = -1
    blocks = []
    while lines_to_go > 0 and block_end_byte > 0:
        if block_end_byte - block_size > 0:
            f.seek(block_number * block_size, 2)
            blocks.append(f.read(block_size))
        else:
            f.seek(0, 0)
            blocks.append(f.read(block_end_byte))
        lines_found = blocks[-1].count(b'\n')
        lines_to_go -= lines_found
        block_end_byte -= block_size
        block_number -= 1
    all_read_text = b''.join(reversed(blocks))
    return b'\n'.join(all_read_text.splitlines()[-total_lines_wanted:]).decode()


def check_bak(file_name):
    """Checks if a file already exists and backs it up if so.

    @param file_name: file to be checked if exists
    """
    import os
    new_name = file_name
    bak_num = 0
    while os.path.isdir(new_name) or os.path.isfile(new_name):
        bak_num += 1
        new_name = new_name.split(".bak")[0] + f".bak{bak_num}"
    if bak_num > 0:
        os.rename(file_name, new_name)
        logger.warning(f"'{file_name}' already present. Backed it up to "
                       f"{new_name}.")


def try_command(command, expct_error_types: list, *args, **kwargs):
    """Try to run a command and record exceptions (expected and not) on a log.

    @param command: method or function, the command to be executed.
    @param expct_error_types: list of tuples, every inner tuple is supposed to
    contain an exception type (eg. ValueError, TypeError, etc.) to be caught and
    a message to print in the log and on the screen explaining the exception.
    Error types that are not allow to be called with a custom message as only
    error argument are not supported.
    The outer tuple encloses all couples of error types and their relative
    messages.
    *args and **kwargs: arguments and keyword-arguments of the command to be
    executed.
    When trying to run 'command' with its args and kwargs, if an exception
    present on the 'error_types' occurs, its relative error message is recorded
    on the log and a same type exception is raised with the custom message.
    """
    unexp_error = "An unexpected error occurred"

    err = False
    try:
        return_val = command(*args, **kwargs)
    except Exception as e:
        for expct_err in expct_error_types:
            if isinstance(e, expct_err[0]):
                logger.error(expct_err[1])
                err = expct_err[0](expct_err[1])
                break
        else:
            logger.exception(unexp_error)
            err = e
    else:
        err = False
        return return_val
    finally:
        if isinstance(err, BaseException):
            raise err


def _human_key(key):
    """Function used as sorting strategy where numbers are sorted human-wise.

    @param key:
    @return:
    """
    import re
    parts = re.split('(\d*\.\d+|\d+)', key)
    return tuple((e.swapcase() if i % 2 == 0 else float(e))
                 for i, e in enumerate(parts))


def is_binary(file):
    """Checks if a file is a text file or a binary one.

    @param file:
    @return:
    """
    try:
        with open(file, "r") as fh:
            fh.read(50)
    except UnicodeDecodeError:
        return True
    else:
        return False
