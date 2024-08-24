from colorama import Fore, Style
from tqdm import tqdm


def get_system_version():
    return "2.1.0"

def print_logo(color=Fore.GREEN):
    tqdm.write(Style.RESET_ALL)