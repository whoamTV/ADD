from colorama import Fore, Style
from tqdm import tqdm


def get_system_version():
    return "1.1.1"


def print_logo(color=Fore.BLUE):
    tqdm.write(color)
    tqdm.write(Style.RESET_ALL)