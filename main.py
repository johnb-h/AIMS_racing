"""
main.py
    Main run file for AIMS racing application
"""

__version__ = '0.0.0'
__project__ = 'aims_racing'
__tested__ = 'N'

# Standard Packages
import argparse
import sys

# Relative Imports
from user_interface.application_manager import ApplicationManager


def main(args: argparse.Namespace) -> None:
    """Executes main game engine"""
    application_manager = ApplicationManager(args.window_width, args.window_height)
    application_manager.run_game_loop()


def parse_args(cli_args: list[str]) -> argparse.Namespace:
    """
    Parses CLI arguments for game execution
    :param cli_args: Command Line Arguments
    :type cli_args: list[str]
    :return: Parsed arguments
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--window_width", type=int, default=1200)
    parser.add_argument("--window_height", type=int, default=800)
    args = parser.parse_args(cli_args)
    return args


if __name__ == "__main__":
    """Demo execution file"""
    args = parse_args(sys.argv[1:])
    main(args)
