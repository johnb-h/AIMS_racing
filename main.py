"""
main.py
    Main run file for AIMS racing application
"""

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
    :type cli_args: list[str]
    :param cli_args: Command Line Arguments
    :return: Parsed arguments
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(description="Run the Evolving Cars game.")
    parser.add_argument("--window_width", type=int, default=None, help="Window width in pixels (defaults to screen width)")
    parser.add_argument("--window_height", type=int, default=None, help="Window height in pixels (defaults to screen height)")
    return parser.parse_args()

if __name__ == "__main__":
    """Demo execution file"""
    args = parse_args(sys.argv[1:])
    main(args)

