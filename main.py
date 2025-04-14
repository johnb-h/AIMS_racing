import argparse

from user_interface.application_manager import ApplicationManager


# Main Function
def main(args):
    application_manager = ApplicationManager(args.window_width, args.window_height)
    application_manager.run_game_loop()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window_width", type=int, default=None, help="Window width in pixels (defaults to screen width)")
    parser.add_argument("--window_height", type=int, default=None, help="Window height in pixels (defaults to screen height)")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
