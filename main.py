import argparse
from user_interface.application_manager import ApplicationManager

def main(args):
    app_manager = ApplicationManager(args.window_width, args.window_height)
    app_manager.run_game_loop()

def parse_args():
    parser = argparse.ArgumentParser(description="Run the Evolving Cars game.")
    parser.add_argument("--window_width", type=int, default=None, help="Window width in pixels (defaults to screen width)")
    parser.add_argument("--window_height", type=int, default=None, help="Window height in pixels (defaults to screen height)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)

