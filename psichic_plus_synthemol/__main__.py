import sys


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "score-building-blocks":
        from .prepare import main as prepare_command_line

        prepare_command_line(sys.argv[2:])
    else:
        from synthemol.generate.generate import generate_command_line

        generate_command_line()


if __name__ == "__main__":
    main()
