import sys


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "score-building-blocks":
        from .prepare import main as prepare_command_line

        prepare_command_line(sys.argv[2:])
    elif len(sys.argv) > 1 and sys.argv[1] in ('rank', 'generate-and-rank', 'resume', 'retry', 'status', '--json'):
        from .cli import main as scoring_command_line

        return scoring_command_line()
    else:
        from .native import native_launcher
        from applications.scoring.config import ConfigurationError

        try:
            generate_command_line = native_launcher().generate_command_line
        except ConfigurationError as error:
            print(f'{error.code}: {error}\n{error.next_action}', file=sys.stderr)
            return 2

        generate_command_line()


if __name__ == "__main__":
    sys.exit(main())
