from aclick import group
from .dataset.dataset import generate_dataset_command as generate_dataset_command

@group
def main():
    pass

main.add_command(generate_dataset_command)

if __name__ == "__main__":
    main()
