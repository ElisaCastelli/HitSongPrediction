import typer
from models.genre_classificator.GTZANPreTrained import pre_train

def launch_pretrain():
    pre_train()

if __name__ == "__main__":
    typer.run(launch_pretrain)