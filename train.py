import argparse

from models.hsp_model.HSPModel import *
import typer

def main(
    problem: str = "c", 
    language: str = "en", 
    n_classes: int = 3
):
    """
        Main function for running the hit song prediction model.

        problem: The problem to be solved by the model. Admissible values: 'c' for classification or 'r' for regression.
        language: The language used for the model. Admissible values:'en' for considering only english songs and 'mul' to consider multilingual songs.
        n_classes: The number of classes for the model, use it only in case of classification.

    """
    
    hit_song_prediction(problem=problem, language=language, num_classes=n_classes)


if __name__ == "__main__":
    typer.run(main, )
