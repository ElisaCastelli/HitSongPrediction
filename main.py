import argparse
from models.Model.HSPModel import *


def main():
    parser = argparse.ArgumentParser(
        description='Hit Song Prediction')

    parser.add_argument('--problem', type=str, help='Classification [c] or Regression [r]', default='c')
    parser.add_argument('--language', type=str, help='Multilingual [mul] or English [en]', default='en')
    parser.add_argument('--n_classes', type=int, help='Number of popularity classes', default=4)

    args = parser.parse_args()
    problem = args.problem
    language = args.language
    n_classes = args.n_classes

    hit_song_prediction(problem=problem, language=language, num_classes=n_classes)


if __name__ == '__main__':
    main()
