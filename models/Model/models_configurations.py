from torch import nn


def select_model(name, num_classes):
    """
        Returns the model configuration based on the name received in input (problem-language)
        6 configurations:
            - c-en: classification with english dataset (double text embeddings)
            - r-en: regression with english dataset
            - c-mul: classification with multilingual dataset (double text embeddings)
            - r-mul: regression with multilingual dataset
            - c-onlyaudio-en: classification without text embeddings using english dataset
            - c-oneembedding-en: classification with english dataset (single text embeddings)
    """
    layers = None
    if name == "r-en":
        # REGRESSION ENGLISH
        layers = nn.Sequential(
            nn.BatchNorm1d(3585),
            nn.Dropout(p=0.5),
            nn.Linear(3585, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.BatchNorm1d(512),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.BatchNorm1d(128),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    # REGRESSION MULTI
    elif name == "r-mul":
        layers = nn.Sequential(
            nn.BatchNorm1d(3585),
            nn.Dropout(p=0.5),
            nn.Linear(3585, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.BatchNorm1d(512),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.BatchNorm1d(128),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
    elif name == "c-mul":
        # MULTI-LANG
        layers = nn.Sequential(
            nn.BatchNorm1d(3585),
            nn.Dropout(p=0.5),
            nn.Linear(3585, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.BatchNorm1d(512),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.BatchNorm1d(128),
            nn.Linear(128, num_classes)
        )
    elif name == "c-en":
        # ENG WEIGHTED TEXT EMBEDDINGS
        layers = nn.Sequential(
            nn.BatchNorm1d(3585),
            nn.Dropout(p=0.5),
            nn.Linear(3585, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.BatchNorm1d(512),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.BatchNorm1d(128),
            nn.Linear(128, num_classes)
        )
    elif name == "c-onlyaudio-en":
        # SOLO ENG SOLO AUDIO
        layers = nn.Sequential(
            nn.BatchNorm1d(2049),
            nn.Dropout(p=0.5),
            nn.Linear(2049, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.BatchNorm1d(512),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.BatchNorm1d(128),
            nn.Linear(128, num_classes)

        )

    elif name == "c-oneembedding-en":
        # ENG E ONE TEXT EMBEDDING
        layers = nn.Sequential(
            nn.BatchNorm1d(2817),
            nn.Dropout(p=0.5),
            nn.Linear(2817, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.BatchNorm1d(512),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.BatchNorm1d(128),
            nn.Linear(128, num_classes)
        )
    return layers
