from torch import nn

NUM_CLASSES = 4


def select_model(name):
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
            nn.Linear(128, NUM_CLASSES)
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
            nn.Linear(128, NUM_CLASSES)
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
            nn.Linear(128, NUM_CLASSES)

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
            nn.Linear(128, NUM_CLASSES)
        )
    return layers
