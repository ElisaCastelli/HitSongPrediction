# Hit-Song-Prediction

Python project with the aim of analyzing Hit Song Prediction. 
* Pytorch Lightning version 2.0.7
* Librosa 0.10.0
* Neptune 1.5.0
* Torch 2.0.1

### Dataset
The dataset from which I've started to train and validate the system proposed is [SpotGenTrack](https://data.mendeley.com/datasets/4m2x4zngny).
This dataset contains: 
* Spotify data of 101940 tracks between songs and podcasts (spotify_track.csv , spotify_album.csv, spotify_artists.csv)
* Low level features computed on each song
* Lyrics features computed on each song
* Preview url for wach one

A cleaning process has been conducted to remove not properly annotated songs, duplicates, songs that contain podcasts and also to update the lyrics using the [MusixMatch API](https://developer.musixmatch.com) since sometime title and lyrics of songs did not coincide.
I've ended up with two different datasets: one with only English songs and another with multi-lingual songs that can be found in the Datasets directory. Starting from the Spotify ID, in spotify_track.csv the url to download the audio file is [available](Datasets). 
<img width="1296" alt="Screenshot 2023-11-03 alle 11 16 42" src="https://github.com/ElisaCastelli/HitSongPrediction/assets/61751277/a20cf51b-7097-4d1a-8ecc-fc052ee30113">

### Model

The model consists in three main components:
* Audio Embedding extractor
* Text Embedding extractor
* Final Multi-Layer Perceptron to predict songs popularity

<img width="860" alt="my_model" src="https://github.com/ElisaCastelli/HitSongPrediction/assets/61751277/6c580c05-53c3-4dca-9c05-f5811b0a387b">


### Train
Starting from the [main.py](main.py) file the training can be started passing three parameters:
* Problem to solve: classification (c) or regression (r)
* Language to consider: english (en) or multilingual (mul)
* Number of popularity classes to be considered in case of classification
