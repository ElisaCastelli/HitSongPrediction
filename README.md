# Hit Song Prediction System

Python project with the aim of analyzing Hit Song Prediction. 


### Dataset
The dataset from which I've started to train and validate the system proposed is [SpotGenTrack](https://data.mendeley.com/datasets/4m2x4zngny).
This dataset contains: 
* Spotify data of 101940 tracks between songs and podcasts (spotify_track.csv , spotify_album.csv, spotify_artists.csv)
* Low level features computed on each song
* Lyrics features computed on each song
* Preview url for wach one

A cleaning process has been conducted to remove not properly annotated songs, duplicates, songs that contain podcasts and also to update the lyrics using the [MusixMatch API](https://developer.musixmatch.com) since sometime title and lyrics of songs did not match.

I've ended up with two different datasets: one with only English songs and another with multi-lingual songs that can be found in the Datasets directory. Starting from the Spotify ID, in spotify_track.csv the url to download the audio file is [available](Datasets). In these datasets information stored are: spotify_ID, musixmatch_ID, release year, lyrics and popularity score assigned by Spotify in 2019.

<img width="1296" alt="Screenshot 2023-11-03 alle 11 16 42" src="https://github.com/ElisaCastelli/HitSongPrediction/assets/61751277/a20cf51b-7097-4d1a-8ecc-fc052ee30113">

### Model

The model consists in three main components:
* Audio Embedding extractor
* Text Embedding extractor
* Final Multi-Layer Perceptron to predict songs popularity

<img width="860" alt="my_model" src="https://github.com/ElisaCastelli/HitSongPrediction/assets/61751277/6c580c05-53c3-4dca-9c05-f5811b0a387b">

### Repository Structure

* datasets: it contains .csv / .parquet files with english songs and multilingual songs. Additional information of songs (title, artist ecc) are stored in /nas/home/ecastelli/Data Sources folder and divided into 3 files: spotify_album.csv, spotify_artists.csv and spotify_tracks.csv. A join can be done using as key the spotify_id and track_id columns.
* models: it contains the models used for this project. 
    * podcast_discriminator contains a .ipynb notebook used to implement a model for distinguishing audio that contains podcasts and podcasts that conatins music
    * genre_classificator is a model pre-trained on GTZAN Genre that is used as audio feature extractor
    * hsp_model is the final multi-layer perceptron used to predict the song popularity
 
### Train

Starting from the [train.py](train.py) file the training can be started passing three parameters:
* Problem to solve: classification (c) or regression (r)
* Language to consider: english (en) or multilingual (mul)
* Number of popularity classes to be considered in case of classification

To start the training process:

* Install the requirements.txt in your virtual environment
* Check the connection to the ISPL servers to be able to have access to the audio files stored there (/nas/home/ecastelli/thesis/Audio) and to the checkpoint of the pre-trained model ("/nas/home/ecastelli/thesis/models/Model/checkpoint/NuovoGTZAN_best.ckpt"). 
* Change the NeptuneLogger parameters to be able to see logs
* Launch from the main folder *python train.py --help* and follow the instructions

