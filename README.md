# Hit-Song-Prediction

Python project with the aim of analyzing hit song prediction 

### Dataset
The dataset from which I've started to train and validate the system proposed is [SpotGenTrack](https://data.mendeley.com/datasets/4m2x4zngny).
This dataset contains: 
* Spotify data of 101940 tracks between songs and podcasts (spotify_track.csv , spotify_album.csv, spotify_artists.csv)
* Low level features computed on each song
* Lyrics features computed on each song
* Preview url for wach one

A cleaning process has been conducted to remove not properly annotated songs, duplicates, songs that contain podcasts and also to update the lyrics using the [MusixMatch API](https://developer.musixmatch.com) since sometime title and lyrics of songs did not coincide.
I've ended up with two different datasets: one with only English songs and another with multi-lingual songs that can be found in the Datasets directory. Starting from the Spotify ID, in spotify_track.csv the url to download the audio file is available. 
<img width="1296" alt="Screenshot 2023-11-03 alle 11 16 42" src="https://github.com/ElisaCastelli/HitSongPrediction/assets/61751277/a20cf51b-7097-4d1a-8ecc-fc052ee30113">
