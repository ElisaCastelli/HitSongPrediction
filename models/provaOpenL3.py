import openl3
from torchvision import transforms
import librosa
audio, sr = librosa.load(path="/Users/elisacastelli/Documents/GitHub/HitPrediction/HitSongPrediction/Audio/5qljLQuKnNJf4F4vfxQB0V.mp3", sr=48000)
model = openl3.models.load_audio_embedding_model(input_repr="mel256", content_type="music", embedding_size=512)
emb, ts1 = openl3.get_audio_embedding(audio, sr, model=model)
print(emb)
print(len(emb))
print(emb.shape)