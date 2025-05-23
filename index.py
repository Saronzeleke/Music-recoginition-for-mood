import librosa
import numpy as np

from tensorflow.keras import layers, models
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import os
from typing import List, Optional
from pydub import AudioSegment

app = FastAPI(title="Protestant Mood & Context Music Recommender with Streaming")

class Mood:
    def __init__(self):
        self.mood = None

    def set_mood(self, mood: str) -> int:
        
        self.mood = mood.lower()
        if self.mood == "happy":
            self.mood = 0
        elif self.mood == "sad":
            self.mood = 1
        elif self.mood == "angry":
            self.mood = 2
        else:
            self.mood = 3
        return self.mood

    def get_mood(self) -> int:
        
        return self.mood

class Music(Mood):
    def __init__(self):
        super().__init__()
        self.model = self.build_context_model()
        self.command_model = self.build_command_model()
        self.music_library = {
            0: [  
                {"title": "Amazing Grace", "artist": "Traditional (John Newton)", "duration": 4.0, "tempo": 90, "mood": "happy", "stream_url": "https://www.youtube.com/watch?v=CDdvReNKKuk"},
                {"title": "Haleluya", "artist": "Aster Abebe", "duration": 3.8, "tempo": 100, "mood": "happy", "stream_url": "https://youtu.be/0AVT8Vp6Jxs?si=7N_g3Bw6M4heK-yp"}
            ],
            1: [  
                {"title": "What a Friend We Have in Jesus", "artist": "Joseph M. Scriven", "duration": 4.2, "tempo": 70, "mood": "sad", "stream_url": "https://www.youtube.com/watch?v=8SCorW9r_Is"},
                {"title": "Rock of Ages", "artist": "Augustus Toplady", "duration": 3.5, "tempo": 65, "mood": "sad", "stream_url": "https://www.youtube.com/watch?v=gM7gt_cSxjw"}
            ],
            2: [  
                {"title": "A Mighty Fortress Is Our God", "artist": "Martin Luther", "duration": 4.1, "tempo": 120, "mood": "angry", "stream_url": "https://www.youtube.com/watch?v=O6k8DFb8fWs"},
                {"title": "Nothing but the Blood of Jesus", "artist": "Robert Lowry", "duration": 3.3, "tempo": 110, "mood": "angry", "stream_url": "https://www.youtube.com/watch?v=BeVZG7J_8e4"}
            ],
            3: [  
                {"title": "How Great Thou Art", "artist": "Carl Boberg", "duration": 4.8, "tempo": 80, "mood": "relaxed", "stream_url": "https://www.youtube.com/watch?v=Cc0QVWzCv9k"},
                {"title": "Be Thou My Vision", "artist": "Traditional Irish", "duration": 3.7, "tempo": 85, "mood": "relaxed", "stream_url": "https://www.youtube.com/watch?v=5XZ3ja-quE0"}
            ]
        }
        self.command_to_mood = {
            "ayzosh": 1,  
            "selam": 0,
            "other": 3
        }
        # Optional Spotify setup for ETEX (uncomment and configure)
        # import spotipy
        # from spotipy.oauth2 import SpotifyClientCredentials
        # self.sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
        #     client_id="YOUR_CLIENT_ID",
        #     client_secret="YOUR_CLIENT_SECRET"
        # ))
        # self.music_library = {
        #     0: [{"title": "Amazing Grace", "artist": "Traditional (John Newton)", "duration": 4.0, "tempo": 90, "mood": "happy", "stream_url": self.get_spotify_url("3z9U1N6zqHK7qZlbS0hS30")}],
        #     1: [{"title": "What a Friend We Have in Jesus", "artist": "Joseph M. Scriven", "duration": 4.2, "tempo": 70, "mood": "sad", "stream_url": self.get_spotify_url("5z9z8q0q4q8q1z2z3z4z5z")}],
        #     2: [{"title": "A Mighty Fortress Is Our God", "artist": "Martin Luther", "duration": 4.1, "tempo": 120, "mood": "angry", "stream_url": self.get_spotify_url("6z9z8q0q4q8q1z2z3z4z5z")}],
        #     3: [{"title": "How Great Thou Art", "artist": "Carl Boberg", "duration": 4.8, "tempo": 80, "mood": "relaxed", "stream_url": self.get_spotify_url("7z9z8q0q4q8q1z2z3z4z5z")}]
        # }

    def get_spotify_url(self, track_id: str) -> str:
       
        track = self.sp.track(track_id)
        return track["preview_url"] if track["preview_url"] else track["external_urls"]["spotify"]

    def build_context_model(self):
       
        inputs = layers.Input(shape=(20, 1))
        x = layers.Conv1D(64, kernel_size=3, activation='relu')(inputs)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.LSTM(32, return_sequences=False)(x)
        x = layers.Dense(16, activation='relu')(x)
        context_input = layers.Input(shape=(3,))
        combined = layers.Concatenate()([x, context_input])
        x = layers.Dense(32, activation='relu')(combined)
        outputs = layers.Dense(4, activation='softmax')(x)
        model = models.Model(inputs=[inputs, context_input], outputs=outputs)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def build_command_model(self):
       
        model = models.Sequential([
            layers.Dense(64, activation='relu', input_shape=(20,)),
            layers.Dense(32, activation='relu'),
            layers.Dense(3, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def extract_audio_features(self, audio_file: str):
       
        try:
            y, sr = librosa.load(audio_file)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            return np.mean(mfcc.T, axis=0)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error processing audio: {e}")

    def recommend_music(self, place: str = None, context: str = None) -> List[dict]:
      
        songs = self.music_library.get(self.mood, self.music_library[3])
        if place == "taxi":
            songs = [s for s in songs if 3 <= s["duration"] <= 5]
        if context == "period_relaxation":
            songs = [s for s in songs if s["tempo"] < 100 and s["mood"] in ["sad", "relaxed"]]
        return songs[:2]

    def analyze_voice_mood(self, voice_file: str, place: str = None, context: str = None):
        
        features = self.extract_audio_features(voice_file)
        if features is not None:
            features = features.reshape(1, 20, 1)
            place_code = 1 if place == "taxi" else 0
            period_flag = 1 if context == "period_relaxation" else 0
            context_inputs = np.array([[self.mood or 3, place_code, period_flag]])
            prediction = self.model.predict([features, context_inputs], verbose=0)
            self.mood = np.argmax(prediction, axis=1)[0]
            return self.mood
        return None

    def recognize_command(self, audio_file: str):
        
        features = self.extract_audio_features(audio_file)
        if features is not None:
            features = features.reshape(1, -1)
            prediction = self.command_model.predict(features, verbose=0)
            command_idx = np.argmax(prediction, axis=1)[0]
            commands = ["Don't worry", "Haleluya", "It's okay"]
            command = commands[command_idx]
            self.mood = self.command_to_mood[command]
            return command
        return None

    def generate_playlist_message(self, place: str = None, context: str = None) -> str:
       if self.mood is None:
           raise HTTPException(status_code=400,detail="Mood not sent 🤦‍♀️")
       
       mood_name = {0: "happy", 1: "sad", 2: "angry", 3: "relaxed"}[self.mood]
       if context == "period_relaxation" and place == "taxi":
           return f"Ayzosh, relax in the taxi with these soothing Protestant {mood_name} hymns."
       elif place == "taxi":
           return f"Enjoy your taxi ride with these Protestant {mood_name} hymns!"
       elif context == "period_relaxation":
           return f"Ayzosh, God is with you  with these Protestant {mood_name} hymns for your comfort."
       return f"Here's some Protestant {mood_name} music for your worship!"

    def generate_preview(self, stream_url: str, output_file: str, duration: int = 10000):
      
        try:
        
            audio = AudioSegment.from_file("sample_song.mp3")
            preview = audio[:duration]
            preview.export(output_file, format="wav")
            return output_file
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error generating preview: {e}")

music_recommender = Music()

class MoodInput(BaseModel):
    mood: str

class ContextInput(BaseModel):
    mood: Optional[str] = None
    place: Optional[str] = None
    context: Optional[str] = None

@app.post("/set-mood", response_model=dict)
async def set_mood(mood_input: MoodInput):
   
    mood_code = music_recommender.set_mood(mood_input.mood)
    return {"mood": mood_input.mood, "mood_code": mood_code}

@app.get("/recommend-music", response_model=dict)
async def get_recommendations():
   
    if music_recommender.get_mood() is None:
        raise HTTPException(status_code=400, detail="Mood not set 😊. Use /set-mood first.")
    songs = music_recommender.recommend_music()
    return {"mood_code": music_recommender.get_mood(), "recommended_songs": songs}

@app.post("/analyze-voice", response_model=dict)
async def analyze_voice(file: UploadFile = File(...), place: Optional[str] = None, context: Optional[str] = None):
    
    if not file.filename.endswith((".wav", ".mp3")):
        raise HTTPException(status_code=400, detail="Only WAV or MP3 files supported.")
    
    temp_file = f"temp_{file.filename}"
    with open(temp_file, "wb") as buffer:
        buffer.write(await file.read())
    
    try:
        mood_code = music_recommender.analyze_voice_mood(temp_file, place, context)
        if mood_code is None:
            raise HTTPException(status_code=400, detail="Failed to analyze voice file.")
        songs = music_recommender.recommend_music(place, context)
        return {
            "predicted_mood_code": mood_code,
            "mood": {0: "happy", 1: "sad", 2: "angry", 3: "relaxed"}[mood_code],
            "recommended_songs": songs
        }
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

@app.post("/recognize-command", response_model=dict)
async def recognize_command(file: UploadFile = File(...)):
    
    if not file.filename.endswith((".wav", ".mp3")):
        raise HTTPException(status_code=400, detail="Only WAV or MP3 files supported.")
    
    temp_file = f"temp_{file.filename}"
    with open(temp_file, "wb") as buffer:
        buffer.write(await file.read())
    
    try:
        command = music_recommender.recognize_command(temp_file)
        if command is None:
            raise HTTPException(status_code=400, detail="Failed to recognize command.")
        songs = music_recommender.recommend_music()
        return {
            "command": command,
            "mood_code": music_recommender.get_mood(),
            "mood": {0: "happy", 1: "sad", 2: "angry", 3: "relaxed"}[music_recommender.get_mood()],
            "recommended_songs": songs
        }
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

@app.post("/recommend-context", response_model=dict)
async def recommend_context(context_input: ContextInput):
    
    if context_input.mood:
        music_recommender.set_mood(context_input.mood)
    if music_recommender.get_mood() is None:
        raise HTTPException(status_code=400, detail="Mood not set.🤦‍♀️")
    songs = music_recommender.recommend_music(context_input.place, context_input.context)
    return {
        "mood_code": music_recommender.get_mood(),
        "mood": {0: "happy", 1: "sad", 2: "angry", 3: "relaxed"}[music_recommender.get_mood()],
        "place": context_input.place,
        "context": context_input.context,
        "recommended_songs": songs
    }

@app.post("/playlist-message", response_model=dict)
async def get_playlist_message(context_input: ContextInput):
    
    if context_input.mood:
        music_recommender.set_mood(context_input.mood)
    if music_recommender.get_mood() is None:
        raise HTTPException(status_code=400, detail="Mood not set 🤦‍♀️")
    message = music_recommender.generate_playlist_message(context_input.place, context_input.context)
    songs = music_recommender.recommend_music(context_input.place, context_input.context)
    return {
        "message": message,
        "mood_code": music_recommender.get_mood(),
        "mood": {0: "happy", 1: "sad", 2: "angry", 3: "relaxed"}[music_recommender.get_mood()],
        "place": context_input.place,
        "context": context_input.context,
        "recommended_songs": songs
    }

@app.get("/stream-song", response_model=dict)
async def stream_song(title: str):
    
    for mood_songs in music_recommender.music_library.values():
        for song in mood_songs:
            if song["title"] == title:
                return {"title": title, "stream_url": song["stream_url"]}
    raise HTTPException(status_code=404, detail="Song not found.😢")

@app.get("/generate-preview", response_model=dict)
async def generate_preview(title: str):
  
    for mood_songs in music_recommender.music_library.values():
        for song in mood_songs:
            if song["title"] == title:
                output_file = f"preview_{title.replace(' ', '_')}.wav"
                preview_file = music_recommender.generate_preview(song["stream_url"], output_file)
                return {"title": title, "preview_file": output_file}
    raise HTTPException(status_code=404, detail="Song not found.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)