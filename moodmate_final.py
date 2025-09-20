# moodmate_final.py
"""
MoodMate Final — webcam + stable emotion detection + Spotify playlist + nice UI
Save as moodmate_final.py and run in your venv:
python moodmate_final.py
"""
import pyttsx3
import os
import sys
import cv2
import time
import random
import threading
import collections
import statistics
import webbrowser
import subprocess


from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, messagebox

# Optional libraries (best-effort imports)
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except Exception as e:
    print("DeepFace not available (demo fallback). Error:", e)
    DEEPFACE_AVAILABLE = False

try:
    import pyttsx3
    TTS_AVAILABLE = True
except Exception:
    TTS_AVAILABLE = False

try:
    import pyjokes
    JOKES_AVAILABLE = True
except Exception:
    JOKES_AVAILABLE = False

# ------------------ User Playlist ------------------
PLAYLIST_URI = "spotify:playlist:3Qk05c3iAoUQiHRb5UdfFJ"
PLAYLIST_WEB = "https://open.spotify.com/playlist/3Qk05c3iAoUQiHRb5UdfFJ?si=1e9c450a87684da5&autoplay=true"


# ------------------ Detection config ------------------
ANALYZE_INTERVAL = 1.0      # seconds between model calls
ROLLING_WINDOW = 7          # frames to keep for smoothing
STABLE_THRESHOLD = 3        # mode must appear this many times to be stable

# map raw emotions to categories + emojis
EMOTION_MAP = {
    'happy': 'happy',
    'sad': 'sad',
    'neutral': 'neutral',
    'angry': 'angry',
    'surprise': 'surprise',
    'fear': 'fear',
    'disgust': 'disgust'
}

EMOJI = {
    'happy': "😄",
    'sad':   "😢",
    'neutral': "😐",
    'angry': "😡",
    'surprise': "😲",
    'fear': "😨",
    'disgust': "🤢"
}

MOOD_TEXT = {
    'happy':    ("You're looking great!", "Shall I play your upbeat playlist or a quick game?"),
    'sad':      ("You look sad.", "Would you like me to play your playlist or tell a joke?"),
    'neutral':  ("Feeling neutral.", "Want some music or a short game to spice things up?"),
    'angry':    ("You seem upset.", "Maybe calming music would help — want me to play it?"),
    'surprise': ("Surprised!", "Want a fun fact or a quick game?"),
    'fear':     ("A bit anxious?", "Let's try a calm playlist or a joke."),
    'disgust':  ("Hmm — that's odd.", "Maybe music or a quick laugh will help.")
}

FALLBACK_JOKES = [
    "Why don’t scientists trust atoms? Because they make up everything!",
    "I told my computer I needed a break, and it said: 'No problem — I'll go to sleep.'",
    "Why did the scarecrow win an award? Because he was outstanding in his field!"
]

# ------------------ Utilities ------------------
def speak(text):
    """Speak using pyttsx3 if available, otherwise print to console."""
    if TTS_AVAILABLE:
        def _s():
            try:
                engine = pyttsx3.init()
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                print("TTS error:", e)
                print("[TTS]", text)
        threading.Thread(target=_s, daemon=True).start()
    else:
        print("[TTS disabled] " + text)

def open_spotify(uri=PLAYLIST_URI, web=PLAYLIST_WEB):
    """Attempt to open Spotify desktop via URI; fallback to web URL with autoplay."""
    try:
        if sys.platform.startswith("win"):
            # Try to open desktop Spotify app
            os.startfile(uri)
            return
        elif sys.platform.startswith("darwin"):
            subprocess.call(["open", uri])
            return
        else:
            subprocess.call(["xdg-open", uri])
            return
    except Exception:
        # fallback to web with autoplay
        autoplay_url = web + "&autoplay=true"
        webbrowser.open(autoplay_url)

def speak_message(message):
    """Make the assistant speak a message aloud."""
    engine.say(message)
    engine.runAndWait()


# ------------------ Emotion Analyzer Thread ------------------
class EmotionAnalyzer(threading.Thread):
    def __init__(self, frame_provider):
        super().__init__(daemon=True)
        self.get_frame = frame_provider
        self.window = collections.deque(maxlen=ROLLING_WINDOW)
        self.lock = threading.Lock()
        self.running = True
        self.paused = False
        self.stable_emotion = None

    def run(self):
        while self.running:
            if self.paused:
                time.sleep(0.15)
                continue
            frame = self.get_frame()
            if frame is None:
                time.sleep(0.15)
                continue
            pred = self._predict(frame)
            with self.lock:
                self.window.append(pred)
                mode_val = None
                try:
                    mode_val = statistics.mode(list(self.window))
                except Exception:
                    mode_val = None
                if mode_val:
                    count_mode = sum(1 for x in self.window if x == mode_val)
                    if count_mode >= STABLE_THRESHOLD:
                        self.stable_emotion = mode_val
            time.sleep(ANALYZE_INTERVAL)

    def _predict(self, frame_bgr):
        # frame_bgr: BGR numpy array from OpenCV
        if DEEPFACE_AVAILABLE:
            try:
                # DeepFace prefers RGB for many wrappers; convert to RGB
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                res = DeepFace.analyze(rgb, actions=['emotion'], enforce_detection=False)
                if isinstance(res, list):
                    res = res[0]
                # Try multiple keys depending on version
                dom = res.get('dominant_emotion') or res.get('dominant_emotion', None)
                if not dom:
                    # sometimes result has 'emotion' dict
                    emdict = res.get('emotion', {})
                    if emdict:
                        dom = max(emdict, key=emdict.get)
                if dom:
                    dom_l = dom.lower()
                    mapped = EMOTION_MAP.get(dom_l, 'neutral')
                    return mapped
            except Exception as e:
                print("DeepFace analyze error -> fallback to demo:", e)
                return random.choice(list(EMOJI.keys()))
        # Demo fallback: random stable choice (but chosen from MOOD_TEXT keys)
        return random.choice(list(MOOD_TEXT.keys()))

    def get_stable(self):
        with self.lock:
            return self.stable_emotion

    def pause(self):
        with self.lock:
            self.paused = True

    def resume(self):
        with self.lock:
            self.paused = False
            self.window.clear()
            self.stable_emotion = None

    def stop(self):
        self.running = False

# ------------------ Main GUI App ------------------
class MoodMateApp:
    def __init__(self, root):
        self.root = root
        root.title("MoodMate")
        root.configure(bg="#0f1724")
        # Start full-screen
        root.attributes("-fullscreen", True)
        root.bind("<Escape>", lambda e: root.attributes("-fullscreen", False))

        # Styles
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TLabel", background="#0f1724", foreground="#e6eef8", font=("Segoe UI", 12))
        style.configure("Title.TLabel", font=("Segoe UI", 20, "bold"))
        style.configure("TButton", padding=6, font=("Segoe UI", 11))

        # Layout
        self.left = tk.Frame(root, bg="#071028")
        self.left.place(relx=0.02, rely=0.04, relwidth=0.62, relheight=0.88)
        self.right = tk.Frame(root, bg="#0b1220")
        self.right.place(relx=0.66, rely=0.04, relwidth=0.32, relheight=0.88)

        # Video label
        self.video_label = tk.Label(self.left, bg="#000000")
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=18, pady=18)

        # Right: emoji + text + buttons
        ttk.Label(self.right, text="MoodMate", style="Title.TLabel").pack(pady=(20,6))
        self.emoji_lbl = ttk.Label(self.right, text="🙂", font=("Segoe UI Emoji", 64))
        self.emoji_lbl.pack(pady=(6,6))
        self.title_lbl = ttk.Label(self.right, text="Waiting for your face...", font=("Segoe UI", 14))
        self.title_lbl.pack(pady=(6,4))
        self.subtitle_lbl = ttk.Label(self.right, text="Bring your face close and stay still for a moment.", wraplength=280, justify="center")
        self.subtitle_lbl.pack(pady=(2,10))

        btn_frame = tk.Frame(self.right, bg="#0b1220")
        btn_frame.pack(pady=10)
        ttk.Button(btn_frame, text="Play Playlist", command=self.action_play_playlist, width=28).pack(pady=6)
        ttk.Button(btn_frame, text="Tell me a joke", command=self.action_tell_joke, width=28).pack(pady=6)
        ttk.Button(btn_frame, text="Play Guess Number", command=self.action_play_guess, width=28).pack(pady=6)
        ttk.Button(btn_frame, text="Play Rock-Paper-Scissors", command=self.action_play_rps, width=28).pack(pady=6)

        control = tk.Frame(self.right, bg="#0b1220")
        control.pack(pady=(14,6))
        ttk.Button(control, text="Resume Detection", command=self.resume_detection).pack(side=tk.LEFT, padx=6)
        ttk.Button(control, text="Pause Detection", command=self.pause_detection).pack(side=tk.LEFT, padx=6)

        self.status_label = ttk.Label(root, text="Status: Initializing camera...", style="TLabel")
        self.status_label.place(relx=0.02, rely=0.94)

        # Setup camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Camera error", "Cannot access webcam. Close other apps and try again.")
            root.destroy()
            return

        # Analyzer thread
        self.analyzer = EmotionAnalyzer(self.get_frame_for_analyzer)
        self.analyzer.start()

        self.current_displayed = None
        self.last_trigger = 0

        # start loops
        self.update_video()
        self.check_stable_periodic()

    def get_frame_for_analyzer(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        # use original (non-mirrored) for analysis
        return frame.copy()

    def update_video(self):
        ret, frame = self.cap.read()
        if ret:
            # mirror for natural user view
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            # scale to left frame size
            w = max(200, int(self.left.winfo_width() - 36))
            h = max(150, int(self.left.winfo_height() - 36))
            img = img.resize((w, h))
            imgtk = ImageTk.PhotoImage(img)
            self.video_label.imgtk = imgtk
            self.video_label.config(image=imgtk)
        self.root.after(30, self.update_video)

    def check_stable_periodic(self):
        stable = self.analyzer.get_stable()
        if stable and stable != self.current_displayed:
            now = time.time()
            if now - self.last_trigger > 1.0:
                self.current_displayed = stable
                self.on_new_emotion(stable)
                self.last_trigger = now
        self.root.after(600, self.check_stable_periodic)

    def on_new_emotion(self, mood):
        emoji = EMOJI.get(mood, "😐")
        title, subtitle = MOOD_TEXT.get(mood, MOOD_TEXT['neutral'])
        self.emoji_lbl.config(text=emoji)
        self.title_lbl.config(text=title)
        self.subtitle_lbl.config(text=subtitle)
        self.status_label.config(text=f"Status: Detected '{mood}'")
        # speak prompt
        speak(f"{title} {subtitle} Would you like me to play music, tell a joke or play a game?")
        # pause analyzer so it doesn't keep popping up dialogs
        self.analyzer.pause()
        # show suggestions dialog
        self.show_suggestion_dialog(mood)

    def show_suggestion_dialog(self, mood):
        meta_title, meta_sub = MOOD_TEXT.get(mood, MOOD_TEXT['neutral'])
        win = tk.Toplevel(self.root)
        win.title("Suggestions")
        win.state("zoomed")
        win.configure(bg="#0b1220")
        win.transient(self.root)
        win.grab_set()  # modal

        ttk.Label(win, text=f"Detected mood: {mood}", font=("Segoe UI", 13)).pack(pady=(12,6))
        ttk.Label(win, text=meta_title, font=("Segoe UI", 11, "bold")).pack(pady=(0,4))
        ttk.Label(win, text=meta_sub, wraplength=380, justify="center").pack(pady=(0,12))

        btnf = tk.Frame(win, bg="#0b1220")
        btnf.pack(pady=8)

        def _play():
            open_spotify()
            # keep paused to avoid re-detection interfering; user presses Resume when ready
            win.destroy()

        def _joke():
            win.destroy()
            self.show_joke_window()

        def _game():
            win.destroy()
            self.action_play_guess()

        ttk.Button(btnf, text="Play Playlist", command=_play, width=18).pack(pady=6)
        ttk.Button(btnf, text="Tell me a joke", command=_joke, width=18).pack(pady=6)
        ttk.Button(btnf, text="Play a Game", command=_game, width=18).pack(pady=6)

        def on_close():
            win.destroy()
            # remain paused until user clicks Resume Detection
        win.protocol("WM_DELETE_WINDOW", on_close)

    def resume_detection(self):
        self.analyzer.resume()
        self.current_displayed = None
        self.status_label.config(text="Status: Detection resumed")
        speak("Resuming detection.")

    def pause_detection(self):
        self.analyzer.pause()
        self.status_label.config(text="Status: Detection paused")
        speak("Paused detection.")

    # ---------- Actions ----------
    def action_play_playlist(self):
        open_spotify()

    def action_tell_joke(self):
        self.show_joke_window()

    def action_play_guess(self):
        self.play_guess_number()

    def action_play_rps(self):
        self.play_rps()

    # ---------- Joke window (cartoon) ----------
    def show_joke_window(self):
        if JOKES_AVAILABLE:
            try:
                joke = pyjokes.get_joke()
            except Exception:
                joke = random.choice(FALLBACK_JOKES)
        else:
            joke = random.choice(FALLBACK_JOKES)

        jw = tk.Toplevel(self.root)
        jw.title("A little laugh")
        jw.state("zoomed")
        jw.configure(bg="#0f1724")
        jw.transient(self.root)

        # cartoon canvas
        c = tk.Canvas(jw, width=160, height=160, bg="#0f1724", highlightthickness=0)
        c.place(x=20, y=20)
        # simple face
        c.create_oval(10,10,150,150, fill="#ffd580", outline="")
        c.create_oval(40,50,60,70, fill="#111", outline="")
        c.create_oval(100,50,120,70, fill="#111", outline="")
        c.create_arc(40,70,120,120, start=200, extent=140, style=tk.ARC, width=3)

        # text area
        txt = tk.Label(jw, text=joke, wraplength=340, justify="left", bg="#0f1724", fg="#e6eef8", font=("Segoe UI", 12))
        txt.place(x=200, y=60)

        # buttons
        ttk.Button(jw, text="Speak", command=lambda: speak(joke)).place(x=220, y=220)
        ttk.Button(jw, text="Close", command=jw.destroy).place(x=320, y=220)

    # ---------- Guess the number ----------
    def play_guess_number(self):
        gw = tk.Toplevel(self.root)
        gw.title("Guess the Number")
        gw.state("zoomed")   # <--- ADD THIS
        gw.configure(bg="#071028")
        gw.transient(self.root)

        secret = random.randint(1, 30)
        ttk.Label(gw, text="Guess a number between 1 and 30", font=("Segoe UI", 12)).pack(pady=12)
        entry = tk.Entry(gw, font=("Segoe UI", 12))
        entry.pack(pady=6)

        res_lbl = ttk.Label(gw, text="", font=("Segoe UI", 11))
        res_lbl.pack(pady=6)

        def check():
            val = entry.get().strip()
            if not val.isdigit():
                messagebox.showinfo("Invalid", "Please enter a number.")
                return
            g = int(val)
            if g == secret:
                res_lbl.config(text="Correct! You got it 🎉")
                speak("Great job! You guessed correctly.")
                ttk.Button(gw, text="Close", command=gw.destroy).pack(pady=8)
            elif g < secret:
                res_lbl.config(text="Too low. Try higher.")
            else:
                res_lbl.config(text="Too high. Try lower.")

        ttk.Button(gw, text="Guess", command=check).pack(pady=6)

    # ---------- RPS ----------
    def play_rps(self):
        rw = tk.Toplevel(self.root)
        rw.title("Rock Paper Scissors")
        rw.state("zoomed")
        rw.configure(bg="#071028")
        rw.transient(self.root)

        ttk.Label(rw, text="Rock • Paper • Scissors", font=("Segoe UI", 14)).pack(pady=12)
        res_lbl = ttk.Label(rw, text="Make your move", font=("Segoe UI", 12))
        res_lbl.pack(pady=6)

        def play(user):
            moves = ["Rock", "Paper", "Scissors"]
            comp = random.choice(moves)
            if user == comp:
                res = f"Tie — both {user}"
            elif (user == "Rock" and comp == "Scissors") or (user == "Paper" and comp == "Rock") or (user == "Scissors" and comp == "Paper"):
                res = f"You win! {user} beats {comp}"
            else:
                res = f"You lose — {comp} beats {user}"
            res_lbl.config(text=res)
            speak(res)

        frm = tk.Frame(rw, bg="#071028")
        frm.pack(pady=14)
        ttk.Button(frm, text="Rock", command=lambda: play("Rock"), width=10).grid(row=0, column=0, padx=6)
        ttk.Button(frm, text="Paper", command=lambda: play("Paper"), width=10).grid(row=0, column=1, padx=6)
        ttk.Button(frm, text="Scissors", command=lambda: play("Scissors"), width=10).grid(row=0, column=2, padx=6)

    def close(self):
        try:
            self.analyzer.stop()
        except Exception:
            pass
        try:
            self.cap.release()
        except Exception:
            pass
        self.root.quit()

# ------------------ Run ------------------
def main():
    root = tk.Tk()
    app = MoodMateApp(root)
    root.protocol("WM_DELETE_WINDOW", app.close)
    root.mainloop()

if __name__ == "__main__":
    main()
