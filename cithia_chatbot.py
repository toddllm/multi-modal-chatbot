import tkinter as tk
from tkinter import messagebox, filedialog, scrolledtext
from tkinter import ttk
from PIL import Image, ImageTk
import subprocess
import threading
import os
import requests
import json
import queue
import time
import shutil  # For copying files
from gtts import gTTS  # For TTS
from vosk import Model, KaldiRecognizer  # For STT
import sounddevice as sd
import sys
import pygame  # For playing audio and controlling playback
from langchain.memory import ConversationBufferMemory  # LangChain for memory

# -------------------- Chatbot Functionality -------------------- #

def query_ollama(model_name, prompt, token_queue, memory):
    """
    Send the prompt to Ollama API with conversation history and stream the response tokens.
    """
    try:
        # Retrieve conversation history from memory
        conversation_history = memory.load_memory_variables({})["history"]
        full_prompt = conversation_history + "\nUser: " + prompt + "\nBot:"
        
        url = 'http://localhost:11434/api/generate'  # Adjust the URL if necessary
        headers = {'Content-Type': 'application/json'}
        payload = {
            'model': model_name,
            'prompt': full_prompt
        }

        # Send a POST request to the Ollama server
        response = requests.post(url, headers=headers, json=payload, stream=True)

        if response.status_code != 200:
            token_queue.put(None)  # Signal an error or end of response
            print("\nError:", response.text)
            return

        # Stream the response and put each token into the queue
        for line in response.iter_lines():
            if line:
                data = line.decode('utf-8')
                json_data = json.loads(data)
                token = json_data.get('response', '')
                token_queue.put(token)
        token_queue.put(None)  # Signal that the response is complete
    except Exception as e:
        token_queue.put(None)  # Signal an error
        print(f"\nAn exception occurred: {e}")
        return

# -------------------- Image Generation Functionality -------------------- #

def run_diffusionkit(prompt, image_queue, status_queue):
    """
    Generate an image using diffusionkit-cli based on the prompt.
    """
    # Prepare the output path with a unique filename
    timestamp = int(time.time())
    output_filename = f"generated_image_{timestamp}.png"
    output_path = os.path.join(os.getcwd(), output_filename)

    # Build the command
    command = [
        "diffusionkit-cli",
        "--prompt", prompt,
        "--model-version=argmaxinc/mlx-FLUX.1-schnell",
        "--output-path", output_path,
        "--steps", "15"
    ]

    try:
        # Run the command
        result = subprocess.run(command, capture_output=True, text=True)

        if result.returncode != 0:
            # Show error message
            status_queue.put(f"Error: {result.stderr.strip()}")
            print(f"Error: {result.stderr.strip()}")
            image_queue.put(None)
        else:
            # Indicate success and provide the image path
            status_queue.put("Image generated successfully!")
            image_queue.put(output_path)
    except Exception as e:
        status_queue.put(f"Exception: {e}")
        print(f"Exception: {e}")
        image_queue.put(None)

# -------------------- TTS and STT Functionality -------------------- #

def speak(text, audio_controller, tts_enabled):
    """
    Convert chatbot text responses to speech using gTTS and play via pygame.
    """
    if not tts_enabled:
        return  # Do not speak if TTS is disabled
    try:
        tts = gTTS(text=text, lang='en')
        tts.save("response.mp3")
        # Load and play the audio using AudioController
        audio_controller.play_audio("response.mp3")
    except Exception as e:
        print(f"Error in speak function: {e}")
        messagebox.showerror("TTS Error", f"Failed to generate speech: {e}")

def recognize_speech(model):
    """
    Recognize speech input from the user and convert to text using Vosk.
    """
    recognizer = KaldiRecognizer(model, 16000)
    q = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        q.put(bytes(indata))

    try:
        with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                               channels=1, callback=callback):
            print("Listening... Speak into your microphone.")
            while True:
                data = q.get()
                if recognizer.AcceptWaveform(data):
                    result = recognizer.Result()
                    result_dict = json.loads(result)
                    text = result_dict.get("text", "")
                    return text
    except Exception as e:
        print(f"Error in recognize_speech function: {e}")
        return ""

# -------------------- Audio Playback Controller -------------------- #

class AudioController:
    """
    Controller for playing and stopping audio using pygame.
    """
    def __init__(self):
        pygame.mixer.init()
        self.currently_playing = False
        self.lock = threading.Lock()

    def play_audio(self, file_path):
        """
        Play audio file. Stop any currently playing audio.
        """
        def play():
            with self.lock:
                if self.currently_playing:
                    pygame.mixer.music.stop()
                try:
                    pygame.mixer.music.load(file_path)
                    pygame.mixer.music.play()
                    self.currently_playing = True
                except Exception as e:
                    print(f"Error playing audio: {e}")
                    self.currently_playing = False

                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
                self.currently_playing = False
                try:
                    os.remove(file_path)  # Remove the file after playback
                except Exception as e:
                    print(f"Error removing audio file: {e}")

        threading.Thread(target=play, daemon=True).start()

    def stop_audio(self):
        """
        Stop the currently playing audio.
        """
        with self.lock:
            if self.currently_playing:
                pygame.mixer.music.stop()
                self.currently_playing = False

# -------------------- Combined GUI Application -------------------- #

class ChatImageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Cithia Chatbot with Image Generation, TTS, and STT")
        self.model_name = 'llama3.1:latest'  # Update as per your model

        # To keep references to images to prevent garbage collection
        self.image_refs = []
        self.last_image_path = None  # To track the last generated image path

        # Initialize LangChain Memory
        self.memory = ConversationBufferMemory(memory_key="history")

        # Initialize Vosk model
        model_path = "vosk-model-small-en-us-0.15"  # Update with your model path
        if not os.path.exists(model_path):
            messagebox.showerror("Model Not Found", f"Please ensure the Vosk model is located at {model_path}")
            sys.exit(1)
        self.model = Model(model_path)

        # Initialize Audio Controller
        self.audio_controller = AudioController()

        # TTS Toggle Variable
        self.tts_enabled = tk.BooleanVar()
        self.tts_enabled.set(True)  # Default: TTS is enabled

        # Create main frame
        self.main_frame = tk.Frame(root)
        self.main_frame.pack(fill='both', expand=True)

        # Chat display area
        self.chat_display = scrolledtext.ScrolledText(self.main_frame, state='disabled', wrap=tk.WORD, height=20)
        self.chat_display.pack(padx=10, pady=10, fill='both', expand=True)

        # Frame for input and buttons
        input_frame = tk.Frame(self.main_frame)
        input_frame.pack(padx=10, pady=5, fill='x')

        # Entry for user input
        self.chat_entry = tk.Entry(input_frame, width=80)
        self.chat_entry.pack(side='left', expand=True, fill='x', padx=(0, 5))
        self.chat_entry.bind("<Return>", self.send_chat)

        # Send button
        self.send_button = tk.Button(input_frame, text="Send", command=self.send_chat)
        self.send_button.pack(side='left')

        # Voice input button for STT
        self.voice_button = tk.Button(input_frame, text="Speak", command=self.speak_input)
        self.voice_button.pack(side='left', padx=(5,0))

        # Stop speaking button for TTS
        self.stop_button = tk.Button(input_frame, text="Stop Speaking", command=self.stop_speaking)
        self.stop_button.pack(side='left', padx=(5,0))

        # TTS Toggle Checkbutton
        self.tts_toggle = tk.Checkbutton(input_frame, text="Enable TTS", variable=self.tts_enabled)
        self.tts_toggle.pack(side='left', padx=(10,0))

        # Setup menu for saving images
        self.setup_menu()

    def send_chat(self, event=None):
        user_input = self.chat_entry.get().strip()
        if user_input == '':
            return  # Do nothing for empty input

        # Update memory with user input
        self.memory.save_context({"input": user_input}, {"output": ""})

        # Display user message
        self.append_chat("You", user_input)
        self.chat_entry.delete(0, tk.END)

        if user_input.lower().startswith("generate image:"):
            # Extract prompt and start image generation
            prompt = user_input[len("generate image:"):].strip()
            if prompt:
                self.generate_image(prompt)
                # After initiating image generation, ask if the user wants prompt suggestions
                bot_message = "Would you like some tips to refine your prompt for better image results?"
                self.append_chat("Bot", bot_message)
                self.memory.save_context({"input": user_input}, {"output": bot_message})
                speak(bot_message, self.audio_controller, self.tts_enabled.get())
                return
            else:
                bot_message = "Please provide a prompt after 'generate image:'."
                self.append_chat("Bot", bot_message)
                self.memory.save_context({"input": user_input}, {"output": bot_message})
                speak(bot_message, self.audio_controller, self.tts_enabled.get())
            return

        # Handle prompt improvement requests
        if "improve prompt" in user_input.lower() or "refine prompt" in user_input.lower():
            improvement = ("Sure! To create a more detailed and vivid image, consider adding specifics such as colors, lighting, "
                           "environment, emotions, and any unique elements you want to include. For example, instead of 'a cat', "
                           "you might say 'a fluffy white cat lounging on a sunlit windowsill with a playful expression.'")
            self.append_chat("Bot", improvement)
            self.memory.save_context({"input": user_input}, {"output": improvement})
            speak(improvement, self.audio_controller, self.tts_enabled.get())
            return

        # Start chatbot response
        token_queue = queue.Queue()
        threading.Thread(target=query_ollama, args=(self.model_name, user_input, token_queue, self.memory), daemon=True).start()
        self.append_chat("Bot", " ", clear=True)  # Prepare for response

        # Handle token streaming
        def handle_tokens():
            response = ""
            while True:
                try:
                    token = token_queue.get(timeout=5)
                    if token is None:
                        break  # End of response
                    response += token
                    self.update_chat("Bot", response)
                except queue.Empty:
                    break

            # Update memory with chatbot response
            self.memory.save_context({"input": user_input}, {"output": response})

            # Speak the final chatbot response if TTS is enabled
            if response:
                speak(response, self.audio_controller, self.tts_enabled.get())

        threading.Thread(target=handle_tokens, daemon=True).start()

    def speak_input(self):
        """Use STT to get user input."""
        def thread_speak_input():
            user_input = recognize_speech(self.model)
            if user_input:
                self.chat_entry.delete(0, tk.END)
                self.chat_entry.insert(tk.END, user_input)
                self.send_chat()

        threading.Thread(target=thread_speak_input, daemon=True).start()

    def append_chat(self, sender, message, clear=False):
        """
        Append a new message to the chat display.
        """
        self.chat_display.config(state='normal')
        if clear:
            self.chat_display.insert(tk.END, f"{sender}: {message}")
        else:
            self.chat_display.insert(tk.END, f"{sender}: {message}\n")
        self.chat_display.config(state='disabled')
        self.chat_display.see(tk.END)

    def update_chat(self, sender, message):
        """
        Update the last line in the chat display.
        """
        self.chat_display.config(state='normal')
        # Remove the last line (incomplete) and add the updated one
        self.chat_display.delete("end-2l", "end-1l")
        self.chat_display.insert(tk.END, f"{sender}: {message}")
        self.chat_display.config(state='disabled')
        self.chat_display.see(tk.END)

    def generate_image(self, prompt):
        """
        Generate an image based on the user's prompt.
        """
        # Display a placeholder message
        self.append_chat("Bot", "Generating image based on your prompt...")
        speak("Generating image based on your prompt.", self.audio_controller, self.tts_enabled.get())

        # Disable the send, voice, and toggle buttons to prevent multiple clicks
        self.send_button.config(state=tk.DISABLED)
        self.voice_button.config(state=tk.DISABLED)
        self.tts_toggle.config(state=tk.DISABLED)

        image_queue = queue.Queue()
        status_queue = queue.Queue()

        # Start the image generation in a separate thread
        threading.Thread(target=run_diffusionkit, args=(prompt, image_queue, status_queue), daemon=True).start()

        # Update GUI based on the image generation result
        def check_generation():
            try:
                status = status_queue.get_nowait()
                self.append_chat("Bot", status)
                speak(status, self.audio_controller, self.tts_enabled.get())
            except queue.Empty:
                pass

            try:
                image_path = image_queue.get_nowait()
                if image_path:
                    self.display_image(image_path)
                    self.last_image_path = image_path  # Update the last image path
                    msg = "You can save the image using the 'Options' menu."
                    self.append_chat("Bot", msg)
                    speak(msg, self.audio_controller, self.tts_enabled.get())
                else:
                    msg = "Failed to generate image."
                    self.append_chat("Bot", msg)
                    speak(msg, self.audio_controller, self.tts_enabled.get())
                # Re-enable the send, voice, and toggle buttons
                self.send_button.config(state=tk.NORMAL)
                self.voice_button.config(state=tk.NORMAL)
                self.tts_toggle.config(state=tk.NORMAL)
            except queue.Empty:
                self.root.after(100, check_generation)
                return

        self.root.after(100, check_generation)

    def display_image(self, image_path):
        """
        Display the generated image in the chat.
        """
        try:
            # Open and resize the image
            img = Image.open(image_path)
            img.thumbnail((400, 400))  # Adjust size as needed
            img_tk = ImageTk.PhotoImage(img)

            # Insert image into chat
            self.chat_display.config(state='normal')
            self.chat_display.image_create(tk.END, image=img_tk)
            self.chat_display.insert(tk.END, "\n")
            self.chat_display.config(state='disabled')
            self.chat_display.see(tk.END)

            # Keep a reference to prevent garbage collection
            self.image_refs.append(img_tk)
        except Exception as e:
            self.append_chat("Bot", f"Failed to display image:\n{e}")
            speak(f"Failed to display image: {e}", self.audio_controller, self.tts_enabled.get())

    def save_last_image(self):
        """
        Save the last generated image to the user's desired location.
        """
        if not self.last_image_path or not os.path.exists(self.last_image_path):
            messagebox.showinfo("No Image", "There is no image to save.")
            speak("There is no image to save.", self.audio_controller, self.tts_enabled.get())
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
            title="Save Image"
        )
        if file_path:
            try:
                shutil.copy(self.last_image_path, file_path)
                messagebox.showinfo("Image Saved", f"Image saved to {file_path}")
                speak(f"Image saved to {file_path}.", self.audio_controller, self.tts_enabled.get())
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save image:\n{e}")
                speak(f"Failed to save image: {e}", self.audio_controller, self.tts_enabled.get())

    def setup_menu(self):
        """
        Setup the application's menu.
        """
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        options_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Options", menu=options_menu)
        options_menu.add_command(label="Save Last Image", command=self.save_last_image)
        options_menu.add_separator()
        options_menu.add_command(label="Exit", command=self.root.quit)

    def stop_speaking(self):
        """
        Stop the TTS audio playback.
        """
        self.audio_controller.stop_audio()
        system_message = "Stopped speaking."
        self.append_chat("System", system_message)
        speak(system_message, self.audio_controller, self.tts_enabled.get())

# -------------------- Main Function -------------------- #

def main():
    root = tk.Tk()
    app = ChatImageApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
