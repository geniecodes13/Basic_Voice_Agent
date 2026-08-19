import sys
import re
import ollama
import multiprocessing
import sounddevice as sd
from RealtimeSTT import AudioToTextRecorder
from pykokoro import KokoroPipeline, PipelineConfig, GenerationConfig



# i didn't do the commenting in code
# this code is a mixture of my brain and AI, and somehow its working. don't change anything!!
# do read requirements.txt first.


# MODALS CONFIG

OLLAMA_MODEL = "llama3.2:1b"
KOKORO_VOICE = "af_heart"  # Options: af_heart, af_bella, am_adam, am_michael, etc.
SAMPLE_RATE = 24000         # Kokoro default sample rate

print("Initializing PyKokoro TTS Pipeline...")
kokoro_config = PipelineConfig(
    voice=KOKORO_VOICE,
    provider="cpu",          # Change to "cuda" if using pykokoro[gpu]
    model_quality="q8",      # Quantized for lower CPU overhead (fp32, fp16, q8, q4)
    generation=GenerationConfig(speed=1.0)
)
tts_pipeline = KokoroPipeline(kokoro_config)



# TTS

def speak_text(text: str):
    """Synthesizes text using PyKokoro and streams it to speaker."""
    clean_text = text.strip()
    if not clean_text:
        return
        
    try:
        # Run synthesis through PyKokoro
        res = tts_pipeline.run(clean_text)
        audio_data = res.audio
        sr = getattr(res, "sample_rate", SAMPLE_RATE)
        
        # Play audio buffer directly
        sd.play(audio_data, samplerate=sr)
        sd.wait()  # Block until the current sentence finishes speaking
    except Exception as e:
        print(f"\n[TTS Error]: {e}")


# BRAIN-OLLAMA

def process_llm_and_speak(prompt: str):
    """
    Streams output tokens from Ollama, groups them into sentences on the fly,
    and feeds each sentence to Kokoro as soon as possible to eliminate latency.
    """
    print(f"\nUser: {prompt}")
    print("Agent: ", end="", flush=True)

    try:
        response_stream = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful voice assistant. Keep answers concise (1 to 2 sentences max)."
                },
                {"role": "user", "content": prompt}
            ],
            stream=True
        )

        sentence_buffer = ""
        # Regex matches punctuation sentence boundaries (. ! ?) followed by whitespace
        sentence_end_regex = re.compile(r'(?<=[.!?])\s+')

        for chunk in response_stream:
            token = chunk['message']['content']
            print(token, end="", flush=True)
            sentence_buffer += token

            # Split buffer into sentences
            parts = sentence_end_regex.split(sentence_buffer)
            if len(parts) > 1:
                # Speak complete sentences, keep the unfinished trailing fragment
                for sentence in parts[:-1]:
                    speak_text(sentence)
                sentence_buffer = parts[-1]

        # Speak any remaining text in the buffer after streaming ends
        if sentence_buffer.strip():
            speak_text(sentence_buffer)
            
        print("\n")

    except Exception as e:
        print(f"\n[Ollama Error]: {e}")


# STT

def on_transcription(transcript: str):
    """Callback function triggered when RealtimeSTT finishes capturing speech."""
    user_text = transcript.strip()
    if not user_text:
        return

    # Check for termination keywords
    if user_text.lower().rstrip('.') in ["exit", "quit", "stop", "goodbye"]:
        print("\nExit keyword received. Goodbye!")
        speak_text("Goodbye!")
        sys.exit(0)

    process_llm_and_speak(user_text)


   
if __name__ == '__main__':
    print("Initializing RealtimeSTT Audio Recorder...")
    multiprocessing.freeze_support()

    recorder = AudioToTextRecorder(
        model="tiny.en",
        language="en",
        compute_type="int8",
        post_speech_silence_duration=0.6,
        spinner=True
    )
    speak_text("Voice agent activated. Speak into your microphone.")
    print("\n[Voice Agent Ready] Speak into microphone. Press Ctrl+C to stop.\n")

    try:
        while True:
            # Main audio-to-text processing loop
            recorder.text(on_transcription)

    except KeyboardInterrupt:
        print("\n\nShutting down pipeline gracefully...")
    
    finally:
        # Clean up resources to prevent Windows BrokenPipeError
        sd.stop()             # Kill active TTS audio
        recorder.stop()       # Stop active audio stream
        recorder.shutdown()   # Close multiprocessing worker processes
        print("Pipeline closed cleanly.")
        sys.exit(0)
