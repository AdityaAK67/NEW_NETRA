import speech_recognition as sr

# Initialize recognizer
recognizer = sr.Recognizer()

# Use the microphone as the source
with sr.Microphone() as source:
    print("Adjusting for ambient noise... Please wait.")
    recognizer.adjust_for_ambient_noise(source)  # Adjusts for ambient noise
    print("Listening for command...")
    audio = recognizer.listen(source)  # Listen for speech

# Try recognizing the speech using Google Web Speech API
try:
    print("You said: " + recognizer.recognize_google(audio))
except sr.UnknownValueError:
    print("Sorry, I could not understand the audio.")
except sr.RequestError:
    print("Could not request results from Google Speech Recognition service.")
