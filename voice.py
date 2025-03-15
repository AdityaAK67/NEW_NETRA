import speech_recognition as sr
import pyttsx3
import subprocess
import time

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Function to speak a message
def speak(message):
    engine.say(message)
    engine.runAndWait()

# Function to perform OCR by running the external script
def ocr_function(script_path):
    try:
        # Run the OCR script
        result = subprocess.run(["python", script_path], capture_output=True, text=True)
        print(result.stdout)
        speak("OCR completed.")
    except Exception as e:
        print(f"Error: {e}")
        speak("An error occurred during OCR.")

# Function to perform object detection by running the external script
def object_detection(script_path):
    try:
        # Run the object detection script
        result = subprocess.run(["python", script_path], capture_output=True, text=True)
        print(result.stdout)
        speak("Object detection completed.")
    except Exception as e:
        print(f"Error: {e}")
        speak("An error occurred during object detection.")

# Function to listen for voice input
def listen_for_input():
    recognizer = sr.Recognizer()
    
    # Loop for multiple retries in case of errors
    while True:
        with sr.Microphone() as source:
            print("Listening for command...")
            speak("Say 'object' for object detection, 'text' for OCR, or 'exit' to exit.")
            recognizer.adjust_for_ambient_noise(source, duration=2)  # Adjust for ambient noise with a 2-second duration
            try:
                audio = recognizer.listen(source, timeout=5)  # Set a 5-second timeout for listening
                print("Audio captured:", audio)  # Debugging: Print audio content
                command = recognizer.recognize_google(audio)  # Use Google Speech Recognition
                print("You said:", command.lower())
                
                # Handle commands based on speech
                if 'text' in command.lower():  # If the command contains "text"
                    script_path = r"C:\Users\ADMIN\text-detection-python-easyocr\text.py"  # OCR file path
                    speak("text detection started")
                    ocr_function(script_path)
                elif 'object' in command.lower():  # If the command contains "object"
                    script_path = r"C:\Users\ADMIN\text-detection-python-easyocr\object.py"  # Object detection file path
                    speak("object detection started")
                    object_detection(script_path)
                elif 'exit' in command.lower():  # If the command contains "exit"
                    speak("Exiting the program.")
                    print("Exiting...")
                    break  # Exit the loop and terminate the program
                else:
                    speak("Invalid command. Please say 'object' for object detection or 'text' for OCR.")
                    print("Invalid command, retrying...")
                    time.sleep(1)  # Small delay before retrying
            except sr.UnknownValueError:
                speak("Sorry, I could not understand that. Please try again.")
                print("Error: Could not understand audio. Retrying...")
                time.sleep(1)  # Retry after a short delay
            except sr.RequestError:
                speak("Sorry, there was an error with the speech service.")
                print("Error: Request error, retrying...")
                time.sleep(1)  # Retry after a short delay
            except Exception as e:
                print(f"Unexpected error: {e}")
                speak("An unexpected error occurred. Retrying...")
                time.sleep(1)  # Retry after a short delay

if __name__ == "__main__":
    listen_for_input()
