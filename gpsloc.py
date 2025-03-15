import openrouteservice
import speech_recognition as sr
import pyttsx3
import requests
import time
from geopy.distance import geodesic

# Initialize the speech engine
engine = pyttsx3.init()

# OpenRouteService API Key (Replace with your valid key)
api_key = '5b3ce3597851110001cf6248c7a3c2ae7b9e4b1fa061ce3acc6b575b'

# Initialize OpenRouteService client
client = openrouteservice.Client(key=api_key)

# Function to convert text to speech
def speak(text):
    print(text)  # Print for debugging
    engine.say(text)
    engine.runAndWait()

# Function to listen to user's voice command
def listen_command():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        print("Listening for a command...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        print("Recognizing...")
        command = recognizer.recognize_google(audio)
        print(f"Command recognized: {command}")
        return command.lower()
    except sr.UnknownValueError:
        speak("Sorry, I couldn't understand. Please repeat.")
        return ""
    except sr.RequestError:
        speak("Could not request results; check your internet connection.")
        return ""

# Function to get live location using IP-based method
def get_current_location():
    try:
        response = requests.get("https://ipinfo.io/json")
        data = response.json()
        if "loc" in data:
            lat, lon = map(float, data["loc"].split(","))
            return [lon, lat]  # OpenRouteService requires [longitude, latitude]
    except Exception as e:
        print("Error getting location:", e)
    return None

# Function to geocode address (convert address to coordinates)
def geocode_address(address):
    result = client.pelias_search(address)
    if result['features']:
        return result['features'][0]['geometry']['coordinates']
    return None

# Function to get route and navigation steps
def get_route_with_directions(start_coords, end):
    end_coords = geocode_address(end)

    if start_coords and end_coords:
        route = client.directions(
            coordinates=[start_coords, end_coords],
            profile='driving-car',
            format='geojson'
        )

        total_distance = route['features'][0]['properties']['segments'][0]['distance'] / 1000  # Convert to km
        steps = route['features'][0]['properties']['segments'][0]['steps']

        return total_distance, steps
    else:
        speak("Could not find the address, please check the input.")
        return None, None

# Function to start live navigation
def start_navigation(total_distance, steps):
    speak(f"Total distance to your destination is {total_distance:.1f} kilometers.")

    for step in steps:
        instruction = step['instruction']
        step_distance = step['distance']  # Distance in meters

        while True:
            current_location = get_current_location()
            if not current_location:
                speak("Unable to retrieve current location. Please check your connection.")
                return

            # Check if user has reached the next instruction step
            step_location = (step['way_points'][0], step['way_points'][1])  # Step coordinates
            distance_to_step = geodesic(current_location[::-1], step_location[::-1]).meters

            if distance_to_step <= 50:  # Announce instruction when user is within 50 meters
                speak(f"In {step_distance:.0f} meters, {instruction}")
                break  # Move to the next step

            time.sleep(2)  # Check location every 2 seconds

    speak("You have reached your destination!")

# Main function to control the flow
def main():
    speak("Welcome! Ready to assist you with navigation.")

    while True:
        command = listen_command()
        
        if "navigate" in command:
            destination = command.replace("navigate to", "").strip()
            if destination:
                speak(f"Navigating to {destination}...")

                start_location = get_current_location()
                if start_location:
                    total_distance, steps = get_route_with_directions(start_location, destination)
                    if steps:
                        start_navigation(total_distance, steps)
                else:
                    speak("Unable to determine your current location.")
            else:
                speak("Please specify a destination.")
        elif "stop" in command:
            speak("Goodbye!")
            break
        else:
            speak("Please say 'navigate to' followed by your destination.")
import openrouteservice
import speech_recognition as sr
import pyttsx3
import time

# Initialize the speech engine
engine = pyttsx3.init()

# OpenRouteService API Key (Replace with your valid key)
api_key = '5b3ce3597851110001cf6248c7a3c2ae7b9e4b1fa061ce3acc6b575b'

# Initialize the OpenRouteService client
client = openrouteservice.Client(key=api_key)

# Function to convert text to speech
def speak(text):
    print(text)  # Print for debugging
    engine.say(text)
    engine.runAndWait()

# Function to listen to user's voice command
def listen_command():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        print("Listening for a command...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        print("Recognizing...")
        command = recognizer.recognize_google(audio)
        print(f"Command recognized: {command}")
        return command.lower()
    except sr.UnknownValueError:
        print("Sorry, I couldn't understand. Please repeat.")
        speak("Sorry, I couldn't understand. Please repeat.")
        return ""
    except sr.RequestError:
        print("Could not request results; check your internet connection.")
        speak("Could not request results; check your internet connection.")
        return ""

# Function to get current location (set manually to Kopargaon)
def get_current_location():
    return [74.4762, 19.8874]  # [longitude, latitude] for Kopargaon

# Function to geocode address (convert address to coordinates)
def geocode_address(address):
    result = client.pelias_search(address)
    if result['features']:
        coordinates = result['features'][0]['geometry']['coordinates']
        return coordinates
    return None

# Function to get turn-by-turn navigation from Kopargaon to destination
def get_route_with_directions(start_coords, end):
    end_coords = geocode_address(end)

    if start_coords and end_coords:
        route = client.directions(
            coordinates=[start_coords, end_coords],
            profile='driving-car',  # Change to 'cycling' or 'walking' if needed
            format='geojson'
        )
        
        # Extract route information
        route_length = route['features'][0]['properties']['segments'][0]['distance']
        route_duration = route['features'][0]['properties']['segments'][0]['duration']
        print(f"Total Distance: {route_length / 1000:.2f} km")
        print(f"Estimated Duration: {route_duration / 60:.2f} minutes")

        speak(f"Total distance: {route_length / 1000:.2f} kilometers. Estimated time: {route_duration / 60:.2f} minutes.")

        # Extract and read turn-by-turn navigation
        steps = route['features'][0]['properties']['segments'][0]['steps']
        for step in steps:
            instruction = step['instruction']
            distance = step['distance']  # Distance to next step in meters
            speak(f"In {distance:.0f} meters, {instruction}")
            time.sleep(2)  # Small delay to simulate real navigation
    else:
        speak("Could not find the address, please check the input.")

# Main function to control the flow
def main():
    speak("Welcome! Ready to assist you with navigation.")
    
    while True:
        # Listen for a voice command
        command = listen_command()
        
        if "navigate" in command:
            # Extract the destination from the command
            destination = command.replace("navigate to", "").strip()
            if destination:
                print(f"Navigating to {destination}...")
                speak(f"Navigating to {destination}...")

                start_location = get_current_location()  # Always Kopargaon
                get_route_with_directions(start_location, destination)
            else:
                speak("Please specify a destination.")
        elif "stop" in command:
            speak("Goodbye!")
            break
        else:
            speak("Sorry, I didn't get that. Please say 'navigate to' followed by your destination.")

if __name__ == "__main__":
    main()

