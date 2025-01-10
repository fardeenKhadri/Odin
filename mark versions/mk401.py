import time
from PIL import Image
import moondream as md
from groq import Groq
import pyttsx3  # Text-to-Speech library
import speech_recognition as sr  # Speech-to-Text library

# Global variables to manage chatbot mode and history
chatbot_mode = "image_analysis"  # Default mode
conversation_history = []        # Stores previous inputs
last_caption = None              # Store the last image caption

# Initialize and load the visual model
try:
    model = md.vl(model="F://bmsit//trial//moondream-0_5b-int8.mf")
    print("Model loaded successfully!")
except Exception as e:
    print("Error loading model:", e)
    exit()

# Initialize Groq client
client = Groq(api_key="")

# Initialize TTS engine
tts_engine = pyttsx3.init()

# Initialize Speech Recognizer
recognizer = sr.Recognizer()

def voice_output(message):
    """Generates a voice output."""
    tts_engine.say(message)
    tts_engine.runAndWait()

def audio_input():
    """Captures audio input from the microphone and converts it to text."""
    with sr.Microphone() as source:
        print("Listening...")
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            text = recognizer.recognize_google(audio)
            print(f"You (Audio): {text}")
            return text
        except sr.UnknownValueError:
            print("Sorry, I could not understand the audio.")
            voice_output("Sorry, I could not understand the audio.")
            return None
        except sr.RequestError as e:
            print(f"Speech recognition error: {e}")
            return None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

def process_image(image_path):
    try:
        print(f"Attempting to open image: {image_path}")  # Debugging print
        image = Image.open(image_path).resize((224, 224))
        encoded_image = model.encode_image(image)
        caption = model.caption(encoded_image)["caption"]
        print(f"Caption: {caption}")
        return caption
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def image_analysis_response(user_input):
    global conversation_history

    # Clear history to retain only the last conversation
    conversation_history = [
        {
            "role": "system",
            "content": (
                "You are now a visual assistant with OCR capability for a visually impaired person. "
                "I will describe the environment to you and perform OCR to extract any visible text from the image. "
                "Your task is to determine if there are any obstacles in the person's path and read any visible text. "
                "If there are obstacles, identify them and suggest a safe direction to move (e.g., 'left,' 'right,' 'backward'). "
                "If it is safe to proceed forward, respond with 'No, it is safe to move forward.' If text is present in the image, read it out loud. "
                "Always think from the perspective of someone walking and using OCR for navigation assistance."
            )
        }
    ]
    # Add the user input to the conversation
    conversation_history.append({"role": "user", "content": user_input})

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=conversation_history,
            temperature=1,
            max_tokens=1024,
            top_p=1,
            stream=False,
            stop=None,
        )

        # Extract the assistant's response
        reply = completion.choices[0].message.content.strip()
        conversation_history.append({"role": "assistant", "content": reply})

        # Print the response
        print(f"Chatbot: {reply}")

        # Trigger voice warning only if obstacle is detected
        if "Yes, there are obstacles" in reply:
            print("Voice Warning: Stop!")
            voice_output("Stop! There are obstacles ahead.")
        
        return reply

    except Exception as e:
        print(f"Chatbot: Oops, something went wrong! {e}")
        return "I'm sorry, I encountered an error while processing your request."

def general_chatbot_response(user_input):
    global conversation_history

    # Retain the last conversation (keep last_caption in memory)
    if last_caption:
        conversation_history = [
            {"role": "system", "content": "You are a friendly and helpful chatbot."},
            {"role": "user", "content": f"I just saw an image that said: {last_caption}"}
        ]

    # Add the user input to the conversation
    conversation_history.append({"role": "user", "content": user_input})

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=conversation_history,
            temperature=0.7,
            max_tokens=1024,
            top_p=1,
            stream=False,
            stop=None,
        )

        # Extract the assistant's response
        reply = completion.choices[0].message.content.strip()
        conversation_history.append({"role": "assistant", "content": reply})

        # Print and speak the response (only in General Chatbot Mode)
        print(f"Chatbot: {reply}")
        voice_output(reply)

        return reply

    except Exception as e:
        print(f"Chatbot: Oops, something went wrong! {e}")
        return "I'm sorry, I encountered an error while processing your request."

def image_analysis_mode(image_path):
    global chatbot_mode, last_caption

    print("Switching to Image Analysis Mode.")

    while True:
        # Process the image and generate caption
        caption = process_image(image_path)
        if caption:
            last_caption = caption  # Save the last caption
            user_input = caption
            print(f"You: {user_input}")

            reply = image_analysis_response(user_input)
            print(f"Chatbot: {reply}\n")
        else:
            print("[Error] Unable to process the image. Exiting Image Analysis Mode.")
            return

        # Prompt the user for input to potentially switch modes
        print("\n[System] Type 'hold' to switch to General Chatbot Mode, 'speak' to provide audio input, or press Enter to continue in Image Analysis Mode.")
        user_command = input("Command: ").strip().lower()

        if user_command == "hold":
            print("Switching to General Chatbot Mode.\n")
            voice_output("Switching to General Chatbot Mode.")
            chatbot_mode = "general_chatbot"
            general_chatbot_mode()
        elif user_command == "speak":
            audio_text = audio_input()
            if audio_text:
                reply = image_analysis_response(audio_text)
                print(f"Chatbot (from Audio): {reply}")
        else:
            print("Continuing in Image Analysis Mode.\n")

def general_chatbot_mode():
    global chatbot_mode

    print("General Chatbot Mode Activated.")
    voice_output("General Chatbot Mode Activated.")

    print("Type 'SEE' to switch back to Image Analysis Mode or 'exit' to terminate the session.")
    while True:
        print("Say something or type your response.")
        user_input = audio_input() or input("You: ").strip()
        if user_input.lower() == 'exit':
            print("Chatbot: Goodbye! Have a great day!")
            voice_output("Goodbye! Have a great day!")
            exit()
        elif user_input.lower() == 'see':
            print("Switching back to Image Analysis Mode.\n")
            chatbot_mode = "image_analysis"
            image_analysis_mode("F://bmsit//trial//n.jpg")  # Update with your actual image path
            break
        else:
            reply = general_chatbot_response(user_input)
            print(f"Chatbot: {reply}\n")

def main():
    # Initial prompt with voice output
    welcome_message = "I, ODIN, call you, to see the world from my eye"
    print(f"Chatbot: {welcome_message}")
    voice_output(welcome_message)
    image_analysis_mode("F://bmsit//trial//n.jpg")  # Update with your actual image path

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nChatbot: Session terminated by user.")
        voice_output("Conversation ends here, Blessaður")
