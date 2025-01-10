import time
from PIL import Image
import moondream as md
from groq import Groq

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
                "You are now a visual assistant for a visually impaired person. "
                "I'll describe the image to you, and you must decide whether the person can walk safely. "
                "If there are obstacles within walkable distance, respond with 'Yes, there are obstacles.' "
                "If it's safe, respond with 'No, it's safe to walk. Think from a person's point of view.'"
            ),
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
        return reply

    except Exception as e:
        print(f"Chatbot: Oops, something went wrong! {e}")
        return "I'm sorry, I encountered an error while processing your request."

def image_analysis_mode(image_path):
    global chatbot_mode, last_caption

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

        # Wait for 10 seconds before prompting
        for remaining in range(10, 0, -1):
            print(f"[System] Next prompt in {remaining} seconds...", end='\r')
            time.sleep(1)

        # Prompt the user for input to potentially switch modes
        print("\n[System] Type 'hold' to switch to General Chatbot Mode or press Enter to continue in Image Analysis Mode.")
        user_command = input("Command: ").strip().lower()

        if user_command == "hold":
            print("Switching to General Chatbot Mode.\n")
            chatbot_mode = "general_chatbot"
            general_chatbot_mode()
        else:
            print("Continuing in Image Analysis Mode.\n")

def general_chatbot_mode():
    global chatbot_mode

    print("General Chatbot Mode Activated. Type 'SEE' to switch back to Image Analysis Mode or 'exit' to terminate the session.")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == 'exit':
            print("Chatbot: Goodbye! Have a great day!")
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
    # Initial prompt
    print("Chatbot: Hello! Let me assist you with the image analysis. (type 'exit' to end the session)")
    image_analysis_mode("F://bmsit//trial//n.jpg")  # Update with your actual image path

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nChatbot: Session terminated by user.")
