
# ODIN Project- Optical Device For Intellegent Navigation 

The **ODIN Project** is a cutting-edge, interactive system designed to integrate both video and audio processing for real-time communication with an AI model. It combines advanced webcam streaming, audio input, and websocket communication with an AI-powered backend to simulate an interactive dialogue. The project leverages Google’s GenAI API for AI-driven conversation, image processing, and audio playback.

## Key Features

- **Webcam Integration**: Streams live video feed and sends image data to a server for processing.
- **Voice Control**: Records and transmits audio in real-time, with real-time interaction through WebSocket.
- **WebSocket Communication**: Establishes a WebSocket connection with an AI backend for real-time text and audio responses.
- **Real-time Media Streaming**: Sends and receives audio (PCM) and image data (JPEG) for AI processing.
- **Audio Worklet**: Processes audio data efficiently using the AudioWorklet API in a browser.
- **Lighting Control**: Simulates tool invocation to control virtual lighting attributes using the `Mjölnir` tool.

## Requirements


- **Web Browser**: This system uses HTML5, JavaScript, and modern browser APIs.
- **Python Environment**: This project includes a Python backend for communication with the GenAI service.

## Installation

To get the project up and running on your local machine:

### Frontend Setup (Web Client)

1. Clone the repository:

   ```bash
   git clone https://github.com/fardeenKhadri/Odin.git
   cd Odin
   ```

2. Open the `index.html` file in a web browser to start the frontend application.

### Backend Setup (Python Server)


1. Run the Python server:

   ```bash
   python server.py
   ```

2. The WebSocket server will start on `ws://localhost:6106` and the web client will be able to connect to it for real-time media processing.

## How It Works

### Frontend (Web Client)
- The **HTML/JavaScript** frontend captures webcam video and audio data, then processes it by:
  - Capturing a frame from the webcam every 3 seconds and converting it to a base64-encoded JPEG.
  - Capturing audio input from the user's microphone, encoding it into PCM16 audio format, and sending it to the backend.
  - Establishing a WebSocket connection to a backend that processes both image and audio data, interacting with a GenAI model for AI-driven responses.

### Backend (Python Server)
- The Python backend uses the **Google GenAI API** to handle incoming WebSocket messages, process the audio and video data, and generate responses.
- The backend is capable of invoking a **tool** (`Mjölnir`) to adjust light parameters (luminosity and aura hue), simulating a fictional system that interacts with the AI.
- The server listens for incoming WebSocket connections and forwards received media chunks (audio/video) to the GenAI service. Responses from the AI are forwarded back to the frontend.

### Audio Processing
- The audio is processed in the frontend using **Web Audio API** and an **Audio Worklet** (`PCMProcessor`), allowing real-time manipulation of the audio stream.
- The audio chunks are sent over the WebSocket to the backend for processing, where responses are played back in real-time.

### WebSocket Communication
- The frontend and backend communicate via WebSocket (`ws://localhost:6106`). 
  - **Sending Media**: Audio (PCM) and image (JPEG) chunks are sent periodically.
  - **Receiving Responses**: The backend processes the received data and sends back AI-generated text and/or audio responses.

### Interaction with the AI
- The server interacts with Google’s **GenAI** API, where the AI listens for media inputs, processes them, and returns text or audio output.
- The AI can also trigger functions such as adjusting light settings through the virtual tool `Mjölnir`.

## How to Use

1. **Start the Server**: Run the backend server to begin handling WebSocket connections.
2. **Open the Web Client**: Open the `index.html` file in a browser to see the live webcam and microphone inputs.
3. **Interact**: Press the **Start Button** to begin recording and sending audio input, or press the **Stop Button** to end the recording session.

### Features to Explore:
- **Voice Interaction**: Record and send audio messages to the AI.
- **Real-time Video Feed**: View the webcam feed while interacting with the backend.
- **Lighting Adjustment**: Simulate tool calls to adjust virtual light properties.

## Contributing

Contributions to this project are welcome! If you have any suggestions or improvements, feel free to fork the repository and submit a pull request.


## Acknowledgements

- **Web Audio API**: For enabling real-time audio processing on the frontend.
- **WebSocket**: For real-time communication between the frontend and backend.


---

