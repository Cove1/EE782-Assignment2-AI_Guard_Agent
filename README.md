# AI Guard Agent: A Multi-Modal AI Surveillance System

![Python Version](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=for-the-badge)

A sophisticated, autonomous surveillance agent that fuses computer vision, speech recognition, and a large language model to monitor a physical space, identify individuals, and engage potential intruders with an escalating, dynamically-generated dialogue.

---

## 🎥 Live Demonstration

*(This is the most important part of your README. Create a high-quality GIF or short video showing the complete workflow and place it here. A great demo is worth a thousand lines of code.)*

**A great demo GIF would show:**
1.  The agent in `STANDBY` mode.
2.  The user activating it with "Start agent" and face confirmation.
3.  The agent greeting a known, "trusted" user who enters the frame.
4.  An "untrusted" user entering, triggering the `ENGAGING` state.
5.  The agent speaking its first warning, the "intruder" staying silent, and the agent escalating to Level 2.
6.  The final escalation to `LOCKDOWN` with the alarm sound playing.

![Demo GIF of AI Guard Agent in Action](https://path-to-your-spectacular-demo.gif)
*The AI Guard Agent activating, recognizing a trusted user, and engaging an intruder with its escalating dialogue protocol.*

---

## ✨ Core Features

The AI Guard Agent is more than just a script; it's a fully-integrated system with a robust feature set designed for intelligent, autonomous operation.

-   🧠 **Multi-Modal AI Integration:** Seamlessly orchestrates three distinct AI modalities:
    -   **Vision (OpenCV, `face_recognition`):** For real-time face detection and identification.
    -   **Speech (Google Speech-to-Text):** For command recognition and capturing intruder responses.
    -   **Language (Google Gemini LLM):** For generating natural, context-aware, and escalating conversational dialogue.

-   🔐 **Two-Factor Biometric Authentication:** Activates and deactivates only with a valid spoken command followed by a successful facial recognition match of a trusted user, preventing unauthorized control.

-   🗣️ **Dynamic, Escalating Dialogue Protocol:** Engages with unrecognized individuals using an LLM-powered, four-level threat protocol. The agent's tone and objective shift from inquisitive to stern to an active alert based on compliance and elapsed time.

-   🖥️ **Real-Time Graphical User Interface (PyQt6):** A clean and responsive GUI provides a live camera feed, a real-time system log, and clear status indicators (`STANDBY`, `GUARDING`, `ENGAGING`, `LOCKDOWN`), offering full operational transparency.

-   🚀 **Multi-Threaded Architecture:** Built for performance and responsiveness. The AI processing, GUI, voice listening, and TTS all run on separate threads, ensuring a non-blocking user experience even during intensive computation or network API calls.

-   🔔 **Automated Alerts & Lockdown Mode:** At the highest threat level, the agent automatically captures a snapshot of the intruder, sends an email alert to the owner, sounds an audible alarm, and enters an irreversible `LOCKDOWN` state.

---

## 🏛️ System Architecture

The system is designed using a layered architectural paradigm, separating concerns into Perception, Cognition, and Action layers. This modular design, managed by a central worker thread, ensures scalability and maintainability.

![System Architecture Diagram](path/to/your/Architecture.png)

The agent's behavior is governed by a Finite State Machine (FSM) to ensure predictable and robust transitions between its operational states.

![Finite State Machine Diagram](path/to/your/FSM.png)

---

## 🛠️ Technology Stack

This project leverages a powerful stack of modern AI and application development libraries.

| Category                | Technologies                                                                                             |
| ----------------------- | -------------------------------------------------------------------------------------------------------- |
| **Core AI & ML**        | `google-generativeai`, `face_recognition`, `SpeechRecognition`, `dlib`, `OpenCV`                         |
| **Application Framework** | `PyQt6`                                                                                                  |
| **Audio I/O**           | `pyttsx3`, `pygame` (for audio playback), `PyAudio`                                                      |
| **Concurrency**         | Python `threading`, `queue`, PyQt `QThread` & `pyqtSignal`                                               |
| **Utilities**           | `python-dotenv`, `NumPy`                                                                                 |

---

## 🚀 Getting Started

Follow these instructions to set up and run the AI Guard Agent on your local machine.

### 1. Prerequisites

-   Python 3.8+
-   A working webcam and microphone
-   `git` for cloning the repository
-   System dependencies for `dlib` (a requirement for `face_recognition`):
    -   **macOS:** `brew install cmake`
    -   **Ubuntu/Debian:** `sudo apt-get install build-essential cmake`
    -   **Windows:** Install [CMake](https://cmake.org/download/) and [Visual Studio with C++ tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).

### 2. Installation & Setup

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/your-username/ai-guard-agent.git
    cd ai-guard-agent
    ```

2.  **Create and Activate a Virtual Environment**
    ```bash
    # Create the virtual environment
    python -m venv venv
    # Activate it
    # On Windows (PowerShell):
    .\venv\Scripts\Activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    A `requirements.txt` is provided for easy installation.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables**
    Create a `.env` file in the root directory by copying the example:
    ```bash
    cp .env.example .env
    ```
    Now, open the `.env` file and add your credentials:
    ```dotenv
    # Get your free API key from Google AI Studio (https://makersuite.google.com/)
    GEMINI_API_KEY="YOUR_GEMINI_API_KEY_HERE"

    # --- Email Alert Configuration (Using Gmail as an example) ---
    # NOTE: You MUST use a Google "App Password", not your main password.
    ENABLE_EMAIL_ALERTS="True"
    SMTP_SERVER="smtp.gmail.com"
    SMTP_PORT="587"
    SMTP_USER="your.email@gmail.com"
    SMTP_PASS="your_16_digit_gmail_app_password"
    ALERT_RECIPIENT_EMAIL="email_to_receive_alerts@example.com"
    ```

5.  **Enroll Trusted Faces**
    -   Create a folder named `known_faces` in the root of the project.
    -   Add clear, well-lit `.jpg` images of trusted individuals.
    -   **Rename the files** to match the person's name, using underscores for spaces (e.g., `Swarup_Patil.jpg`). This name will be used for greetings.

### 3. Run the Application

With your virtual environment activated, launch the agent:
```bash
python main.py
