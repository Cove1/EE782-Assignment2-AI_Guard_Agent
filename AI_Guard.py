# Standard library and third-party imports
import cv2
import face_recognition
import speech_recognition as sr
import pyttsx3
import os
import time
import threading
import re
import queue
import platform
import logging
import signal
import sys
import subprocess
import tempfile
import pygame
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import google.generativeai as genai
from dotenv import load_dotenv
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import traceback

# PyQt6 imports for GUI
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QTextEdit, QListWidget, QListWidgetItem, QMessageBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt6.QtGui import QImage, QPixmap, QColor

# --- CONFIGURATION AND CORE CLASSES ---

# Load environment variables from .env file
load_dotenv()

@dataclass
class Config:
    # Core settings
    camera_width: int = 640
    camera_height: int = 480
    process_every_n_frames: int = 5
    face_detection_scale: float = 0.25
    face_tolerance: float = 0.5
    known_faces_dir: str = "known_faces"
    intruder_logs_dir: str = "intruder_logs"

    # Voice command settings
    activation_phrase: str = "start agent"
    deactivation_phrase: str = "stop agent"
    command_confirmation_timeout: int = 5

    # TTS settings
    tts_rate: int = 180
    tts_volume: float = 0.9
    
    # Intruder protocol settings
    intruder_engagement_delay: int = 5       # Time before engaging a new intruder
    engagement_escalation_time: int = 15     # Time of non-compliance to escalate threat
    intruder_reengagement_delay: int = 20    # Cooldown before re-engaging a cleared intruder
    conversation_timeout: int = 10           # Time to wait for intruder's verbal response
    reset_delay: int = 5                     # Time after intruder is gone to reset state
    greeting_cooldown: int = 180             # Cooldown for greeting known users

    # Threat keywords
    threat_keywords: List[str] = ("terrorist", "attack", "gun", "weapon", "bomb", "rob", "hostage", "kill")

    # LLM settings
    gemini_api_key: str = os.getenv('GEMINI_API_KEY', '')
    model_name: str = "gemini-2.5-flash"

    # Notification settings
    enable_email_alerts: bool = os.getenv('ENABLE_EMAIL_ALERTS', 'False').lower() == 'true'
    smtp_server: str = os.getenv('SMTP_SERVER', '')
    smtp_port: int = int(os.getenv('SMTP_PORT', 587))
    smtp_user: str = os.getenv('SMTP_USER', '')
    smtp_pass: str = os.getenv('SMTP_PASS', '')
    alert_recipient_email: str = os.getenv('ALERT_RECIPIENT_EMAIL', '')
    
    # Alarm settings
    alarm_sound_enabled: bool = True
    alarm_sound_file: str = "alarm.wav"
    alarm_duration_seconds: int = 10

# --- HELPER CLASSES (TTS, FaceRecognizer, Voice, AI) ---
# NOTE: These classes are largely unchanged from your original robust implementation.
# They are collapsed here for brevity but are included in the full script.

class MultiTTSEngine:
    """A robust, multi-backend Text-to-Speech engine."""
    def __init__(self, config: Config):
        self.config = config; self.current_engine = None; self.available_engines = []; self.speech_queue = queue.Queue()
        self.shutdown_flag = threading.Event(); self.tts_thread = None; self.pygame_available = False
        try: pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512); self.pygame_available = True
        except Exception as e: logging.warning(f"Pygame mixer init failed: {e}. Pygame TTS disabled.")
    def clear_queue(self):
        with self.speech_queue.mutex: self.speech_queue.queue.clear()
        logging.info("TTS queue cleared.")
    def _test_pyttsx3(self) -> bool:
        try:
            engine = pyttsx3.init()
            if not engine or not engine.getProperty('voices'): return False
            engine.say("TTS test"); engine.runAndWait(); engine.stop(); logging.info("✅ pyttsx3 engine test passed"); return True
        except Exception: return False
    def initialize_engines(self) -> bool:
        logging.info("🔍 Detecting available TTS engines..."); 
        if self._test_pyttsx3(): self.available_engines.append("pyttsx3")
        if not self.available_engines: logging.error("❌ No working TTS engines found!"); return False
        self.current_engine = self.available_engines[0]; logging.info(f"✅ Using TTS engine: {self.current_engine}"); return True
    def start(self) -> bool:
        if not self.initialize_engines(): return False
        self.shutdown_flag.clear(); self.tts_thread = threading.Thread(target=self._tts_worker, daemon=True); self.tts_thread.start()
        self.speak("TTS system ready"); logging.info("🗣️ TTS engine started successfully"); return True
    def stop(self): self.shutdown_flag.set(); self.clear_queue()
    def speak(self, text: str):
        if not self.shutdown_flag.is_set() and text.strip(): self.speech_queue.put(text)
    def _tts_worker(self):
        while not self.shutdown_flag.is_set():
            try:
                text = self.speech_queue.get(timeout=1.0); self._speak_pyttsx3(text); self.speech_queue.task_done()
            except queue.Empty: continue
            except Exception as e: logging.error(f"TTS worker error: {e}")
    def _speak_pyttsx3(self, text: str):
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', self.config.tts_rate); engine.setProperty('volume', self.config.tts_volume)
            engine.say(text); engine.runAndWait(); engine.stop()
        except Exception as e: logging.error(f"pyttsx3 speak error: {e}")

class OptimizedFaceRecognizer:
    """Handles loading known faces and recognizing them efficiently."""
    def __init__(self, config: Config): self.config = config; self.known_encodings = []; self.known_names = []; self.last_greetings = {}
    def load_known_faces(self):
        faces_dir = Path(self.config.known_faces_dir); faces_dir.mkdir(parents=True, exist_ok=True); logging.info("Loading known faces...")
        for img_path in faces_dir.glob("*.jpg"):
            try:
                image = face_recognition.load_image_file(str(img_path)); encodings = face_recognition.face_encodings(image)
                if encodings: self.known_encodings.append(encodings[0]); name = img_path.stem.replace('_', ' ').title(); self.known_names.append(name); logging.info(f"✅ Loaded face: {name}")
            except Exception as e: logging.error(f"Failed to load {img_path}: {e}")
    def recognize_faces(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=self.config.face_detection_scale, fy=self.config.face_detection_scale); rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame, model="hog"); face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations); face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(self.known_encodings, face_encoding, tolerance=self.config.face_tolerance); name = "Unknown"
            if True in matches: first_match_index = matches.index(True); name = self.known_names[first_match_index]
            face_names.append(name)
        scale = 1.0 / self.config.face_detection_scale; face_locations_scaled = [(int(t * scale), int(r * scale), int(b * scale), int(l * scale)) for t, r, b, l in face_locations]; return face_locations_scaled, face_names
    def should_greet(self, name: str) -> bool:
        if name == "Unknown": return False
        now = time.time()
        if now - self.last_greetings.get(name, 0) > self.config.greeting_cooldown: self.last_greetings[name] = now; return True
        return False

class VoiceListener:
    """Listens for voice commands in a background thread."""
    def __init__(self, config: Config, text_queue: queue.Queue):
        self.config = config; self.text_queue = text_queue; self.recognizer = sr.Recognizer(); self.microphone = None
        self._stop_event = threading.Event(); self._thread = None; self.mode = 'keywords'; self.callbacks = {}
        self.recognizer.dynamic_energy_threshold = True; self.recognizer.pause_threshold = 1.2
    def start(self):
        try:
            self.microphone = sr.Microphone(); 
            with self.microphone as source: self.recognizer.adjust_for_ambient_noise(source, duration=1); logging.info("🎤 Microphone calibrated and ready")
            self._stop_event.clear(); self._thread = threading.Thread(target=self._listen_loop, daemon=True); self._thread.start()
        except Exception as e: logging.error(f"Microphone setup failed: {e}"); return False
        return True
    def stop(self): self._stop_event.set(); logging.info("👂 Voice listening stopped")
    def set_mode(self, mode: str): self.mode = mode
    def register_command(self, phrase: str, callback): self.callbacks[phrase.lower()] = callback
    # In the VoiceListener class, REPLACE this method

    def _listen_loop(self):
        logging.info("👂 Voice listener thread started.")
        # <<< MODIFICATION: Define command keywords outside the loop >>>
        activation_keywords = self.config.activation_phrase.lower().split()
        deactivation_keywords = self.config.deactivation_phrase.lower().split()

        while not self._stop_event.is_set():
            try:
                with self.microphone as source:
                    audio = self.recognizer.listen(source, timeout=8, phrase_time_limit=10)
                text = self.recognizer.recognize_google(audio).lower()
                logging.info(f"Voice detected ({self.mode} mode): '{text}'")

                if self.mode == 'keywords':
                    # --- IMPROVED LOGIC START ---
                    # Check if ALL keywords for a command are present in the text
                    if all(keyword in text for keyword in deactivation_keywords):
                        if self.callbacks.get(self.config.deactivation_phrase):
                            self.callbacks[self.config.deactivation_phrase]()
                            continue # Prioritize deactivation

                    if all(keyword in text for keyword in activation_keywords):
                        if self.callbacks.get(self.config.activation_phrase):
                            self.callbacks[self.config.activation_phrase]()
                    # --- IMPROVED LOGIC END ---

                elif self.mode == 'conversation':
                    self.text_queue.put(text)
            
            except sr.RequestError as e: logging.warning(f"Network error: {e}"); time.sleep(5)
            except (sr.UnknownValueError, sr.WaitTimeoutError):
                if self.mode == 'conversation': self.text_queue.put(None)
            except Exception as e: logging.error(f"Voice recognition error: {e}")
        

class ConversationalAI:
    """Manages interactions with the Google Gemini LLM."""
    def __init__(self, config: Config):
        self.config = config; self.model = None; self.chat_session = None
        if config.gemini_api_key:
            try: genai.configure(api_key=config.gemini_api_key); self.model = genai.GenerativeModel(config.model_name); logging.info("✅ Gemini LLM model ready")
            except Exception as e: logging.error(f"Gemini LLM init failed: {e}")
    def start_conversation(self, threat_level: int) -> str:
        if not self.model: return "Unidentified person. Please leave this area immediately."
        system_prompts = {
            1: ("You are a security AI. Your tone is inquisitive but firm. Ask the person to state their purpose for being here."),
            2: ("You are a security AI. Your tone is now stern. The person has not complied. Firmly state that this is a restricted area and demand they leave immediately."),
        }
        prompt = system_prompts.get(threat_level, system_prompts[1])
        try:
            self.chat_session = self.model.start_chat(history=[])
            response = self.chat_session.send_message(f"System instruction: {prompt}. Start the conversation now."); return response.text.strip()
        except Exception as e: logging.error(f"AI initial response failed: {e}"); return "This is a restricted area. Identify yourself."
    def continue_conversation(self, user_input: str) -> str:
        if not self.chat_session: return "Error: No active conversation."
        try: response = self.chat_session.send_message(user_input); return response.text.strip()
        except Exception as e: logging.error(f"AI response failed: {e}"); return "I do not understand. Leave now."
    def end_conversation(self): self.chat_session = None

class NotificationManager:
    """Handles sending email notifications in a background thread."""
    def __init__(self, config: Config):
        self.config = config

    def send_email_alert(self, subject: str, body: str, image_path: Optional[str] = None):
        if not self.config.enable_email_alerts:
            logging.warning("Email alerts are disabled in the configuration.")
            return
        
        # Run email sending in a separate thread to avoid blocking
        email_thread = threading.Thread(
            target=self._send_email_worker,
            args=(subject, body, image_path),
            daemon=True
        )
        email_thread.start()

    def _send_email_worker(self, subject: str, body: str, image_path: Optional[str]):
        logging.info(f"Preparing to send email alert to {self.config.alert_recipient_email}")
        msg = MIMEMultipart()
        msg['From'] = self.config.smtp_user
        msg['To'] = self.config.alert_recipient_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        if image_path and Path(image_path).exists():
            try:
                with open(image_path, 'rb') as f:
                    img_data = f.read()
                image = MIMEImage(img_data, name=Path(image_path).name)
                msg.attach(image)
                logging.info(f"Attached intruder snapshot: {image_path}")
            except Exception as e:
                logging.error(f"Failed to attach image {image_path}: {e}")
        
        try:
            with smtplib.SMTP(self.config.smtp_server, self.config.smtp_port) as server:
                server.starttls()
                server.login(self.config.smtp_user, self.config.smtp_pass)
                server.send_message(msg)
            logging.info("✅ Email alert sent successfully.")
        except Exception as e:
            logging.error(f"❌ Failed to send email alert: {e}")

class QtLogHandler(logging.Handler, QObject):
    log_signal = pyqtSignal(str, str)
    def __init__(self, parent=None):
        super().__init__(); QObject.__init__(self, parent); self.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', '%H:%M:%S'))
    def emit(self, record):
        msg = self.format(record); self.log_signal.emit(record.levelname, msg)


# Add this new class directly below the MultiTTSEngine class definition

class SynchronizedTTS(MultiTTSEngine):
    """
    An extension of the TTS engine that allows for blocking, synchronous speech.
    This is crucial for ensuring critical announcements are fully spoken before
    the next action (like an alarm) is triggered.
    """
    def speak_and_wait(self, text: str, timeout: float = 10.0):
        """
        Speaks a given text and blocks until the speech is finished or a timeout is reached.
        """
        if not text.strip():
            return

        # Use a thread-safe event to signal completion
        completion_event = threading.Event()
        
        # We wrap the original speak method's text with a tuple containing
        # the text and the event to signal.
        # This allows the worker to know this is a special synchronous request.
        self.speech_queue.put((text, completion_event))

        # Wait for the event to be set by the TTS worker thread
        logging.info(f"Waiting for TTS to finish speaking: '{text}'")
        event_was_set = completion_event.wait(timeout=timeout)

        if not event_was_set:
            logging.warning(f"TTS speak_and_wait timed out after {timeout} seconds for text: '{text}'")
        else:
            logging.info("TTS speak_and_wait completed.")

    def _tts_worker(self):
        """
        Overrides the base worker to handle both synchronous and asynchronous requests.
        """
        while not self.shutdown_flag.is_set():
            try:
                item = self.speech_queue.get(timeout=1.0)
                
                text_to_speak = ""
                completion_event = None

                if isinstance(item, tuple):
                    # This is a synchronous request
                    text_to_speak, completion_event = item
                else:
                    # This is a standard asynchronous request
                    text_to_speak = item
                
                if text_to_speak:
                    self._speak_pyttsx3(text_to_speak)

                if completion_event:
                    # Signal that we are done speaking
                    completion_event.set()

                self.speech_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"TTS worker error: {e}")

# --- MAIN AI WORKER THREAD ---

class AIGuardWorker(QObject):
    frame_ready = pyqtSignal(np.ndarray); status_changed = pyqtSignal(str)
    threat_level_changed = pyqtSignal(int); faces_detected = pyqtSignal(list)
    system_shutdown = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.config = Config()
        self.tts = SynchronizedTTS(self.config) 
        self.face_recognizer = OptimizedFaceRecognizer(self.config)
        self.ai = ConversationalAI(self.config)
        self.notification_manager = NotificationManager(self.config)
        self.recognized_text_queue = queue.Queue()
        self.voice_listener = VoiceListener(self.config, self.recognized_text_queue)
        
        self._shutdown_flag = threading.Event()
        self.last_valid_frame: Optional[np.ndarray] = None
        
        # <<< MODIFICATION START: Load alarm sound on initialization >>>
        self.alarm_sound: Optional[pygame.mixer.Sound] = None
        if self.config.alarm_sound_enabled:
            try:
                # The TTS engine already initializes pygame.mixer, so we can use it.
                alarm_path = Path(self.config.alarm_sound_file)
                if alarm_path.exists():
                    self.alarm_sound = pygame.mixer.Sound(str(alarm_path))
                    logging.info(f"✅ Alarm sound '{self.config.alarm_sound_file}' loaded successfully.")
                else:
                    logging.warning(f"⚠️ Alarm sound file not found at '{alarm_path}'. Alarm will be silent.")
            except Exception as e:
                logging.error(f"❌ Failed to load alarm sound: {e}")
        # <<< MODIFICATION END >>>

        # State variables
        self.guard_active = False; self.lockdown_mode = False
        self.intruder_detected = False; self.intruder_start_time = None
        self.intruder_last_seen = None; self.threat_level = 0
        self.last_escalation_time = 0; self.pending_activation = False
        self.pending_deactivation = False; self.command_request_time = 0
        self.engaging_intruder = False; self.conversation_thread = None
        self.last_engagement_end_time = 0
    
    def initialize(self) -> bool:
        logging.info("🚀 Initializing AI Guard System...")
        try:
            Path(self.config.intruder_logs_dir).mkdir(exist_ok=True)
            if not self.tts.start(): return False
            self.face_recognizer.load_known_faces()
            self.voice_listener.register_command(self.config.activation_phrase, self.handle_activation_phrase)
            self.voice_listener.register_command(self.config.deactivation_phrase, self.handle_deactivation_phrase)
            if not self.voice_listener.start(): return False
            self.tts.speak("AI Guardian system initialized and ready.")
            logging.info("✅ System ready!")
            return True
        except Exception as e:
            logging.critical(f"CRITICAL INITIALIZATION FAILURE: {e}")
            return False

    def activate_guard(self):
        if self.guard_active: return
        self.guard_active = True; self.reset_intruder_state(); self.tts.speak("Guardian mode activated.")
        logging.info("🛡️ GUARD ACTIVATED"); self.status_changed.emit("GUARDING")
    
    def deactivate_guard(self):
        if self.lockdown_mode:
            self.tts.speak("System lockdown is active. Deactivation is not possible."); logging.warning("Deactivation denied due to lockdown.")
            return
        if not self.guard_active: return
        self.guard_active = False; self.reset_intruder_state(); self.tts.speak("Standing down.")
        logging.info("🛡️ GUARD DEACTIVATED"); self.status_changed.emit("STANDBY")
    
    def handle_activation_phrase(self):
        if not self.guard_active:
            logging.info("Activation phrase heard. Awaiting two-factor face confirmation...")
            self.tts.speak("Please look at the camera to confirm activation.")
            self.pending_activation = True; self.command_request_time = time.time()
    
    def handle_deactivation_phrase(self):
        if self.guard_active:
            logging.info("Deactivation phrase heard. Awaiting two-factor face confirmation...")
            self.tts.speak("Please look at the camera to confirm deactivation.")
            self.pending_deactivation = True; self.command_request_time = time.time()
    
    def process_pending_commands(self, known_face_present: bool, current_time: float):
        is_pending = self.pending_activation or self.pending_deactivation
        if not is_pending: return
        if current_time - self.command_request_time > self.config.command_confirmation_timeout:
            logging.warning("Command confirmation timed out."); self.pending_activation = False; self.pending_deactivation = False
            return
        if known_face_present:
            if self.pending_activation: self.activate_guard()
            elif self.pending_deactivation: self.deactivate_guard()
            self.pending_activation = False; self.pending_deactivation = False
            
    # In the AIGuardWorker class, REPLACE the reset_intruder_state method

    def reset_intruder_state(self):
        logging.info("Resetting intruder state...")
        # This flag is critical. Setting it to False will cause the conversation loop to exit gracefully.
        self.engaging_intruder = False 
        
        self.tts.clear_queue()
        self.intruder_detected = False
        self.intruder_start_time = None
        self.intruder_last_seen = None
        if self.threat_level > 0:
            self.threat_level = 0
            self.threat_level_changed.emit(self.threat_level)
        self.last_escalation_time = 0
        
        if not self.lockdown_mode:
            self.status_changed.emit("GUARDING" if self.guard_active else "STANDBY")

    # In AIGuardWorker class, replace the escalate_threat method

    def escalate_threat(self, new_level: int, frame: np.ndarray):
        if new_level <= self.threat_level: return
        self.threat_level = new_level
        self.threat_level_changed.emit(self.threat_level)
        logging.critical(f"🚨🚨 THREAT ESCALATED TO LEVEL {self.threat_level} 🚨🚨")

        # Use speak() for non-blocking announcements
        if self.threat_level == 1:
            # This is handled by the conversation loop, no speech needed here
            pass 
        
        elif self.threat_level == 2:
            self.tts.speak("Warning. You have failed to comply. Your photograph is being recorded as evidence.")
            self.capture_intruder_snapshot(frame)
        
        # <<< MODIFICATION START: Use speak_and_wait for critical, sequential actions >>>
        elif self.threat_level == 3:
            # Speak the warning and WAIT for it to finish before sounding the alarm.
            self.tts.speak_and_wait("Final warning. An audible alarm is being activated.")
            if self.config.alarm_sound_enabled:
                self.trigger_alarm()
        
        elif self.threat_level == 4:
            self.tts.clear_queue()
            # Speak the final warning and WAIT before sending notifications and locking down.
            self.tts.speak_and_wait("Maximum threat level reached. Authorities are being notified. System is entering lockdown.")
            snapshot_path = self.capture_intruder_snapshot(frame)
            self.notification_manager.send_email_alert(
                subject="AI GUARDIAN - INTRUDER ALERT (MAXIMUM THREAT)",
                body="An intruder has been detected and has failed to comply with multiple warnings. System is now in lockdown.",
                image_path=snapshot_path
            )
            self.lockdown_mode = True
            self.status_changed.emit("LOCKDOWN")
        # <<< MODIFICATION END >>>

    def capture_intruder_snapshot(self, frame: np.ndarray) -> Optional[str]:
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"intruder_{timestamp}.jpg"
            filepath = Path(self.config.intruder_logs_dir) / filename
            cv2.imwrite(str(filepath), frame)
            logging.info(f"📸 Intruder snapshot saved to {filepath}")
            return str(filepath)
        except Exception as e:
            logging.error(f"Failed to capture intruder snapshot: {e}")
            return None
            
    # <<< MODIFICATION START: Replace the entire trigger_alarm method >>>
    def trigger_alarm(self):
        """Triggers the audible alarm using the pre-loaded WAV file."""
        if not self.alarm_sound:
            logging.warning("Alarm triggered, but no sound file is loaded. Alarm will be silent.")
            # Fallback to terminal bell if you want
            # print('\a', end='', flush=True)
            return

        def alarm_worker():
            logging.info(f"🔊 ALARM ACTIVATED (playing {self.config.alarm_sound_file})")
            
            # Play the sound in an infinite loop
            self.alarm_sound.play(loops=-1)
            
            # Wait for the specified duration, checking for shutdown signal
            start_time = time.time()
            while time.time() - start_time < self.config.alarm_duration_seconds:
                if self._shutdown_flag.is_set():
                    break
                time.sleep(0.1)

            # Stop the sound
            self.alarm_sound.stop()
            logging.info("🔊 Alarm deactivated")

        # Run the alarm in a non-blocking thread
        threading.Thread(target=alarm_worker, daemon=True).start()
    # <<< MODIFICATION END >>>

    # In the AIGuardWorker class, REPLACE the entire handle_intruder_logic method

    def handle_intruder_logic(self, current_time: float, faces: List[str], frame: np.ndarray):
        if not self.guard_active or self.lockdown_mode: return
        unknown_present = "Unknown" in faces

        if unknown_present:
            self.intruder_last_seen = current_time
            if not self.intruder_detected:
                # A new intruder has just appeared
                self.intruder_detected = True
                self.intruder_start_time = current_time
                logging.warning("⚠️ Unknown person detected. Starting engagement timer.")

            # Check if we should START a new engagement
            elif not self.engaging_intruder and \
                 (current_time - self.intruder_start_time > self.config.intruder_engagement_delay) and \
                 (current_time - self.last_engagement_end_time > self.config.intruder_reengagement_delay):
                
                # Set the flag to true and start the conversation thread.
                # The conversation thread will now handle all further escalations.
                self.engaging_intruder = True
                self.status_changed.emit("ENGAGING")
                self.conversation_thread = threading.Thread(target=self.intruder_conversation_loop, daemon=True)
                self.conversation_thread.start()

        elif self.intruder_detected:
            # Intruder was present, but is now gone. Check if we should reset.
            if self.intruder_last_seen and (current_time - self.intruder_last_seen > self.config.reset_delay):
                logging.info("✅ Intruder appears to have left the area. Resetting state.")
                self.reset_intruder_state()

    # In the AIGuardWorker class, REPLACE the entire intruder_conversation_loop method

    def intruder_conversation_loop(self):
        logging.info("Starting intruder engagement protocol...")
        self.voice_listener.set_mode('conversation')
        
        # Escalate to level 1 to start the conversation
        self.escalate_threat(1, self.last_valid_frame)
        initial_response = self.ai.start_conversation(self.threat_level)
        self.tts.speak(initial_response)
        
        # This will track time within the conversation for escalations
        self.last_escalation_time = time.time()

        # This loop now runs continuously until the main loop sets engaging_intruder to False
        while self.engaging_intruder:
            try:
                # Wait for the intruder to speak
                intruder_speech = self.recognized_text_queue.get(timeout=self.config.conversation_timeout)
                
                if intruder_speech:
                    # Intruder responded
                    self.last_escalation_time = time.time() # Reset escalation timer
                    if any(keyword in intruder_speech for keyword in self.config.threat_keywords):
                        self.escalate_threat(4, self.last_valid_frame)
                        continue # Loop will terminate as engaging_intruder becomes false
                    
                    ai_response = self.ai.continue_conversation(intruder_speech)
                    self.tts.speak(ai_response)
                else:
                    # This case handles if the voice recognizer returns None (e.g., noise)
                    # We treat it like a timeout.
                    pass 

            except queue.Empty:
                # This block catches the timeout from queue.get()
                logging.warning("Conversation timed out due to intruder silence.")
                
                current_time = time.time()
                if current_time - self.last_escalation_time > self.config.engagement_escalation_time:
                    # Enough time has passed without compliance, escalate
                    self.last_escalation_time = current_time
                    self.tts.speak("No response detected. This is being treated as non-compliance.")
                    self.escalate_threat(self.threat_level + 1, self.last_valid_frame)
                else:
                    # Not enough time has passed to escalate, just prompt again
                    self.tts.speak("I am waiting for your response.")
        
        # Cleanup after the loop finishes
        logging.info("Intruder engagement protocol has concluded.")
        self.ai.end_conversation()
        self.last_engagement_end_time = time.time()
        self.voice_listener.set_mode('keywords')
        if not self.lockdown_mode and self.guard_active:
            self.status_changed.emit("GUARDING")

    def run_surveillance(self):
        if not self.initialize(): self.system_shutdown.emit(); return
        cap = cv2.VideoCapture(1)
        if not cap.isOpened(): logging.error("❌ Cannot open camera"); self.system_shutdown.emit(); return
        
        logging.info("📷 Camera started"); frame_count = 0; current_face_names = []
        
        while not self._shutdown_flag.is_set():
            try:
                ret, frame = cap.read()
                if not ret: logging.warning("Camera frame could not be read."); time.sleep(0.5); continue
                self.last_valid_frame = frame.copy()
                current_time = time.time()
                
                # Process faces intermittently for performance
                if frame_count % self.config.process_every_n_frames == 0:
                    face_locations, face_names = self.face_recognizer.recognize_faces(frame)
                    current_face_names = face_names
                    self.faces_detected.emit(face_names)
                    # Draw overlay on the original frame
                    for (top, right, bottom, left), name in zip(face_locations, face_names):
                        color = (0, 0, 255) if name == "Unknown" else (0, 255, 0)
                        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

                known_face_present = any(name != "Unknown" for name in current_face_names)
                self.process_pending_commands(known_face_present, current_time)
                
                if self.guard_active:
                    for name in current_face_names:
                        if self.face_recognizer.should_greet(name): self.tts.speak(f"Hello {name}.")
                
                self.handle_intruder_logic(current_time, current_face_names, self.last_valid_frame)
                
                self.frame_ready.emit(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frame_count += 1
                time.sleep(0.02) # Small sleep to yield CPU
            except Exception as e:
                logging.error(f"FATAL ERROR in surveillance loop: {e}")
                logging.error(traceback.format_exc())
                time.sleep(5) # Prevent rapid-fire crashes
                
        cap.release(); self.shutdown()

    def shutdown(self):
        if not self._shutdown_flag.is_set():
            logging.info("🔄 Shutting down worker..."); self._shutdown_flag.set()
            self.voice_listener.stop(); self.tts.stop()
            self.system_shutdown.emit(); logging.info("✅ Worker shutdown complete")


# --- PYQT6 GUI APPLICATION ---

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__(); self.setWindowTitle("AI Guardian Professional"); self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet(self.get_stylesheet())
        self.status_colors = {
            "STANDBY": ("STANDBY", "#FFA726"), "GUARDING": ("GUARDING", "#66BB6A"),
            "ENGAGING": ("ENGAGING", "#29B6F6"), "LOCKDOWN": ("LOCKDOWN", "#F44336")
        }
        self.setup_ui(); self.setup_worker_thread()

    def setup_ui(self):
        central_widget = QWidget(); self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        # Left Panel (Video + Logs)
        left_panel = QWidget(); video_layout = QVBoxLayout(left_panel)
        self.video_label = QLabel("Initializing Camera..."); self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 480); self.video_label.setStyleSheet("background-color: #000;")
        video_layout.addWidget(self.video_label)
        video_layout.addWidget(self.create_header("LIVE EVENT LOG"))
        self.log_box = QTextEdit(); self.log_box.setReadOnly(True)
        video_layout.addWidget(self.log_box)
        main_layout.addWidget(left_panel, 2)
        # Right Panel (Controls)
        right_panel = QWidget(); control_panel_layout = QVBoxLayout(right_panel); control_panel_layout.setSpacing(15)
        control_panel_layout.addWidget(self.create_header("SYSTEM STATUS"))
        self.status_label = self.create_status_label("STANDBY", self.status_colors["STANDBY"][1])
        control_panel_layout.addWidget(self.status_label)
        self.threat_label = self.create_status_label("THREAT LEVEL: 0", "#4CAF50")
        control_panel_layout.addWidget(self.threat_label)
        control_panel_layout.addWidget(self.create_header("DETECTED FACES"))
        self.faces_list = QListWidget()
        control_panel_layout.addWidget(self.faces_list)
        control_panel_layout.addStretch()
        self.quit_button = QPushButton("🔌 Quit Application"); self.quit_button.clicked.connect(self.close)
        control_panel_layout.addWidget(self.quit_button)
        main_layout.addWidget(right_panel, 1)

    def create_header(self, text): label = QLabel(text); label.setObjectName("Header"); return label
    def create_status_label(self, text, color): label = QLabel(text); label.setAlignment(Qt.AlignmentFlag.AlignCenter); label.setStyleSheet(f"background-color: {color}; color: #fff; font-weight: bold; padding: 10px; border-radius: 5px;"); return label
    def get_stylesheet(self): return "QMainWindow, QWidget { background-color: #2E2E2E; color: #F0F0F0; font-family: 'Segoe UI', Arial; } QLabel#Header { font-size: 16px; font-weight: bold; color: #03A9F4; border-bottom: 2px solid #03A9F4; padding-bottom: 5px; margin-top: 10px; } QPushButton { background-color: #F44336; color: white; border: none; padding: 12px; border-radius: 5px; font-size: 14px; font-weight: bold; } QPushButton:hover { background-color: #E53935; } QTextEdit, QListWidget { background-color: #212121; border: 1px solid #424242; border-radius: 5px; font-family: 'Consolas', monospace; } QListWidget::item { padding: 5px; }"
    
    def setup_worker_thread(self):
        self.worker_thread = QThread(); self.worker = AIGuardWorker()
        self.worker.moveToThread(self.worker_thread)
        self.worker.frame_ready.connect(self.update_video_frame)
        self.worker.status_changed.connect(self.update_status)
        self.worker.threat_level_changed.connect(self.update_threat_level)
        self.worker.faces_detected.connect(self.update_faces_list)
        self.worker.system_shutdown.connect(self.on_worker_shutdown)
        self.worker_thread.started.connect(self.worker.run_surveillance)
        self.worker_thread.finished.connect(self.worker.deleteLater)
        self.worker_thread.start()
        
    def update_video_frame(self, frame_rgb): h, w, ch = frame_rgb.shape; bytes_per_line = ch * w; qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888); pixmap = QPixmap.fromImage(qt_image); self.video_label.setPixmap(pixmap.scaled(self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
    def update_status(self, status: str): text, color = self.status_colors.get(status, ("UNKNOWN", "#757575")); self.status_label.setText(text); self.status_label.setStyleSheet(self.status_label.styleSheet().split(';')[0] + f"; background-color: {color};")
    def update_threat_level(self, level): colors = {0: "#4CAF50", 1: "#FFC107", 2: "#FF9800", 3: "#F44336", 4: "#B71C1C"}; self.threat_label.setText(f"THREAT LEVEL: {level}"); self.threat_label.setStyleSheet(self.threat_label.styleSheet().split(';')[0] + f"; background-color: {colors.get(level, '#B71C1C')};")
    def update_faces_list(self, names):
        self.faces_list.clear()
        unique_names = sorted(list(set(names)))
        if not unique_names: self.faces_list.addItem("---")
        for name in unique_names: item = QListWidgetItem(name); item.setForeground(QColor("#F44336") if name == "Unknown" else QColor("#66BB6A")); self.faces_list.addItem(item)
    
    def append_log(self, level, message):
        color_map = {"INFO": "#FFFFFF", "WARNING": "#FFEB3B", "ERROR": "#FF5252", "CRITICAL": "#F44336"}
        color = color_map.get(level, "#FFFFFF")
        self.log_box.append(f'<span style="color:{color};">{message}</span>')
    
    def on_worker_shutdown(self): logging.info("GUI received shutdown signal. Closing."); self.close()
    def closeEvent(self, event):
        logging.info("Close event triggered. Shutting down system...")
        if hasattr(self, 'worker'): self.worker.shutdown()
        if hasattr(self, 'worker_thread'): self.worker_thread.quit(); self.worker_thread.wait(5000)
        event.accept()

# --- GLOBAL EXCEPTION HANDLER ---

def handle_exception(exc_type, exc_value, exc_traceback):
    """Global crash catcher to log unhandled exceptions and show a message."""
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logging.critical("="*80)
    logging.critical("UNHANDLED EXCEPTION CAUGHT")
    logging.critical("="*80)
    tb_info = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    logging.critical(tb_info)
    
    # Also log to a dedicated crash file
    with open("global_crash_log.txt", "a") as f:
        f.write(f"\n\n--- {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        f.write(tb_info)

    # Show a user-friendly message box
    error_msg = (
        "A critical error occurred and the application must close.\n\n"
        "A detailed crash report has been saved to 'global_crash_log.txt'.\n\n"
        f"Error: {exc_value}"
    )
    error_box = QMessageBox()
    error_box.setIcon(QMessageBox.Icon.Critical)
    error_box.setText(error_msg)
    error_box.setWindowTitle("Application Error")
    error_box.exec()
    
    # Ensure the application exits
    QApplication.quit()

def main():
    # Setup logging to console and GUI
    log_handler = QtLogHandler()
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout)])
    
    # Gracefully handle Ctrl+C in console
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    
    app = QApplication(sys.argv)
    
    # Attach our custom logger to the main logger
    logging.getLogger().addHandler(log_handler)
    
    # Set the global exception hook AFTER creating QApplication
    sys.excepthook = handle_exception
    
    window = MainWindow()
    log_handler.log_signal.connect(window.append_log) # Connect logger to GUI
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()