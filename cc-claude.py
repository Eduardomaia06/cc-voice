#!/usr/bin/env python3
"""
Voice to Claude. Press Space to toggle recording. Sends directly to Claude CLI.
Streams responses with visual distinction between thinking and final output.
"""

import tempfile
import wave
import os
import subprocess
import sys
import json
import signal
import atexit
import gc
import time
import threading
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

# Windows keyboard input without requiring focus detection
if sys.platform == 'win32':
    import msvcrt

    def wait_for_key():
        """Wait for any key press. Returns the key."""
        return msvcrt.getch()

    def key_pressed():
        """Check if a key is pressed (non-blocking)."""
        return msvcrt.kbhit()
else:
    import termios
    import tty

    def wait_for_key():
        """Wait for any key press."""
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            return sys.stdin.read(1).encode()
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

# Enable ANSI colors on Windows
if sys.platform == 'win32':
    os.system('')  # Enables ANSI escape codes in Windows terminal
    # Alternative: use colorama if available
    try:
        import colorama
        colorama.init()
    except ImportError:
        pass

# Config
WHISPER_MODEL = "large-v3"
SAMPLE_RATE = 16000
CLAUDE_TIMEOUT = 3600  # seconds (1 hour for very long tasks)

# ANSI color codes for terminal formatting
class Colors:
    RESET = "\033[0m"
    DIM = "\033[2m"           # Dim/faint text for thinking
    ITALIC = "\033[3m"        # Italic
    CYAN = "\033[36m"         # Cyan for thinking header
    WHITE = "\033[97m"        # Bright white for response
    YELLOW = "\033[33m"       # Yellow for user input
    GREEN = "\033[32m"        # Green for success
    MAGENTA = "\033[35m"      # Magenta for tool usage
    BLUE = "\033[34m"         # Blue for tool results

    # Combined styles
    THINKING = "\033[2;3;36m"  # Dim + Italic + Cyan for thinking content
    RESPONSE = "\033[0;97m"    # Reset + Bright white for response
    TOOL = "\033[2;35m"        # Dim + Magenta for tool info


class Spinner:
    """Animated spinner with evolving status text."""

    # Braille spinner frames for smooth animation
    FRAMES = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']

    # Status words that cycle for visual interest
    STATUS_WORDS = [
        'Connecting', 'Processing', 'Thinking', 'Analyzing',
        'Working', 'Computing', 'Reasoning', 'Evaluating'
    ]

    def __init__(self):
        self._thread = None
        self._stop_event = threading.Event()
        self._status = "Processing"
        self._lock = threading.Lock()
        self._visible = False

    def start(self, status="Processing"):
        """Start the spinner animation."""
        if self._thread and self._thread.is_alive():
            self.update(status)
            return

        self._stop_event.clear()
        self._status = status
        self._visible = True
        self._thread = threading.Thread(target=self._animate, daemon=True)
        self._thread.start()

    def _animate(self):
        """Animation loop running in background thread."""
        frame_idx = 0
        word_idx = 0
        word_timer = 0

        while not self._stop_event.is_set():
            with self._lock:
                status = self._status

            # Cycle through status words every ~2 seconds if using default
            if status == "Processing":
                word_timer += 1
                if word_timer >= 20:  # 20 * 0.1s = 2 seconds
                    word_timer = 0
                    word_idx = (word_idx + 1) % len(self.STATUS_WORDS)
                    status = self.STATUS_WORDS[word_idx]

            frame = self.FRAMES[frame_idx]
            # \r moves cursor to start of line, \033[K clears to end of line
            sys.stdout.write(f"\r{Colors.CYAN}{frame} {status}...{Colors.RESET}\033[K")
            sys.stdout.flush()

            frame_idx = (frame_idx + 1) % len(self.FRAMES)
            self._stop_event.wait(0.1)

    def update(self, status):
        """Update the status text."""
        with self._lock:
            self._status = status

    def stop(self, clear=True):
        """Stop the spinner and optionally clear the line."""
        if not self._visible:
            return

        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=0.5)

        if clear:
            # Clear the spinner line
            sys.stdout.write("\r\033[K")
            sys.stdout.flush()

        self._visible = False

    def stop_with_message(self, message, color=Colors.GREEN):
        """Stop spinner and show a final message."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=0.5)

        sys.stdout.write(f"\r{color}{message}{Colors.RESET}\033[K\n")
        sys.stdout.flush()
        self._visible = False


# Global spinner instance
spinner = Spinner()

# State
recording = False
audio_buffer = []
stream = None
model = None
is_first_message = True  # Track if this is the first message in the session
cleanup_done = False


class ToolPrinter:
    """Simple tool output."""

    def __init__(self):
        self.start_time = 0
        self.first_in_group = True

    def start(self, tool_name):
        """Print tool. First tool after text gets a newline."""
        self.start_time = time.time()
        if self.first_in_group:
            print(f"\n{Colors.TOOL}▸ {tool_name}{Colors.RESET}", flush=True)
            self.first_in_group = False
        else:
            print(f"{Colors.TOOL}▸ {tool_name}{Colors.RESET}", flush=True)

    def complete(self):
        """Return elapsed time."""
        elapsed = time.time() - self.start_time if self.start_time else None
        self.start_time = 0
        return elapsed

    def stop(self):
        """Reset."""
        self.start_time = 0

    def reset_group(self):
        """Call when new text/response starts."""
        self.first_in_group = True


# Global tool printer
progress = ToolPrinter()


def cleanup():
    """Clean up all resources on exit."""
    global model, stream, cleanup_done

    if cleanup_done:
        return
    cleanup_done = True

    print(f"\n{Colors.DIM}Cleaning up...{Colors.RESET}")

    # Stop and close audio stream
    if stream is not None:
        try:
            stream.stop()
            stream.close()
        except:
            pass

    # Clean up Whisper model and free GPU memory
    if model is not None:
        try:
            del model
        except:
            pass

    # Force garbage collection
    gc.collect()

    # Clear CUDA cache if available
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass

    print(f"{Colors.DIM}Done.{Colors.RESET}")


def signal_handler(signum, frame):
    """Handle termination signals."""
    cleanup()
    sys.exit(0)


# Register cleanup handlers
atexit.register(cleanup)
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
if sys.platform == 'win32':
    signal.signal(signal.SIGBREAK, signal_handler)


def close_current_block(in_thinking, in_response):
    """Close any open formatting blocks."""
    if in_thinking or in_response:
        print(f"{Colors.RESET}", end="", flush=True)  # Just reset colors
    return False, False


def format_ask_user_question(tool_input):
    """Format and display an AskUserQuestion for the user to respond to."""
    questions = tool_input.get('questions', [])
    if not questions:
        return "Claude has a question for you."

    lines = []
    for i, q in enumerate(questions):
        question_text = q.get('question', 'Question?')
        options = q.get('options', [])

        lines.append(f"\n{Colors.CYAN}❓ {question_text}{Colors.RESET}")

        if options:
            for j, opt in enumerate(options, 1):
                label = opt.get('label', f'Option {j}')
                desc = opt.get('description', '')
                if desc:
                    lines.append(f"   {Colors.WHITE}{j}. {label}{Colors.RESET} - {Colors.DIM}{desc}{Colors.RESET}")
                else:
                    lines.append(f"   {Colors.WHITE}{j}. {label}{Colors.RESET}")

    return '\n'.join(lines)


def format_tool_detail(tool_name, tool_input):
    """Format tool usage with helpful details about what it's doing."""
    # Extract the most relevant info based on tool type
    if tool_name == 'Read':
        path = tool_input.get('file_path', '')
        if path:
            # Show just filename for brevity
            filename = os.path.basename(path)
            return f"Reading: {filename}"
        return "Reading file..."

    elif tool_name == 'Write':
        path = tool_input.get('file_path', '')
        if path:
            filename = os.path.basename(path)
            return f"Writing: {filename}"
        return "Writing file..."

    elif tool_name == 'Edit':
        path = tool_input.get('file_path', '')
        if path:
            filename = os.path.basename(path)
            return f"Editing: {filename}"
        return "Editing file..."

    elif tool_name == 'Bash':
        cmd = tool_input.get('command', '')
        desc = tool_input.get('description', '')
        if desc:
            return f"Running: {desc}"
        elif cmd:
            # Truncate long commands
            if len(cmd) > 50:
                cmd = cmd[:47] + "..."
            return f"Running: {cmd}"
        return "Running command..."

    elif tool_name == 'Glob':
        pattern = tool_input.get('pattern', '')
        return f"Searching: {pattern}" if pattern else "Searching files..."

    elif tool_name == 'Grep':
        pattern = tool_input.get('pattern', '')
        return f"Searching for: {pattern}" if pattern else "Searching content..."

    elif tool_name == 'Task':
        desc = tool_input.get('description', '')
        if desc:
            return f"Task: {desc}"
        return "Running task..."

    elif tool_name == 'WebFetch':
        url = tool_input.get('url', '')
        if url:
            # Show domain only
            from urllib.parse import urlparse
            try:
                domain = urlparse(url).netloc
                return f"Fetching: {domain}"
            except:
                return f"Fetching URL..."
        return "Fetching web content..."

    elif tool_name == 'WebSearch':
        query = tool_input.get('query', '')
        return f"Searching: {query}" if query else "Web search..."

    elif tool_name == 'TodoWrite':
        return "Updating task list..."

    elif tool_name == 'AskUserQuestion':
        return "Asking question..."

    else:
        return f"Using: {tool_name}"


def stream_claude_response(text):
    """Send text to Claude CLI and stream response with formatted thinking/response."""
    global is_first_message

    print(f"\n{Colors.YELLOW}> {text}{Colors.RESET}\n")

    try:
        cmd = [
            'claude',
            '--dangerously-skip-permissions',
            '-p', text,
            '--output-format', 'stream-json',
            '--verbose',
            '--include-partial-messages'
        ]
        if not is_first_message:
            cmd.insert(1, '--continue')

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )

        current_block_type = None
        in_thinking = False
        in_response = False
        in_tool = False
        response_text = []
        tool_count = 0
        current_tool_name = ""
        current_tool_input_json = ""
        pending_question = None  # Track if AskUserQuestion was called

        # Start spinner while waiting for first response
        spinner.start("Connecting")

        for line in process.stdout:
            # Stop spinner when we receive any data
            spinner.stop()
            line = line.strip()
            if not line:
                continue

            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            event_type = event.get('type')

            # Handle stream events
            if event_type == 'stream_event':
                inner_event = event.get('event', {})
                inner_type = inner_event.get('type')

                # Content block start - detect thinking vs text vs tool_use
                if inner_type == 'content_block_start':
                    content_block = inner_event.get('content_block', {})
                    block_type = content_block.get('type')

                    if block_type == 'thinking':
                        progress.stop()
                        progress.reset_group()
                        if in_response:
                            in_thinking, in_response = close_current_block(False, True)
                        if not in_thinking:
                            print(f"\n{Colors.CYAN}Thinking...{Colors.RESET}")
                            print(Colors.THINKING, end='')
                            in_thinking = True
                        current_block_type = 'thinking'

                    elif block_type == 'text':
                        progress.stop()
                        progress.reset_group()
                        if in_thinking:
                            in_thinking, in_response = close_current_block(True, False)
                        if not in_response:
                            print(f"\n{Colors.RESPONSE}", end='')
                            in_response = True
                        current_block_type = 'text'

                    elif block_type == 'tool_use':
                        progress.stop()
                        # Close any open blocks
                        if in_thinking or in_response:
                            in_thinking, in_response = close_current_block(in_thinking, in_response)
                        current_tool_name = content_block.get('name', 'Unknown')
                        current_tool_input_json = ""
                        tool_count += 1
                        current_block_type = 'tool_use'
                        in_tool = True

                # Content delta - stream the text
                elif inner_type == 'content_block_delta':
                    delta = inner_event.get('delta', {})
                    delta_type = delta.get('type')

                    if delta_type == 'thinking_delta':
                        thinking_text = delta.get('thinking', '')
                        print(thinking_text, end='', flush=True)
                    elif delta_type == 'text_delta':
                        text_chunk = delta.get('text', '')
                        print(text_chunk, end='', flush=True)
                        response_text.append(text_chunk)
                    elif delta_type == 'input_json_delta':
                        # Accumulate tool input JSON
                        current_tool_input_json += delta.get('partial_json', '')

                # Content block stop
                elif inner_type == 'content_block_stop':
                    if current_block_type == 'tool_use':
                        # Parse the accumulated tool input
                        try:
                            tool_input = json.loads(current_tool_input_json) if current_tool_input_json else {}
                        except json.JSONDecodeError:
                            tool_input = {}

                        # Track AskUserQuestion for later handling
                        if current_tool_name == 'AskUserQuestion':
                            pending_question = tool_input

                        # Format and display tool info
                        tool_detail = format_tool_detail(current_tool_name, tool_input)
                        progress.start(tool_detail)
                        # Start spinner while tool executes
                        spinner.start("Executing")
                        in_tool = False

                # Message stop
                elif inner_type == 'message_stop':
                    progress.stop()
                    in_thinking, in_response = close_current_block(in_thinking, in_response)

            # Handle assistant message (for multi-turn with tools)
            elif event_type == 'assistant':
                # New assistant turn after tool results - reset response state to allow new response block
                # This happens when Claude uses tools and then continues responding
                if in_response:
                    # Don't close, just reset flag so next text block can continue
                    pass
                in_response = False  # Reset so next text block opens fresh if needed

            # Handle user message (tool results being fed back)
            elif event_type == 'user':
                spinner.stop()
                progress.complete()

                message = event.get('message', {})
                content = message.get('content', [])
                for block in content:
                    if block.get('type') == 'tool_result':
                        is_error = block.get('is_error', False)

                        if is_error:
                            error_content = block.get('content', '')
                            # Show error details
                            if isinstance(error_content, str) and error_content:
                                error_preview = error_content[:80] + "..." if len(error_content) > 80 else error_content
                                print(f"{Colors.YELLOW}  ✗ {error_preview}{Colors.RESET}", flush=True)
                            else:
                                print(f"{Colors.YELLOW}  ✗ Failed{Colors.RESET}", flush=True)

            # Handle errors
            elif event_type == 'error':
                spinner.stop()
                progress.stop()
                error_msg = event.get('error', {}).get('message', 'Unknown error')
                print(f"\n{Colors.YELLOW}⚠ Error: {error_msg}{Colors.RESET}")

            # Handle result event (final summary)
            elif event_type == 'result':
                spinner.stop()
                progress.stop()
                # Close any remaining open blocks
                in_thinking, in_response = close_current_block(in_thinking, in_response)

                subtype = event.get('subtype', '')
                if subtype == 'error':
                    error_msg = event.get('result', 'Unknown error')
                    print(f"\n{Colors.YELLOW}⚠ Error: {error_msg}{Colors.RESET}")

                # Show summary stats
                cost = event.get('cost_usd', 0)
                duration = event.get('duration_ms', 0) / 1000
                turns = event.get('num_turns', 1)

                stats = []
                if duration > 0:
                    if duration < 60:
                        stats.append(f"{duration:.1f}s")
                    else:
                        mins = int(duration // 60)
                        secs = int(duration % 60)
                        stats.append(f"{mins}m{secs}s")
                if tool_count > 0:
                    stats.append(f"{tool_count} tool{'s' if tool_count > 1 else ''}")
                if cost > 0:
                    stats.append(f"${cost:.4f}")

                if stats:
                    print(f"\n{Colors.DIM}[{' · '.join(stats)}]{Colors.RESET}")

        spinner.stop()  # Ensure spinner is stopped
        process.wait()  # No timeout - wait indefinitely
        is_first_message = False

        # Handle pending question if AskUserQuestion was called
        if pending_question:
            question_display = format_ask_user_question(pending_question)
            print(question_display)
            print(f"\n{Colors.DIM}Press Space to record your answer...{Colors.RESET}")

            # Wait for space to start recording
            while True:
                key = wait_for_key()
                if key == b' ':
                    break
                elif key == b'\x03':  # Ctrl+C
                    return ''.join(response_text) if response_text else ""

            # Record and transcribe user response
            user_answer = record_response_sync()
            if user_answer:
                # Send answer as follow-up message (will use --continue)
                return stream_claude_response(user_answer)

        return ''.join(response_text) if response_text else ""

    except KeyboardInterrupt:
        spinner.stop()
        progress.stop()
        print(f"\n{Colors.YELLOW}Cancelled by user{Colors.RESET}")
        process.terminate()
        try:
            process.wait(timeout=5)
        except:
            process.kill()
        return "[Cancelled]"
    except FileNotFoundError:
        spinner.stop()
        progress.stop()
        return "[Error: Claude CLI not found. Make sure 'claude' is in PATH]"
    except Exception as e:
        spinner.stop()
        progress.stop()
        return f"[Error: {str(e)}]"


def record_response_sync():
    """Record audio synchronously and return transcribed text. Used for answering questions."""
    print(f"{Colors.YELLOW}● Recording answer...{Colors.RESET} (Space to stop)")

    audio_chunks = []
    is_recording = True

    def callback(indata, frames, time_info, status):
        if is_recording:
            audio_chunks.append(indata.copy())

    rec_stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype=np.float32, callback=callback)
    rec_stream.start()

    # Wait for Space to stop
    while True:
        key = wait_for_key()
        if key == b' ':
            break
        elif key == b'\x03':  # Ctrl+C
            rec_stream.stop()
            rec_stream.close()
            return None

    is_recording = False
    rec_stream.stop()
    rec_stream.close()

    if len(audio_chunks) < 10:
        print(f"{Colors.DIM}Recording too short{Colors.RESET}")
        return ""

    audio = np.concatenate(audio_chunks, axis=0)

    # Save temp wav
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_path = f.name
        with wave.open(f.name, 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(SAMPLE_RATE)
            wav.writeframes((audio * 32767).astype(np.int16).tobytes())

    # Transcribe
    print(f"{Colors.DIM}Transcribing...{Colors.RESET}")
    segments, _ = model.transcribe(temp_path, beam_size=5, vad_filter=True, language="en")
    text = " ".join(s.text for s in segments).strip()
    os.unlink(temp_path)

    if text:
        print(f"{Colors.GREEN}> {text}{Colors.RESET}")

    return text


def send_to_claude(text):
    """Send text to Claude CLI with streaming response."""
    return stream_claude_response(text)


def start_recording():
    global recording, audio_buffer, stream

    if recording:
        return

    audio_buffer = []
    recording = True

    def callback(indata, frames, time, status):
        if recording:
            audio_buffer.append(indata.copy())

    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype=np.float32, callback=callback)
    stream.start()
    print(f"{Colors.YELLOW}● Recording...{Colors.RESET} (Space to stop)")


def stop_recording():
    global recording, stream

    if not recording:
        return

    recording = False
    stream.stop()
    stream.close()

    if len(audio_buffer) < 10:
        print(f"{Colors.DIM}Recording too short{Colors.RESET}")
        return

    audio = np.concatenate(audio_buffer, axis=0)

    # Save temp wav
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_path = f.name
        with wave.open(f.name, 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(SAMPLE_RATE)
            wav.writeframes((audio * 32767).astype(np.int16).tobytes())

    # Transcribe
    print(f"{Colors.DIM}Transcribing...{Colors.RESET}")
    segments, _ = model.transcribe(temp_path, beam_size=5, vad_filter=True, language="en")
    text = " ".join(s.text for s in segments).strip()
    os.unlink(temp_path)

    if text:
        send_to_claude(text)  # Response is already printed via streaming
        print()
        print(f"{Colors.DIM}─────────────────────────────────────────────────{Colors.RESET}")
        print(f"{Colors.DIM}Ready. Press Space to record.{Colors.RESET}")
    else:
        print(f"{Colors.DIM}No speech detected{Colors.RESET}")


def toggle_recording():
    if recording:
        stop_recording()
    else:
        start_recording()


if __name__ == "__main__":
    print(f"{Colors.DIM}Loading Whisper model...{Colors.RESET}")
    model = WhisperModel(WHISPER_MODEL, device="cuda", compute_type="float16")
    print(f"{Colors.GREEN}Ready.{Colors.RESET} Press {Colors.WHITE}Space{Colors.RESET} to record, {Colors.WHITE}Ctrl+C{Colors.RESET} to exit.")
    print(f"{Colors.DIM}Conversation context maintained until exit.{Colors.RESET}")
    print()

    try:
        while True:
            key = wait_for_key()
            if key == b' ':  # Space
                toggle_recording()
            elif key == b'\x03':  # Ctrl+C
                break
    except KeyboardInterrupt:
        pass
    finally:
        cleanup()
