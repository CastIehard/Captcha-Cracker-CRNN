import os
from datetime import datetime

class Logger:
    def __init__(self, log_path):
        """
        Initialize logger with given file path.

        Args:
            log_path (str): Path to log file.
        """
        self.log_path = log_path

        # Ensure the directory for the log file exists
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def log(self, message, timestamp=True):
        """
        Write a message to the log file.

        Args:
            message (str): The message to write.
            timestamp (bool): Whether to prepend timestamp.
        """
        if timestamp:
            message = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"

        # Open the file in append mode and write the message
        with open(self.log_path, "a") as f:
            f.write(message + "\n")

    def section(self, header):
        """
        Write a section divider with optional header.

        Args:
            header (str): Section header text.
        """
        self.log("") # Blank line before the section
        self.log("=" * 40) # Top border line
        self.log(header) # Section header
        self.log("=" * 40) # Bottom border line
        self.log("") # Blank line after the section
