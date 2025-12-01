import json
import os
from dotenv import load_dotenv

# Constants
CONFIG_FILE = 'config.json'

# Load environment variables from .env
load_dotenv()

def save_config(data_path, persist_directory, collection_name):
    """
    Save configuration to a JSON file.
    This function accepts arguments and writes them to a config.json file.
    Sensitive data (e.g., API keys) are not written to the file.
    """
    # Default paths
    if data_path is None:
        data_path = '/tmp/data'
    if persist_directory is None:
        persist_directory = '/tmp/db'
        
    config = {
        'data_path': data_path,
        'persist_directory': persist_directory,
        'collection_name': collection_name
    }
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=4)  # Add indent for better readability
    print(f"Configuration saved to {CONFIG_FILE}.")

def load_config():
    """
    Load configuration from JSON file and environment variables.
    Returns the complete configuration as a dictionary.
    """
    try:
        # Load JSON config file if it exists
        if not os.path.exists(CONFIG_FILE):
            raise FileNotFoundError(f"{CONFIG_FILE} not found. Please save the configuration first.")

        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)

        # Validate required keys in config.json
        required_keys = ['data_path', 'persist_directory', 'collection_name']
        for key in required_keys:
            if key not in config:
                raise KeyError(f"Missing required configuration key: {key}")

        # Add GROQ_API_KEY from environment variables (fallback to .env)
        config['groq_api_key'] = os.getenv('GROQ_API_KEY')
        if not config['groq_api_key']:
            raise ValueError("GROQ_API_KEY is not set in environment variables.")

        return config

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"Configuration error: {e}")
        return None