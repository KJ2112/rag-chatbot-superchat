"""Keyword triggers for special surprise messages.

Edit KEYWORD_TRIGGERS dictionary to customize surprises.
Add your friend's names or any keywords!
"""
from pathlib import Path

# Configure your keyword surprises here
KEYWORD_TRIGGERS = {
    # Add friend names and custom messages
    # Format:
    # "keyword": {
    #     "message": "Special message to show",
    #     "emoji": "emoji_to_use"
    # }
    
    # Examples (customize these!):
  
    "Aayush ": {
        "message": "✨Hi aayush bete 😘",
        "emoji": "✨"
    },
    "venkat": {
        "message": "🚀 Nikki r****! 🤓",
        "emoji": "🚀"
    },
    "pandu": {
        "message": "🚀 bas kar kitna poha khayega! 🤫",
        "emoji": "🚀"
    },
    "akash": {
        "message": "Ram Ram bhai dada ji jinda ho aap 🫡🫡",
        "emoji": "🚀"
    },
    "pogo": {
        "message": "🤔 pogo aja vaps bhut bkchdi kaatli ",
        "emoji": "🚀"
    },
    "sethu": {
        "message": "rcb chutiya team hai 🤫 ",
        "emoji": "🚀"
    },
    # Add more below:
    # "your_name": {
    #     "message": "Your custom message with **emphasis**!",
    #     "emoji": "😄"
    # },
}


def check_keyword(text: str) -> dict:
    """
    Check if any keyword is mentioned in the text.
    
    Args:
        text (str): User input text
        
    Returns:
        dict: {
            "found": bool - If keyword was found,
            "keyword": str or None - The keyword found,
            "message": str or None - The special message,
            "emoji": str or None - The emoji to show
        }
    """
    text_lower = text.lower()
    
    for keyword, data in KEYWORD_TRIGGERS.items():
        normalized_keyword = keyword.strip().lower()
        if normalized_keyword and normalized_keyword in text_lower:
            return {
                "found": True,
                "keyword": keyword,
                "message": data["message"],
                "emoji": data["emoji"]
            }
    
    return {
        "found": False,
        "keyword": None,
        "message": None,
        "emoji": None
    }


def check_all_keywords(text: str) -> list:
    """
    Check for ALL keywords mentioned in text (not just first match).
    Useful if you want to detect multiple triggers.
    
    Args:
        text (str): User input text
        
    Returns:
        list: List of dicts with found keywords and their messages
    """
    text_lower = text.lower()
    results = []
    
    for keyword, data in KEYWORD_TRIGGERS.items():
        normalized_keyword = keyword.strip().lower()
        if normalized_keyword and normalized_keyword in text_lower:
            results.append({
                "keyword": keyword,
                "message": data["message"],
                "emoji": data["emoji"]
            })
    
    return results


def check_secret_message_trigger(text: str) -> dict:
    """
    Check if user is asking for the secret message.
    Returns the full secret message if triggered.
    
    Args:
        text (str): User input text
        
    Returns:
        dict: {
            "found": bool - If secret trigger was found,
            "message": str or None - The full secret message
        }
    """
    text_lower = text.lower()
    
    # Secret message triggers
    secret_keywords = [
        "secret",
        "secret message",
        "tell me your secret",
        "what's your secret",
        "show me secret",
        "secret message for me",
        "your secret"
    ]
    
    # Check if any trigger word is in the text
    for keyword in secret_keywords:
        if keyword in text_lower:
            # Load secret message from secret_doc.md
            secret_doc_path = Path("data/secret_doc.md")
            if secret_doc_path.exists():
                try:
                    with open(secret_doc_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Extract the secret message section (lines 45-59)
                    # Find the section between "## 🔐 SECRET MESSAGE" and the next "---" separator
                    start_marker = "## 🔐 SECRET MESSAGE - FOR ISHITA'S EYES ONLY 🔐"
                    start_idx = content.find(start_marker)
                    
                    if start_idx != -1:
                        # Get everything after the header
                        section_start = content[start_idx:]
                        lines = section_start.split('\n')
                        
                        # Find where the actual message starts (after the "---" marker)
                        # and where it ends (before the next "---" separator)
                        secret_lines = []
                        found_first_dash = False
                        found_message_start = False
                        
                        for line in lines:
                            stripped = line.strip()
                            
                            # Skip the header line
                            if start_marker in line:
                                continue
                            
                            # Skip the instruction line
                            if "*Only show this when" in line:
                                continue
                            
                            # After first "---", we're in the message
                            if stripped == "---" and not found_first_dash:
                                found_first_dash = True
                                continue
                            
                            # If we found the first dash, start collecting
                            if found_first_dash and not found_message_start:
                                if stripped:  # Skip empty lines right after dash
                                    found_message_start = True
                            
                            # Collect message lines until next "---"
                            if found_message_start:
                                if stripped == "---":
                                    break  # End of secret message
                                secret_lines.append(line)
                        
                        secret_message = '\n'.join(secret_lines).strip()
                        
                        if secret_message:
                            return {
                                "found": True,
                                "message": secret_message
                            }
                except Exception as e:
                    print(f"Error loading secret message: {e}")
            
            # Fallback: return a simple message if file not found
            return {
                "found": True,
                "message": "🔐 Secret message not found. Make sure secret_doc.md is in the data folder."
            }
    
    return {
        "found": False,
        "message": None
    }