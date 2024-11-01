
# e.g. "Shot Tags" -> "shot_tags"
def label_to_track(label: str) -> str:
    return label.lower().replace(" ", "_")

def track_to_label(key: str) -> str:
    if key == "speech_to_text":
        return "Speech to Text"
    if key == "llava_caption":
        return "LLAVA Caption"
    return key.replace("_", " ").title()