import ollama

print(f"Available attributes in ollama module: {dir(ollama)}")

try:
    print("\nTrying to use ollama module...")
    for attr in dir(ollama):
        if not attr.startswith('_'):
            print(f"{attr}: {type(getattr(ollama, attr))}")
except Exception as e:
    print(f"Error: {e}") 