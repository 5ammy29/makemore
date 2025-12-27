def load_names() -> list[str]:
    words = open("data/names.txt", "r").read().splitlines()
    return words
