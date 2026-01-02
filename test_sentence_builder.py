from sentence_builder import SentenceBuilder

sb = SentenceBuilder()

inputs = [
    "hello", "hello",
    "PAUSE",
    "A", "A", "B",
    "PAUSE",
    "thank_you",
    "CLEAR",
    "A", "B", "C"
]

for i in inputs:
    print(i, "→", sb.add(i))
