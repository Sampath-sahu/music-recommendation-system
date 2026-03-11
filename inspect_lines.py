path = r"d:\Music Recommendation\app.py"
with open(path, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f, start=1):
        if 270 <= i <= 298:
            print(i, repr(line))