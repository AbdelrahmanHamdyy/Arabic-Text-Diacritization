def readFile(path):
    sentences = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            sentences.append(line.strip())

    return sentences
