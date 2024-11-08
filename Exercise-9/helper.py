with open('important.txt', 'w') as f:
    for i in range(1, 21):
        f.write(f"\\section*{{Task {i}}}\n\n")
        f.write(f"\\begin{{task*}}[{i}]\n\n")
        f.write(f"\\end{{task*}}\n\n")
