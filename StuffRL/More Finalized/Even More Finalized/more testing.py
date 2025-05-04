import numpy as np
from termcolor import colored
import tkinter as tk

# Define board dimensions
width, height = 8, 8

# Create a sample board with values from -1 to 7
np.random.seed(0)
board = np.random.randint(-1, 8, size=(width * height))
board = board.astype(np.int32)

# Define player colors (index -1 is mapped to index 0)
PlayertoColor = ["black", "white", "yellow", "blue", "green", "magenta", "cyan", "red"]

# Terminal output using termcolor
print("Terminal Output:")
for i in range(height):
    row = " ".join(colored(str(x) if x != -1 else " ", PlayertoColor[x + 1]) for x in board[i*width:(i+1)*width])
    print(row)

# Tkinter GUI output
root = tk.Tk()
root.title("Board Display")

for i in range(height):
    for j in range(width):
        x = board[i * width + j]
        text = str(x) if x != -1 else " "
        color = PlayertoColor[x + 1]  # Adjusted for -1 indexing
        label = tk.Label(root, text=text, width=4, height=2, bg=color, fg="white" if color != "white" else "black")
        label.grid(row=i, column=j, padx=1, pady=1)

root.mainloop()
