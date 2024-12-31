import numpy as np
import matplotlib.pyplot as plt
height, width = 19, 19 #27
def ChineseCheckersPattern():
    finalpattern = "." * width
      # holes = [1,2,3,4,13,12,11,10,9,10,11,12,13,4,3,2,1]
    holes = [1,2,3,4,5,6,7,8,9,8,7,6,5,4,3,2,1]
    for n in holes:
        pattern = ""
        for i in range(n):
            pattern += "x."
        pattern = pattern[:-1]
        while len(pattern) != width:
          pattern = "." + pattern + "."
        finalpattern += pattern
    finalpattern += "." * width
    return np.array([(255,255,255) if char == 'x' else (0,0,0) for char in finalpattern], dtype=np.uint8).reshape((height, width, 3))

data_array = ChineseCheckersPattern()

# Convert the list to a NumPy array and reshape it
# data_array = np.array(data_list, dtype=np.uint8).reshape((height, width))

# Display the image
plt.imshow(data_array.squeeze())
plt.axis('off')  # Turn off axis numbers and ticks
plt.show()