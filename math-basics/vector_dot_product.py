import math

from numpy import dot

# Define two vectors
# [x, y]

print("Two vectors same direction")
print(dot([1, 0], [1, 0]))
print(dot([1, 1], [1, 1]))
print(dot([0, 1], [0, 1]))
print(dot([-1, 1], [-1, 1]))
print(dot([-1, 0], [-1, 0]))
print(dot([-1, -1], [-1, -1]))
print(dot([0, -1], [0, -1]))
print(dot([1, -1], [1, -1]))

print("Two vectors perpendicular direction")
print(dot([0, 1], [1, 0]))
print(dot([-1, 1], [1, 1]))
print(dot([-1, 0], [0, 1]))
print(dot([-1, -1], [-1, 1]))
print(dot([0, -1], [-1, 0]))
print(dot([1, -1], [-1, -1]))
print(dot([1, 0], [0, -1]))
print(dot([1, 1], [1, -1]))


print("Two vectors opposite direction")
print(dot([1, 0], [-1, 0]))
print(dot([1, 1], [-1, -1]))
print(dot([0, 1], [0, -1]))
print(dot([-1, 1], [1, -1]))
print(dot([-1, 0], [1, 0]))
print(dot([-1, -1], [1, 1]))
print(dot([0, -1], [0, 1]))
print(dot([1, -1], [-1, 1]))

print("Increasing distance")
for i in range(0, 360):
    cos_i = round(math.cos(math.radians(i)), 2)
    sin_i = round(math.sin(math.radians(i)), 2)
    dot_i = round(dot([1, 0], [cos_i, sin_i]), 2)
    print(f"cos({i}) = {cos_i} , sin({i}) = {sin_i} , dot = {dot_i}")
