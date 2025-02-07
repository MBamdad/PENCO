
import torch.nn as nn
import torch.nn.functional as F

print(bin(352))

layer = nn.Linear(10, 5)
print(layer)
nn.Linear()
import math  # (d) Module

PI_CONST = 3.1415  # (C) Constant


class Circle:  # (c) Class
    type_hint: str = "2D Shape"  # (t) Type hint / Class attribute

    def __init__(self, radius: float):  # (p) Parameter
        self.radius = radius  # (v) Instance variable

    @property
    def diameter(self) -> float:  # (P) Property
        return self.radius * 2

    def area(self) -> float:  # (m) Method
        return PI_CONST * self.radius ** 2


def compute_circumference(radius: float) -> float:  # (f) Function
    return 2 * PI_CONST * radius

compute_circumference()
# Usage
circle = Circle(2)  # (v) Variable
a = circle.diameter

print(f"Diameter: {circle.diameter}")  # (P) Property
print(f"Area: {circle.area()}")  # (m) Method
print(f"Circumference: {compute_circumference(circle.radius)}")  # (f) Function call
print(f"Using math module: {math.sqrt(16)}")  # (d) Module usage
