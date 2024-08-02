
class Value:
    def __init__(self, scalar):
        self.value = scalar

    def __add__(self, other):
        assert isinstance(other, Value)
        return Value(self.value + other.value)

    def __sub__(self, other):
        return Value(self.value - other.value)

    def __mul__(self, other):
        return Value(self.value * other.value)

    def __pow__(self, other):
        return Value(self.value ** other.value)

    def __truediv__(self, other):
        return Value(self.value * other.value**-1)

    def __str__(self):
        return f"{self.value}"


def main():
    a = Value(3.2)
    b = Value(4.7)
    print(a + b)
    print(a - b)
    print(a * b)
    print(a / b)


if __name__ == "__main__":
    main()
