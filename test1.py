def test_valerror(i):
    if i > 5:
        raise ValueError("Either one of `group` or `qid` should be None.")


test_valerror(4)


class MyClass:
    x = 0
    y = ""

    def __init__(self, anyNumber, anyString):
        self.x = anyNumber
        self.y = anyString

    def fn(self):
        return self.x

    def __str__(self):
        return "MyClass(x= " + str(self.x) + "y=" + self.y + ")"
    
    def testing_stuff(self):
        d []


myObject = MyClass(12345, "Hello")

print(myObject.fn())
print(myObject)
print(str(myObject))
print(myObject.__repr__())
