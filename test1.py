import os
# 1 testing exceptions
class Test:
    def __init__(self, i):
        self.i = i

    def test_valerror(self):
        if self.i > 5:
            raise ValueError("Either one of `group` or `qid` should be None.")


obj = Test(4)
obj.test_valerror()

# 2 testing str and repr
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

    # def testing_stuff(self):


myObject = MyClass(12345, "Hello")

print("myObject.fn:         ",myObject.fn())
print("myObject:            ", myObject)
print("str(myObject):       ", str(myObject))
print("myObject.__repr__(): ", myObject.__repr__())

# 3 testing joins
x = (f"{''.join(map(chr, [98, 114, 117, 109]))}, " * 5)[:-2]
print("join example:        ",x)

os.chdir('./model')
print(format(os.getcwd()))
