---
layout: default
title: Python and Its Libraries
parent: "Appendices"
nav_order: 805
has_children: false
---

### Python and Its Libraries

In this *appendix*, we present a brief review of certain aspects of the Python programming language, as well as some key libraries that are essential to understand the code examples..

Python is a widely used programming language (its source code is available under the GNU General Public License - GPL), originally developed by Dutch programmer Guido van Rossum. It supports multiple programming paradigms.

Although it is an interpreted language rather than a compiled one—and therefore has some execution overhead—Python has a gentle learning curve and its simplicity allows for quick productivity. In addition, performance issues are often mitigated using libraries like NumPy, which are implemented in compiled languages such as C.

#### First Steps

Currently, two major Python versions exist (with several sub-versions): 2.x and 3.x. Python 2 has reached end of life and will no longer receive updates. In this book, we will use version 3, which is fully supported by the most popular Python libraries. For our purposes, differences between both versions are minimal. The most notable difference is that in Python 3, print is a function and requires parentheses, unlike Python 2.

Python allows importing entire libraries or specific functions using the import and from keywords:


    from random import randint
    randomint = randint(1, 100)
    print(randomint)

23


    import random
    randomint = random.randint(1, 100)
    print(randomint)

2

In the second case, we import the entire module and must prefix the function name with the module name.

Refer to the official Python documentation[^1] for comprehensive information about the language and its libraries.

#### Indentation in Python

Python does not require a special character to terminate statements. Instead, code blocks are defined by indentation (i.e., no braces {} are used for class, function, or control flow definitions).

Statements that begin a block must end with a colon :. The amount of indentation can vary, but it must be consistent within the same block. Incorrect indentation raises an error:


    IndentationError: unexpected indent

#### Variables, Operators, and Data Types

As in most languages, variables in Python are data containers defined by a name and a value. Python is case-sensitive, meaning train_accuracy and Train_Accuracy are different variables.

Python uses implicit typing—no need to declare types. Variables can be of many types:


    x = 10
    print(x)
    print(type(x))

10

\<class 'int'\>

You can change the type simply by assigning a new value:


    x = "hello"
    print(x)
    print(type(x))

hello

\<class 'str'\>

Other examples of types:


    x = 10.0
    print(x)
    print(type(x))

10.0

\<class 'float'\>


    x = True
    print(x)
    print(type(x))

True

\<class 'bool'\>

Python supports arithmetic operations as expected:


    print(5 / 2)   

2.5


    print(5 // 2)  #  (floor division)

2


    print(5 % 2)   #  (modulo)

1


    print(5 ** 2)  # (exponent)

25

Assignment uses =, while comparison uses ==. Python also supports += and -= operators, even with strings:


    int_var = 10
    int_var += 10
    print(int_var)  

20


    str_var = "Deep"
    str_var += " Learning"
    print(str_var)  

Deep Learning

Multiple assignments and value swapping:


    int_var, str_var = str_var, int_var
    print(int_var)  

Deep Learning


    print(str_var)  

20

#### Data Structures

The main data structures available in Python are lists, tuples, and dictionaries.

*Lists* are ordered, mutable collections of values, enclosed in square brackets and separated by commas. They can be seen as one-dimensional arrays, but Python also supports nested lists (lists of lists) or tuples. Moreover, a list can contain elements of different types. For example:


    x = [5, "hello", 1.8]
    print(x)

\[5, 'hello', 1.8\]

In this case, the variable x is a list that contains an integer, a string, and a float. We can get the number of elements in a list using the len() function:


    len(x)

3

We can append elements to a list using the append() method:


    x.append(100)
    print(x)
    print(len(x))

\[5, 'hello', 1.8, 100\]

4

It is also possible to modify an element of the list or perform operations between lists:


    x[1] = "deep"
    print(x)

\[5, 'deep', 1.8, 100\]


    y = [2.1, "learning"]
    z = x + y
    print(z)

\[5, 'deep', 1.8, 100, 2.1, 'learning'\]

Indexing and slicing allow us to retrieve specific elements from a list. Keep in mind that indices can be positive (starting from 0) or negative (from -1 backward, where -1 refers to the last element in the list):


    x = [5, "deep", 1.8]
    print("First element -->    x[0]:", x[0])
    print("Second element -->   x[1]:", x[1])
    print("Last element -->     x[-1]:", x[-1])
    print("Second last -->      x[-2]:", x[-2])

First element --\> x\[0\]: 5

Second element --\> x\[1\]: deep

Last element --\> x\[-1\]: 1.8

Second last --\> x\[-2\]: deep

To slice lists, we use the colon symbol :. Here are some examples:


    print("All elements -->")
    print("x[:]:", x[:])

    print("From index 1 to end -->")
    print("x[1:]:", x[1:])



    print("From index 1 to 2 (excluding 2) -->")
    print("x[1:2]:", x[1:2])

    print("From index 0 up to last (excluding last) -->")
    print("x[:-1]:", x[:-1])

All elements --\>

x\[:\]: \[5, 'deep', 1.8\]

From index 1 to end --\>

x\[1:\]: \['deep', 1.8\]

From index 1 to 2 (excluding 2) --\>

x\[1:2\]: \['deep'\]

From index 0 up to last (excluding last) --\>

x\[:-1\]: \[5, 'deep'\]

*Tuples* are ordered and immutable collections (i.e., once created, their content cannot be changed). They are typically used to store values that should remain constant. Tuples are enclosed in parentheses:


    x = (1, "deep", 3)
    print(x)
    print(x[1])

(1, 'deep', 3)

deep

Tuples can be extended by concatenation:


    x = x + (4, "learning")
    print(x)
    print(x[-1])

(1, 'deep', 3, 4, 'learning')

learning

*Dictionaries* are unordered, mutable, and indexed collections of key-value pairs. That is, they are associative arrays where each key is linked to a specific value. A key must be unique—duplicate keys are overwritten.

Unlike lists or tuples, dictionaries are not indexed numerically and do not maintain a specific order (although in Python 3.7+ insertion order is preserved). Key-value pairs can be added and accessed as follows:


    person = {'name': 'Jordi Torres',
              'profession': 'professor'}
    print(person)
    print(person['name'])
    print(person['profession'])

{'name': 'Jordi Torres', 'profession': 'professor'}

Jordi Torres

professor

Modifying a value or adding a new key-value pair is straightforward:


    person['profession'] = 'researcher'
    print(person)

{'name': 'Jordi Torres', 'profession': 'researcher'}


    person['group'] = 24
    print(person)

{'name': 'Jordi Torres', 'profession': 'researcher', 'group': 24}

This dictionary now has three elements:


    print(len(person))

3

Finally, remember that all these data structures—lists, tuples, and dictionaries—can contain any type of data, including other structures, and support mixing data types within the same structure.

#### Control Flow Statements

Python provides several types of control flow statements to programmers, including if, for, and while. Let’s briefly review their syntax.

The *if statement* is commonly used in conjunction with elif (short for *else if*) and else to define alternative conditions. A colon : is required at the end of each condition to mark the beginning of the block, and the following line must be properly indented.


    x = 6.5
    if x < 5:
        rating = 'low'
    elif x <= 7:  # elif = else if
        rating = 'medium'
    else:
        rating = 'high'
    print(rating)

medium

You can also use a boolean variable in a condition:


    x = True
    if x:
        print("condition is met")

condition is met

Python, like many other languages, supports writing these conditionals in a single line:


    a = 20
    if a >= 22:
        print("Paris")
    elif a <= 21:
        print("Barcelona")

Barcelona

Using a one-line conditional expression (ternary operator):


    b = 1
    print("Paris" if b >= 92 else "Barcelona")

Barcelona

The *for loop* is one of the most frequently used constructs in this book. It allows iteration over a collection of values such as lists, tuples, or dictionaries. The indented code block is executed once for each element in the collection:


    for a in range(1, 4):
        print(a)

1

2

3

In this example, we use the range() function to generate a sequence of numbers, starting at 1 and ending before 4. The values do not need to be numerical. For example:


    universities = ["MIT", "ETH", "UPC"]
    for university in universities:
        print(university)

MIT

ETH

UPC

When a for loop encounters the break command, it immediately exits the loop. Any remaining elements in the list are skipped:


    for university in universities:
        if university == "ETH":
            break
        print(university)

MIT

Conversely, when the loop encounters the continue command, it skips the rest of the code for that iteration and continues with the next item:


    for university in universities:
        if university == "ETH":
            continue
        print(university)

MIT

ETH

The *while loop* continues to execute as long as the specified condition is True. The continue and break commands can also be used within while loops:


    x = 3
    while x > 0:
        x = x - 1
        print(x)

2

1

0

#### 

#### Functions

Python allows defining functions using the keyword def followed by the function name:


    def my_university(university="UPC"):
        y = "My university is " + university
        return y

    print(my_university("MIT"))
    print(my_university(university="ETH"))
    print(my_university())

My university is MIT

My university is ETH

My university is UPC

Optionally, arguments can be assigned default values (such as "UPC" in this example), meaning they are not required when calling the function. While not mandatory, it is considered good practice to use keyword arguments (e.g., university="ETH") for clarity.

Python functions can return a single value or a tuple of multiple values:


    def foo():
        a, b = 4, 5
        return a, b

    foo()

(4, 5)

Python also supports lambda functions (also known as anonymous functions). These provide a concise way to declare functions in a single line. They are typically used when the function is needed only once and does not require a name. Like regular functions, lambda functions accept parameters.

Here's an example that computes the Fibonacci sequence using a lambda function:


    fibonacci = (lambda x: 1 if x <= 2 else 
                 fibonacci(x - 1) + fibonacci(x - 2))

    fibonacci(10)

55

#### Classes

Classes are object constructors and are a fundamental component of object-oriented programming in Python. They are composed of a set of methods that define the behavior of the class. Classes act as blueprints that define and group the properties and operations of the objects they instantiate.

In Python, a class is defined using the class keyword followed by a generic name. To make it concrete, let’s look at the following simple example of a calculator class:


    class Calculator(object):
        """A simple calculator class"""

        def __init__(self):
            """Initializes the calculator"""
            self.value = 0

        def add(self, n):
            """Adds number n to the current value"""
            self.value += n

        def get_value(self):
            """Returns the current value"""
            return self.value

Classes define a special method named \_\_init\_\_() that acts as the constructor. The parameter self refers to the instance itself (similar to this in other languages).

Once the Calculator class is defined, we can use it by instantiating an object and calling its methods:


    calc = Calculator()
    calc.add(2)
    print(calc.get_value())
    calc.add(2)
    print(calc.get_value())

2

4

Python also supports iterators, a type of object that enables iteration over data containers such as lists, tuples, files, sockets, and more.

Iterable objects return iterators that can access values one by one in sequence. This is useful in many scenarios. Here's an example:


    vec = [1, 2, 3]
    it = iter(vec)


    print(next(it))  # 1
    print(next(it))  # 2
    print(next(it))  # 3

    print(type(vec))  # <class 'list'>
    print(type(it))   # <class 'list_iterator'>

1

2

3

class 'list'

class 'list_iterator'

In this example, vec is iterable, and it is an iterator that returns the elements of vec one by one using next().

#### Decorators

Functions allow us to modularize and reuse code. However, we may often want to add behavior before or after the function executes—sometimes across many functions. For that, Python provides decorators.

A decorator is a function that takes another function as input, adds behavior (before or after), and returns a new function. The original function remains unchanged internally. This is why it is referred to as “decorating” the original.

Decorators are a powerful and elegant Python feature. They allow us to treat functions as first-class objects—passing them as arguments to other functions and wrapping them dynamically.

Let’s imagine a simple function that prints "Hello":


    def say_hello():
        print("Hello")

    say_hello()

Hello

Now suppose we want to add a message before and after the greeting, without modifying the body of say_hello(). We can achieve this by defining a decorator:


    def say_hello():
        print("Hello")

    def my_decorator(func):
        def wrapper():
            print("[Notice: I'm about to say something]")
            func()
            print("[Notice: I have just said something]")
        return wrapper

    say_hello = my_decorator(say_hello)
    say_hello()

\[Notice: I'm about to say something\]

Hello

\[Notice: I have just said something\]

Python provides a more elegant syntax for applying decorators by using the @ symbol. Instead of writing:


    say_hello = my_decorator(say_hello)

We can decorate the function as follows:


    @my_decorator
    def say_hello():
        print("Hello")

When we now call say_hello(), we get the same result:

\[Notice: I'm about to say something\]

Hello

\[Notice: I have just said something\]

Decorators are heavily used in frameworks such as TensorFlow 2.0. For example, the @tf.function decorator transforms a Python function into a graph that can be optimized and executed efficiently on available hardware.


    @tf.function
    def my_model(...):
        ...

This provides a powerful level of hardware abstraction, allowing you to write standard Python code while taking advantage of accelerated execution.

#### NumPy Library

There are several Python libraries that we will need throughout this book to implement and run neural networks. Among them, the most relevant are:

- NumPy: the core library for efficient numerical computation. It forms the foundation for many other scientific libraries in Python.

- Pandas: a library built on top of NumPy for data structures and exploratory data analysis.

- Matplotlib: a popular visualization library that we will use to generate graphs and charts throughout this book.

- Scikit-Learn: the most widely used general-purpose Machine Learning library in Python, with many well-known algorithms and preprocessing tools.

Whenever needed, we will introduce and explain these libraries directly in the code used in the book. However, we will review here some core NumPy concepts that are essential for understanding how tensors work, as they appear frequently in Deep Learning frameworks.

NumPy is primarily used for the manipulation and processing of array-based data. Its popularity stems from its high performance (since it is written in C) and its simple and expressive API.

#### Tensor

Deep learning frameworks rely on NumPy arrays as their primary data structure. These multidimensional arrays are commonly referred to as tensors. Conceptually, a tensor has three main attributes:

- Number of axes (rank): A tensor containing a single number is called a *scalar*, or a 0-dimensional tensor (0D). A one-dimensional array of numbers is a *vector* or 1D tensor. An array of vectors is a *matrix* or 2D tensor. If we group a set of matrices, we get a 3D tensor, which we can visualize as a cube. By extending this idea, we can create 4D tensors, and so on. In NumPy, this is represented by the attribute ndim.

- Shape: A tuple of integers that specifies the size of the tensor along each axis. In NumPy, this is accessed using the shape attribute. For example, a vector with 5 elements has shape (5,), while a scalar has an empty shape ().

- Data type: This indicates the type of data stored in the tensor, such as uint8, float32, float64, etc. In deep learning applications, it is rare to encounter tensors of type char, and string tensors are almost never used. In NumPy, the data type is specified by the dtype attribute.

Let’s examine these attributes using a few examples.

Scalar (0D tensor):


    import numpy as np

    x = np.array(8) 
    print("x: ", x)
    print("x ndim: ", x.ndim)
    print("x shape:", x.shape)
    print("x size: ", x.size)
    print("x dtype:", x.dtype)

x: 8

x ndim: 0

x shape: ()

x size: 1

x dtype: int64

Vector (1D tensor):


    x = np.array([2.3, 4.2, 3.3, 1.8])
    print("x: ", x)
    print("x ndim: ", x.ndim)
    print("x shape:", x.shape)
    print("x size: ", x.size)
    print("x dtype:", x.dtype)

x: \[2.3 4.2 3.3 1.8\]

x ndim: 1

x shape: (4,)

x size: 4

x dtype: float64

Matrix (2D tensor):


    x = np.array([[1,2,3], [4,5,6], [7,8,9]])
    print("x:\n", x)
    print("x ndim: ", x.ndim)
    print("x shape:", x.shape)
    print("x size: ", x.size)
    print("x dtype:", x.dtype)

x:

\[\[1 2 3\]

\[4 5 6\]

\[7 8 9\]\]

x ndim: 2

x shape: (3, 3)

x size: 9

x dtype: int64

3D Tensor:


    x = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
    print("x:\n", x)
    print("x ndim: ", x.ndim)
    print("x shape:", x.shape)
    print("x size: ", x.size)
    print("x dtype:", x.dtype)

x:

\[\[\[1 2\]

\[3 4\]\]

\[\[5 6\]

\[7 8\]\]\]

x ndim: 3

x shape: (2, 2, 2)

x size: 8

x dtype: int64

NumPy also provides several useful functions for quickly creating tensors:


    print("np.zeros((3,3)):\n", np.zeros((3,3)))
    print("np.ones((3,3)):\n", np.ones((3,3)))
    print("np.eye((3)) (identity matrix):\n", np.eye(3))
    print("np.random.random((3,3)):\n", np.random.random((3,3)))

np.zeros((3,3)):

\[\[0. 0. 0.\]

\[0. 0. 0.\]

\[0. 0. 0.\]\]

np.ones((3,3)):

\[\[1. 1. 1.\]

\[1. 1. 1.\]

\[1. 1. 1.\]\]

np.eye((3)) (identity matrix):

\[\[1. 0. 0.\]

\[0. 1. 0.\]

\[0. 0. 1.\]\]

np.random.random((3,3)):

\[\[0.36 0.61 0.07\]

\[0.37 0.93 0.65\]

\[0.40 0.79 0.32\]\]

#### Tensor Manipulation

How can we access and manipulate parts of a tensor? As with Python lists, indexing starts at 0. Negative indexing is also allowed, where -1 refers to the last element. To slice a portion of a tensor, we can use the colon symbol (:). Consider the following example:


    x = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
    print(x)
    print("x column 1: ", x[:, 1])
    print("x row 0: ", x[0, :])
    print("x rows 0,1 & cols 1,2:\n", x[0:2, 1:3])

\[\[ 1 2 3 4\]

\[ 5 6 7 8\]

\[ 9 10 11 12\]\]

x column 1: \[ 2 6 10\]

x row 0: \[1 2 3 4\]

x rows 0,1 & cols 1,2:

\[\[2 3\]

\[6 7\]\]

Let us now explore matrix multiplication, one of the most important operations in NumPy. The dot product between two matrices requires that the inner dimensions match. For example, if a has shape (2, 3) and b has shape (3, 2), the resulting product will have shape (2, 2):


    a = np.array([[1,2,3], [4,5,6]], dtype=np.int32)
    b = np.array([[7,8], [9,10], [11, 12]], dtype=np.int32)
    c = a.dot(b)
    print(f"{a.shape} · {b.shape} = {c.shape}")
    print(c)

(2, 3) · (3, 2) = (2, 2)

\[\[ 58 64\]

\[139 154\]\]

Another frequent operation is reshaping tensors. For instance:


    x = np.array([[1,2,3,4,5,6]])
    print(x)
    print("x.shape:", x.shape)

    y = np.reshape(x, (2, 3))
    print("y:\n", y)
    print("y.shape:", y.shape)

    z = np.reshape(x, (2, -1))
    print("z:\n", z)
    print("z.shape:", z.shape)

\[\[1 2 3 4 5 6\]\]

x.shape: (1, 6)

y:

\[\[1 2 3\]

\[4 5 6\]\]

y.shape: (2, 3)

z:

\[\[1 2 3\]

\[4 5 6\]\]

z.shape: (2, 3)

Note that using -1 in reshape automatically infers the appropriate dimension based on the number of elements.

You can also add or remove dimensions of a tensor using np.expand_dims and np.squeeze:


    x = np.array([[1,2,3],[4,5,6]])
    print("x:\n", x)
    print("x.shape:", x.shape)

    y = np.expand_dims(x, 1)
    print("y:\n", y)
    print("y.shape:", y.shape)

x:

\[\[1 2 3\]

\[4 5 6\]\]

x.shape: (2, 3)

y:

\[\[\[1 2 3\]\]

\[\[4 5 6\]\]\]

y.shape: (2, 1, 3)

As you can see, a new level of brackets appears in the output due to the added dimension. Removing a dimension:


    x = np.array([[[1,2,3]],[[4,5,6]]])
    print("x:\n", x)
    print("x.shape:", x.shape)

    y = np.squeeze(x, 1)  # remove dimension 1
    print("y:\n", y)
    print("y.shape:", y.shape)

x:

\[\[\[1 2 3\]\]

\[\[4 5 6\]\]\]

x.shape: (2, 1, 3)

y:

\[\[1 2 3\]

\[4 5 6\]\]

y.shape: (2, 3)

#### Finding the Maximum Value in a Tensor

To conclude this review *appendix*, we present a function frequently used throughout the book: argmax. It returns the index of the maximum value along a given axis. If you're only interested in the value itself, use max.


    x = np.array([1,2,3,4])
    print(x)
    print(np.argmax(x))  # index of max
    print(np.max(x))     # value of max

\[1 2 3 4\]

3

4

If there are multiple elements with the same maximum value, argmax returns the first occurrence:


    x = np.array([1,2,4,4])
    print(x)
    print(np.argmax(x))
    print(np.max(x))

\[1 2 4 4\]

2

4

The same functions work on multidimensional arrays:


    x = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
    print(x)
    print(np.argmax(x))
    print(np.max(x))

\[\[ 1 2 3 4\]

\[ 5 6 7 8\]

\[ 9 10 11 12\]\]

11

12

In this case, index 11 corresponds to the last element of the array (12), as NumPy flattens the array when no axis is specified.

You can also specify the axis along which to search:


    print(np.argmax(x, axis=0))  # column-wise
    print(np.max(x, axis=0))

    print(np.argmax(x, axis=1))  # row-wise
    print(np.max(x, axis=1))

\[2 2 2 2\]

\[ 9 10 11 12\]

\[3 3 3\]

\[ 4 8 12\]

[^1]: https://docs.python.org
