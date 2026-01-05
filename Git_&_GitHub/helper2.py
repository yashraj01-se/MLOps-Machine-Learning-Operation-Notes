## helper2 file
def greet(name):
    return f"Hello, {name}! Welcome to the Git & GitHub tutorial."
def farewell(name):
    return f"Goodbye, {name}! Happy coding with Git & GitHub."
def add(a, b):
    return a + b
def subtract(a, b):
    return a - b
def multiply(a, b):
    return a * b
def divide(a, b):
    if b == 0:
        return "Error: Division by zero is not allowed."
    return a / b
def power(base, exponent):
    return base ** exponent
def modulus(a, b):
    return a % b
def is_even(number):
    return number % 2 == 0
def is_odd(number):
    return number % 2 != 0
def factorial(n):
    if n < 0:
        return "Error: Factorial is not defined for negative numbers."
    if n == 0 or n == 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result
def fibonacci(n):
    if n <= 0:
        return "Error: Input should be a positive integer."
    fib_sequence = [0, 1]
    while len(fib_sequence) < n:
        next_value = fib_sequence[-1] + fib_sequence[-2]
        fib_sequence.append(next_value)
    return fib_sequence[:n]
def reverse_string(s):
    return s[::-1]
def is_palindrome(s):
    cleaned_s = ''.join(c.lower() for c in s if c.isalnum())
    return cleaned_s == cleaned_s[::-1]
def count_vowels(s):
    vowels = 'aeiouAEIOU'
    return sum(1 for char in s if char in vowels)
def count_consonants(s):
    vowels = 'aeiouAEIOU'
    return sum(1 for char in s if char.isalpha() and char not in vowels)

if __name__ == "__main__":
    # Test the functions
    print(greet("Alice"))
    print(farewell("Bob"))
    print("Addition:", add(5, 3))
    print("Subtraction:", subtract(10, 4))
    print("Multiplication:", multiply(7, 6))
    print("Division:", divide(8, 2))
    print("Power:", power(2, 3))
    print("Modulus:", modulus(10, 3))
    print("Is Even (4):", is_even(4))
    print("Is Odd (7):", is_odd(7))
    print("Factorial (5):", factorial(5))
    print("Fibonacci (7):", fibonacci(7))
    print("Reverse String ('hello'):", reverse_string("hello"))
    print("Is Palindrome ('Racecar'):", is_palindrome("Racecar"))
    print("Count Vowels ('Hello World'):", count_vowels("Hello World"))
    print("Count Consonants ('Hello World'):", count_consonants("Hello World"))

## helper2 --- IGNORE ---
## helper2 --- IGNORE ---
## helper2 --- IGNORE ---