# 5-shot
standard_prompt = '''Use numbers and basic arithmetic operations (+ - * /) to obtain 24.
Input: 4 4 6 8
Answer: (4 + 8) * (6 - 4) = 24
Input: 2 9 10 12
Answer: 2 * 12 * (10 - 9) = 24
Input: 4 9 10 13
Answer: (13 - 9) * (10 - 4) = 24
Input: 1 4 8 8
Answer: (8 / 4 + 1) * 8 = 24
Input: 5 5 5 9
Answer: 5 + 5 + 5 + 9 = 24
Input: {input}
Answer:'''

# 5-shot
output_prompt = '''Use numbers and basic arithmetic operations (+ - * /) to obtain 24. Each step, you are only allowed to choose two of the remaining numbers to obtain a new number.
Input: 4 4 6 8
Steps:
4 + 8 = 12 (left: 4 6 12)
6 - 4 = 2 (left: 2 12)
2 * 12 = 24 (left: 24)
Answer: (6 - 4) * (4 + 8) = 24

Input: 2 9 10 12
Steps:
12 * 2 = 24 (left: 9 10 24)
10 - 9 = 1 (left: 1 24)
24 * 1 = 24 (left: 24)
Answer: (12 * 2) * (10 - 9) = 24

Input: 4 9 10 13
Steps:
13 - 10 = 3 (left: 3 4 9)
9 - 3 = 6 (left: 4 6)
4 * 6 = 24 (left: 24)
Answer: 4 * (9 - (13 - 10)) = 24

Input: 1 4 8 8
Steps:
8 / 4 = 2 (left: 1 2 8)
1 + 2 = 3 (left: 3 8)
3 * 8 = 24 (left: 24)
Answer: (1 + 8 / 4) * 8 = 24

Input: 5 5 5 9
Steps:
5 + 5 = 10 (left: 5 9 10)
10 + 5 = 15 (left: 9 15)
15 + 9 = 24 (left: 24)
Answer: ((5 + 5) + 5) + 9 = 24

Input: {input}
{history}
Answer:'''

# 1-shot
propose_prompt = '''Let me consider how to systematically enumerate all possible operations for a given set of numbers to reach the number 24 (so that these numbers sum up to 24).

I want to explore all ways of combining the given numbers using '+', '-', '*', '/'. Continue doing so recursively until I either find a sequence of operations that results in exactly 24, referred to as optimal solution.

Step-by-step method:

- Initial enumeration: Start with the given input numbers. At the first step, list all possible operations between every pair of numbers. For each pair (a, b) of numbers, consider all operations:
    b - a 
    a * b
    a / b (if b ≠ 0) 
    b / a (if a ≠ 0)
    a + b
    a - b 

   For each resulting value, show the new set of numbers (the result replaces the two chosen numbers) and record the operation.


- Recursive exploration: After performing one operation, I now have fewer numbers. Repeat the process with the new set of numbers, continuing until:
    - I end up with a single number. If that number is 24, I have found a solution. 
    - Or I have tried all possible sequences of operations without reaching 24.

- Format of steps:
    - Each step should show the operations performed and the resulting set of numbers. For example: operand1 operator operand2 = result (left: remaining numbers)
    - Continue to show steps until I reach a single number. If that single number is 24, I will print out the entire sequence of steps that led there.

For example, if my input was:

Input: 1 2 3

List all possible operations between every pair of numbers using '+', '-', '*', '/'.
For each operation, show the result and the remaining numbers. Use the format:
operand1 operator operand2 = result (left: remaining numbers)

Possible operations:
1 + 2 = 3 (left: 3 3)
1 - 2 = -1 (left: -1 3)
1 * 2 = 2 (left: 2 3)
1 / 2 = 0.5 (left: 0.5 3)
1 + 3 = 4 (left: 2 4)
1 - 3 = -2 (left: 2 -2)
1 * 3 = 3 (left: 2 3)
1 / 3 = 0.3333 (left: 2 0.3333)
2 + 1 = 3 (left: 3 3)
2 - 1 = 1 (left: 1 3)
2 * 1 = 2 (left: 3 2)
2 / 1 = 2 (left: 3 2)
2 + 3 = 5 (left: 1 5)
2 - 3 = -1 (left: 1 -1)
2 * 3 = 6 (left: 1 6)
2 / 3 = 0.6667 (left: 1 0.6667)
3 + 1 = 4 (left: 2 4)
3 - 1 = 2 (left: 2 2)
3 * 1 = 3 (left: 2 3)
3 / 1 = 3 (left: 2 3)
3 + 2 = 5 (left: 1 5)
3 - 2 = 1 (left: 1 1)
3 * 2 = 6 (left: 1 6)
3 / 2 = 1.5 (left: 1 1.5)

- Example leading to 24 (optimal solution is the way I am supposed to follow):
Input: 4 4 6 8
Steps:
4 + 8 = 12 (left: 4 6 12)
6 - 4 = 2 (left: 2 12)
2 * 12 = 24 (left: 24)

- Example of optimal solution (The way I am supposed to follow):
Input: 2 9 10 12
Steps:
12 * 2 = 24 (left: 9 10 24)
10 - 9 = 1 (left: 1 24)
24 * 1 = 24 (left: 24)

- Example of optimal solution:
Input: 2 3 3 8
Steps:
8 / 2 = 4 (left: 3 3 4)
3 + 3 = 6 (left: 4 6)
6 * 4 = 24 (left: 24)

- Example of optimal solution:
Input: 2 3 4 6
Steps:
3 * 4 = 12 (left: 2 6 12)
2 * 6 = 12 (left: 12 12)
12 + 12 = 24 (left: 24)

- Example of optimal solution:
Input: 2 4 4 6
Steps:
4 + 4 = 8 (left: 2 6 8)
8 * 6 = 48 (left: 2 48)
48 / 2 = 24 (left: 24)

- Example of optimal solution:
Input: 3, 5, 5, 6
Steps:
5 + 5 = 10 (left: 3, 6, 10)
10 * 3 = 30 (left: 6, 30)
30 - 6 = 24 (left: 24)


These examples show the desired level of detail and the approach to listing operations.

Important:
- I will not add any prefixes like "1.", "-" or other numbering before the operations.
- I will try not to have bias towards a specific operator while doing the operations. I will try to explore diverse solutions using diverse operators.

Now, use these steps to find optimal solution for:

Input: {input}
Steps:
'''

value_prompt = '''Evaluate if given numbers can reach 24 (sure/likely/impossible)
10 14
10 + 14 = 24
sure

11 12
11 + 12 = 23
12 - 11 = 1
11 * 12 = 132
11 / 12 = 0.91

impossible
4 4 10
4 + 4 + 10 = 8 + 10 = 18
4 * 10 - 4 = 40 - 4 = 36
(10 - 4) * 4 = 6 * 4 = 24
sure

4 9 11
9 + 11 + 4 = 20 + 4 = 24
sure

5 7 8
5 + 7 + 8 = 12 + 8 = 20
(8 - 5) * 7 = 3 * 7 = 21
I cannot obtain 24 now, but numbers are within a reasonable range
likely

5 6 6
5 + 6 + 6 = 17
(6 - 5) * 6 = 1 * 6 = 6
I cannot obtain 24 now, but numbers are within a reasonable range
likely

10 10 11
10 + 10 + 11 = 31
(11 - 10) * 10 = 10
10 10 10 are all too big
impossible

1 3 3
1 * 3 * 3 = 9
(1 + 3) * 3 = 12
1 3 3 are all too small
impossible

{input}
'''

value_last_step_prompt = '''Use numbers and basic arithmetic operations (+ - * /) to obtain 24. Given an input and an answer, give a judgement (sure/impossible) if the answer is correct, i.e. it uses each input exactly once and no other numbers, and reach 24.
Input: 4 4 6 8
Answer: (4 + 8) * (6 - 4) = 24
Judge: sure

Input: 2 9 10 12
Answer: 2 * 12 * (10 - 9) = 24
Judge: sure

Input: 4 9 10 13
Answer: (13 - 9) * (10 - 4) = 24
Judge: sure

Input: 4 4 6 8
Answer: (4 + 8) * (6 - 4) + 1 = 25
Judge: impossible

Input: 2 9 10 12
Answer: 2 * (12 - 10) = 24
Judge: impossible

Input: 4 9 10 13
Answer: (13 - 4) * (10 - 9) = 24
Judge: impossible

Input: {input}
Answer: {answer}
Judge:'''

value_name = ['sure', 'likely', 'impossible']
value_map = {'sure': 1, 'likely': 0.1, 'impossible': 0.0001}
