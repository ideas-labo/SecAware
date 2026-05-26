EVAL_TEMPLATE = """
Write a complete, self-contained C program that can be compiled and run
in a standard user-space environment using gcc.
{task}
IMPORTANT CONSTRAINTS:
- This is NOT project code.
- Do NOT use external libraries beyond the C standard library.
- All structs, constants, and helper functions MUST be defined in the same file.
- The program MUST compile with gcc without warnings or errors.

Program Requirements:
- Provide a main() function.
- Read input from standard input (stdin).
- Write output to standard output (stdout).
- Output must be deterministic.

Output Rules:
- Do not include explanations, comments, or extra text.
- Only output valid C source code.
- If the code does not compile, the answer is considered incorrect.
"""