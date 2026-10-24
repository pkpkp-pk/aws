"""Buggy-function dataset for decider-guided patch search.

Each task: a small buggy function, a plain-assert test script that fails
on the buggy version and passes on the reference fix, and metadata.

Invariants (validated by tests/test_tasks.py):
- tests FAIL on `source` and PASS on `reference`
- every test script imports only from `solution` + stdlib

Splits: "dev" tasks tune the ranking formula and thresholds.
"heldout" tasks report results. Never tune on heldout.
"""
from __future__ import annotations

import textwrap
from dataclasses import dataclass


@dataclass(frozen=True)
class Task:
    name: str
    bug_class: str
    split: str  # "dev" | "heldout"
    source: str
    tests: str
    reference: str


def _t(name, bug_class, split, source, tests, reference):
    clean = lambda s: textwrap.dedent(s).strip() + "\n"
    return Task(name, bug_class, split, clean(source), clean(tests), clean(reference))


TASKS = [
    # ---------------- dev split (10) ----------------
    _t("dev_clamp", "wrong_variable", "dev",
       """
       def clamp(x, lo, hi):
           return min(max(x, hi), lo)
       """,
       """
       from solution import clamp
       assert clamp(5, 0, 10) == 5
       assert clamp(-3, 0, 10) == 0
       assert clamp(15, 0, 10) == 10
       assert clamp(7, 7, 7) == 7
       """,
       """
       def clamp(x, lo, hi):
           return min(max(x, lo), hi)
       """),
    _t("dev_count_vowels", "missing_edge_case", "dev",
       """
       def count_vowels(s):
           return sum(1 for c in s if c in "aeiou")
       """,
       """
       from solution import count_vowels
       assert count_vowels("hello") == 2
       assert count_vowels("AEIOU") == 5
       assert count_vowels("") == 0
       assert count_vowels("xyz") == 0
       """,
       """
       def count_vowels(s):
           return sum(1 for c in s.lower() if c in "aeiou")
       """),
    _t("dev_binary_search", "wrong_variable", "dev",
       """
       def binary_search(arr, target):
           lo, hi = 0, len(arr) - 1
           while lo <= hi:
               mid = (lo + hi) // 2
               if arr[mid] == target:
                   return mid
               if arr[mid] < target:
                   lo = mid + 1
               else:
                   hi = mid - 1
           return lo
       """,
       """
       from solution import binary_search
       assert binary_search([1, 3, 5, 7], 7) == 3
       assert binary_search([1, 3, 5, 7], 1) == 0
       assert binary_search([1, 3, 5, 7], 4) == -1
       assert binary_search([2], 2) == 0
       """,
       """
       def binary_search(arr, target):
           lo, hi = 0, len(arr) - 1
           while lo <= hi:
               mid = (lo + hi) // 2
               if arr[mid] == target:
                   return mid
               if arr[mid] < target:
                   lo = mid + 1
               else:
                   hi = mid - 1
           return -1
       """),
    _t("dev_is_palindrome", "wrong_operator", "dev",
       """
       def is_palindrome(s):
           return s == s[::1]
       """,
       """
       from solution import is_palindrome
       assert is_palindrome("racecar")
       assert not is_palindrome("hello")
       assert is_palindrome("")
       assert is_palindrome("a")
       """,
       """
       def is_palindrome(s):
           return s == s[::-1]
       """),
    _t("dev_fizzbuzz", "wrong_boundary", "dev",
       """
       def fizzbuzz(n):
           out = []
           for i in range(1, n):
               if i % 15 == 0:
                   out.append("FizzBuzz")
               elif i % 3 == 0:
                   out.append("Fizz")
               elif i % 5 == 0:
                   out.append("Buzz")
               else:
                   out.append(str(i))
           return out
       """,
       """
       from solution import fizzbuzz
       assert fizzbuzz(1) == ["1"]
       assert fizzbuzz(3) == ["1", "2", "Fizz"]
       assert fizzbuzz(5)[-1] == "Buzz"
       assert fizzbuzz(15)[-1] == "FizzBuzz"
       assert len(fizzbuzz(16)) == 16
       """,
       """
       def fizzbuzz(n):
           out = []
           for i in range(1, n + 1):
               if i % 15 == 0:
                   out.append("FizzBuzz")
               elif i % 3 == 0:
                   out.append("Fizz")
               elif i % 5 == 0:
                   out.append("Buzz")
               else:
                   out.append(str(i))
           return out
       """),
    _t("dev_max_subarray", "wrong_variable", "dev",
       """
       def max_subarray(nums):
           best = cur = 0
           for x in nums:
               cur = max(0, cur + x)
               best = max(best, cur)
           return best
       """,
       """
       from solution import max_subarray
       assert max_subarray([-2, 1, -3, 4, -1, 2, 1, -5, 4]) == 6
       assert max_subarray([-5, -1, -8]) == -1
       assert max_subarray([1]) == 1
       """,
       """
       def max_subarray(nums):
           best = cur = nums[0]
           for x in nums[1:]:
               cur = max(x, cur + x)
               best = max(best, cur)
           return best
       """),
    _t("dev_merge_sorted", "missing_edge_case", "dev",
       """
       def merge_sorted(a, b):
           out = []
           i = j = 0
           while i < len(a) and j < len(b):
               if a[i] <= b[j]:
                   out.append(a[i]); i += 1
               else:
                   out.append(b[j]); j += 1
           return out
       """,
       """
       from solution import merge_sorted
       assert merge_sorted([1, 3], [2, 4]) == [1, 2, 3, 4]
       assert merge_sorted([], [1]) == [1]
       assert merge_sorted([1, 1], [1]) == [1, 1, 1]
       assert merge_sorted([5], []) == [5]
       """,
       """
       def merge_sorted(a, b):
           out = []
           i = j = 0
           while i < len(a) and j < len(b):
               if a[i] <= b[j]:
                   out.append(a[i]); i += 1
               else:
                   out.append(b[j]); j += 1
           return out + a[i:] + b[j:]
       """),
    _t("dev_second_largest", "missing_edge_case", "dev",
       """
       def second_largest(nums):
           first = second = 0
           for x in nums:
               if x > first:
                   first, second = x, first
               elif x > second:
                   second = x
           return second
       """,
       """
       from solution import second_largest
       assert second_largest([1, 2, 3]) == 2
       assert second_largest([5, 1, 4, 2]) == 4
       assert second_largest([10, 9]) == 9
       assert second_largest([3, 3, 2]) == 2
       """,
       """
       def second_largest(nums):
           first = second = float("-inf")
           for x in nums:
               if x > first:
                   first, second = x, first
               elif first > x > second:
                   second = x
           return second
       """),
    _t("dev_reverse_words", "wrong_variable", "dev",
       """
       def reverse_words(sentence):
           return " ".join(sentence.split(" ")[::-1])[::-1]
       """,
       """
       from solution import reverse_words
       assert reverse_words("the sky is blue") == "blue is sky the"
       assert reverse_words("hello") == "hello"
       assert reverse_words("a b") == "b a"
       """,
       """
       def reverse_words(sentence):
           return " ".join(sentence.split(" ")[::-1])
       """),
    _t("dev_sum_digits", "wrong_boundary", "dev",
       """
       def sum_digits(n):
           total = 0
           while n > 0:
               total += n % 10
               n //= 10
           return total
       """,
       """
       from solution import sum_digits
       assert sum_digits(123) == 6
       assert sum_digits(0) == 0
       assert sum_digits(-45) == 9
       assert sum_digits(999) == 27
       """,
       """
       def sum_digits(n):
           total = 0
           for d in str(abs(n)):
               total += int(d)
           return total
       """),
    # ---------------- heldout split (20) ----------------
    _t("ho_power", "off_by_one", "heldout",
       """
       def power(base, exp):
           result = 1
           for _ in range(exp + 1):
               result *= base
           return result
       """,
       """
       from solution import power
       assert power(2, 10) == 1024
       assert power(3, 0) == 1
       assert power(5, 1) == 5
       assert power(2, 3) == 8
       """,
       """
       def power(base, exp):
           result = 1
           for _ in range(exp):
               result *= base
           return result
       """),
    _t("ho_unique_sorted", "wrong_operator", "heldout",
       """
       def unique_sorted(nums):
           out = []
           for x in nums:
               if x not in out:
                   out.append(x)
           return out
       """,
       """
       from solution import unique_sorted
       assert unique_sorted([3, 1, 2, 1]) == [1, 2, 3]
       assert unique_sorted([]) == []
       assert unique_sorted([7]) == [7]
       assert unique_sorted([2, 2]) == [2]
       """,
       """
       def unique_sorted(nums):
           out = []
           for x in sorted(nums):
               if not out or out[-1] != x:
                   out.append(x)
           return out
       """),
    _t("ho_title_case", "missing_edge_case", "heldout",
       """
       def title_case(s):
           return s.title()
       """,
       """
       from solution import title_case
       assert title_case("hello world") == "Hello World"
       assert title_case("they're bill's") == "They're Bill's"
       assert title_case("") == ""
       """,
       """
       def title_case(s):
           import string
           return string.capwords(s)
       """),
    _t("ho_is_anagram", "wrong_operator", "heldout",
       """
       def is_anagram(a, b):
           return sorted(a) == sorted(b) and len(a) > 0
       """,
       """
       from solution import is_anagram
       assert is_anagram("listen", "silent")
       assert not is_anagram("hello", "world")
       assert is_anagram("ab", "ba")
       assert is_anagram("Abc", "bca")
       """,
       """
       def is_anagram(a, b):
           return sorted(a.lower()) == sorted(b.lower()) and len(a) > 0
       """),
    _t("ho_count_primes", "off_by_one", "heldout",
       """
       def count_primes(n):
           def is_prime(k):
               if k < 2:
                   return False
               for i in range(2, int(k ** 0.5)):
                   if k % i == 0:
                       return False
               return True
           return sum(1 for k in range(2, n) if is_prime(k))
       """,
       """
       from solution import count_primes
       assert count_primes(10) == 4
       assert count_primes(2) == 0
       assert count_primes(3) == 1
       assert count_primes(20) == 8
       """,
       """
       def count_primes(n):
           def is_prime(k):
               if k < 2:
                   return False
               for i in range(2, int(k ** 0.5) + 1):
                   if k % i == 0:
                       return False
               return True
           return sum(1 for k in range(2, n) if is_prime(k))
       """),
    _t("ho_flatten", "wrong_operator", "heldout",
       """
       def flatten(nested):
           out = []
           for sub in nested:
               out.append(sub)
           return out
       """,
       """
       from solution import flatten
       assert flatten([[1, 2], [3]]) == [1, 2, 3]
       assert flatten([]) == []
       assert flatten([[], [1]]) == [1]
       assert flatten([[1], []]) == [1]
       """,
       """
       def flatten(nested):
           out = []
           for sub in nested:
               out.extend(sub)
           return out
       """),
    _t("ho_trailing_zeros", "wrong_boundary", "heldout",
       """
       def trailing_zeros(n):
           count = 0
           while n > 0 and n % 10 == 0:
               count += 1
               n //= 10
           return count
       """,
       """
       from solution import trailing_zeros
       assert trailing_zeros(100) == 2
       assert trailing_zeros(7) == 0
       assert trailing_zeros(0) == 1
       assert trailing_zeros(10203000) == 3
       """,
       """
       def trailing_zeros(n):
           if n == 0:
               return 1
           count = 0
           while n % 10 == 0:
               count += 1
               n //= 10
           return count
       """),
    _t("ho_rotate", "missing_edge_case", "heldout",
       """
       def rotate(lst, k):
           if not lst:
               return lst
           return lst[-k:] + lst[:-k] if k else lst
       """,
       """
       from solution import rotate
       assert rotate([1, 2, 3, 4, 5], 2) == [4, 5, 1, 2, 3]
       assert rotate([1, 2, 3], 4) == [3, 1, 2]
       assert rotate([], 5) == []
       assert rotate([1, 2, 3], -1) == [2, 3, 1]
       """,
       """
       def rotate(lst, k):
           if not lst:
               return lst
           k %= len(lst)
           return lst[-k:] + lst[:-k] if k else lst
       """),
    _t("ho_longest_run", "missing_edge_case", "heldout",
       """
       def longest_run(s):
           best = cur = 0
           prev = None
           for c in s:
               if c == prev:
                   cur += 1
               else:
                   best = max(best, cur)
                   cur = 1
                   prev = c
           return best
       """,
       """
       from solution import longest_run
       assert longest_run("aaabbbbc") == 4
       assert longest_run("") == 0
       assert longest_run("abc") == 1
       assert longest_run("aabb") == 2
       assert longest_run("aaaa") == 4
       """,
       """
       def longest_run(s):
           best = cur = 0
           prev = None
           for c in s:
               if c == prev:
                   cur += 1
               else:
                   cur = 1
                   prev = c
               best = max(best, cur)
           return best
       """),
    _t("ho_chunk", "off_by_one", "heldout",
       """
       def chunk(lst, size):
           return [lst[i:i + size - 1] for i in range(0, len(lst), size)]
       """,
       """
       from solution import chunk
       assert chunk([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]
       assert chunk([1, 2, 3], 3) == [[1, 2, 3]]
       assert chunk([], 2) == []
       assert chunk([1, 2, 3, 4], 1) == [[1], [2], [3], [4]]
       """,
       """
       def chunk(lst, size):
           return [lst[i:i + size] for i in range(0, len(lst), size)]
       """),
    _t("ho_median", "missing_edge_case", "heldout",
       """
       def median(nums):
           s = sorted(nums)
           n = len(s)
           return s[n // 2]
       """,
       """
       from solution import median
       assert median([3, 1, 2]) == 2
       assert median([4, 1, 3, 2]) == 2.5
       assert median([7]) == 7
       """,
       """
       def median(nums):
           s = sorted(nums)
           n = len(s)
           mid = n // 2
           if n % 2 == 1:
               return s[mid]
           return (s[mid - 1] + s[mid]) / 2
       """),
    _t("ho_gcd", "wrong_variable", "heldout",
       """
       def gcd(a, b):
           while b:
               a, b = a, a % b
           return abs(a)
       """,
       """
       from solution import gcd
       assert gcd(12, 18) == 6
       assert gcd(7, 0) == 7
       assert gcd(-12, 8) == 4
       assert gcd(17, 5) == 1
       """,
       """
       def gcd(a, b):
           while b:
               a, b = b, a % b
           return abs(a)
       """),
    _t("ho_remove_duplicates", "wrong_operator", "heldout",
       """
       def remove_duplicates(s):
           seen = set()
           out = []
           for c in s:
               if c not in seen:
                   seen.add(c)
                   out.append(c)
           return "".join(sorted(out))
       """,
       """
       from solution import remove_duplicates
       assert remove_duplicates("banana") == "ban"
       assert remove_duplicates("") == ""
       assert remove_duplicates("aabbcc") == "abc"
       assert remove_duplicates("cba") == "cba"
       """,
       """
       def remove_duplicates(s):
           seen = set()
           out = []
           for c in s:
               if c not in seen:
                   seen.add(c)
                   out.append(c)
           return "".join(out)
       """),
    _t("ho_is_sorted", "wrong_boundary", "heldout",
       """
       def is_sorted(lst):
           return all(lst[i] < lst[i + 1] for i in range(len(lst) - 1))
       """,
       """
       from solution import is_sorted
       assert is_sorted([1, 2, 3])
       assert not is_sorted([3, 1])
       assert is_sorted([])
       assert is_sorted([1, 1])
       assert is_sorted([5])
       """,
       """
       def is_sorted(lst):
           return all(lst[i] <= lst[i + 1] for i in range(len(lst) - 1))
       """),
    _t("ho_count_words", "missing_edge_case", "heldout",
       """
       def count_words(s):
           return len(s.split(" "))
       """,
       """
       from solution import count_words
       assert count_words("the quick fox") == 3
       assert count_words("") == 0
       assert count_words("  spaced   out ") == 2
       assert count_words("one") == 1
       """,
       """
       def count_words(s):
           return len(s.split())
       """),
    _t("ho_caesar", "wrong_variable", "heldout",
       """
       def caesar(text, shift):
           out = []
           for c in text:
               if c.isalpha():
                   base = ord("A") if c.isupper() else ord("a")
                   out.append(chr((ord(c) - base - shift) % 26 + base))
               else:
                   out.append(c)
           return "".join(out)
       """,
       """
       from solution import caesar
       assert caesar("abc", 1) == "bcd"
       assert caesar("xyz", 3) == "abc"
       assert caesar("Hello, World!", 5) == "Mjqqt, Btwqi!"
       assert caesar("a", 0) == "a"
       """,
       """
       def caesar(text, shift):
           out = []
           for c in text:
               if c.isalpha():
                   base = ord("A") if c.isupper() else ord("a")
                   out.append(chr((ord(c) - base + shift) % 26 + base))
               else:
                   out.append(c)
           return "".join(out)
       """),
    _t("ho_rle_encode", "off_by_one", "heldout",
       """
       def rle_encode(s):
           if not s:
               return ""
           out = []
           count = 1
           for i in range(1, len(s) + 1):
               if i < len(s) and s[i] == s[i - 1]:
                   count += 1
               else:
                   out.append(f"{s[i - 1]}{count}")
                   count = 0
           return "".join(out)
       """,
       """
       from solution import rle_encode
       assert rle_encode("aaab") == "a3b1"
       assert rle_encode("") == ""
       assert rle_encode("abc") == "a1b1c1"
       assert rle_encode("xxxx") == "x4"
       """,
       """
       def rle_encode(s):
           if not s:
               return ""
           out = []
           count = 1
           for i in range(1, len(s) + 1):
               if i < len(s) and s[i] == s[i - 1]:
                   count += 1
               else:
                   out.append(f"{s[i - 1]}{count}")
                   count = 1
           return "".join(out)
       """),
    _t("ho_interleave", "missing_edge_case", "heldout",
       """
       def interleave(a, b):
           out = []
           for i in range(min(len(a), len(b))):
               out.append(a[i])
               out.append(b[i])
           return out
       """,
       """
       from solution import interleave
       assert interleave([1, 2], ["a", "b"]) == [1, "a", 2, "b"]
       assert interleave([1, 2, 3], ["a"]) == [1, "a", 2, 3]
       assert interleave([], [1]) == [1]
       assert interleave([1], []) == [1]
       """,
       """
       def interleave(a, b):
           out = []
           for i in range(max(len(a), len(b))):
               if i < len(a):
                   out.append(a[i])
               if i < len(b):
                   out.append(b[i])
           return out
       """),
    _t("ho_valid_parens", "wrong_operator", "heldout",
       """
       def valid_parens(s):
           depth = 0
           for c in s:
               if c == "(":
                   depth += 1
               elif c == ")":
                   depth -= 1
           return depth == 0
       """,
       """
       from solution import valid_parens
       assert valid_parens("()()")
       assert valid_parens("(())")
       assert not valid_parens(")(")
       assert not valid_parens("(()")
       assert valid_parens("")
       """,
       """
       def valid_parens(s):
           depth = 0
           for c in s:
               if c == "(":
                   depth += 1
               elif c == ")":
                   depth -= 1
                   if depth < 0:
                       return False
           return depth == 0
       """),
    _t("ho_sum_range", "wrong_boundary", "heldout",
       """
       def sum_range(nums, i, j):
           return sum(nums[i:j + 1]) if j >= i else 0
       """,
       """
       from solution import sum_range
       assert sum_range([1, 2, 3, 4], 0, 2) == 3
       assert sum_range([1, 2, 3, 4], 1, 4) == 9
       assert sum_range([5], 0, 1) == 5
       assert sum_range([1, 2, 3], 2, 2) == 0
       """,
       """
       def sum_range(nums, i, j):
           return sum(nums[i:j])
       """),
]


def load_tasks(split: str | None = None) -> list[Task]:
    """All tasks, or only one split ("dev" / "heldout")."""
    if split is None:
        return list(TASKS)
    return [t for t in TASKS if t.split == split]
