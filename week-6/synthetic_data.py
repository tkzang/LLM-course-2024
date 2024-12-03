from llama_index.llms.ollama import Ollama
import dspy

llm = Ollama(model="llama3", request_timeout=60.0)

# you can use DSPY (https://github.com/stanfordnlp/dspy), but you can also choose another method of interacting with an LLM
dspy.settings.configure(lm=llm)

# Task: implement a method, that will take a query string as input and produce N misspelling variants of the query.
# These variants with typos will be used to test a search engine quality.
# Example
# Query: machine learning applications
# Possible Misspellings:
# "machin learning applications" (missing "e" in "machine")
# "mashine learning applications" (phonetically similar spelling of "machine")
# "machine lerning aplications" (missing "a" in "learning" and "p" in "applications")
# "machin lerning aplications" (combining multiple typos)
# "mahcine learing aplication" (transposed letters in "machine" and typos in "learning" and "applications")
#
# Questions:
# 1. Does the search engine produce the same results for all the variants?
# 2. Do all variants make sense?
# 3. How to improve robustness of the method, for example, skip known abbreviations, like JFK or NBC.
# 4. Can you test multiple LLMs and figure out which one is the best?
# 5. Do the misspellings capture a variety of error types (phonetic, omission, transposition, repetition)?