import re

class RegexReplacer(object):
	def __init__(self, replacement_patterns):
		self.patterns = replacement_patterns # list of tuples

	def replace(self, text):
		cur_text = text
		for (pattern, repl) in self.patterns:
			cur_replace_regex = re.compile(pattern, re.IGNORECASE)
			cur_text = cur_replace_regex.sub(repl, cur_text)
		return cur_text


"""
replacement_patterns = [
    (r"its", "it is"),
    (r"it's", "it is"),
    (r"you're", "your are"),
    (r"that's", "that is"),
    (r"'ve", " have"), #general
    (r"n't", "_not"), #general
    (r"'nt", "_not"), #general
    (r"'ll", " will"), #general
    (r"I'm", "i am"),
    (r"he's", "he is"), # also does she's
    (r"there's", "there is"),
    (r"'re", " are"),
    (r"let's", "let us"),
    (r"'d", " would"),
    (r" \.", "."),
    (r"here's", "here is"),
    (r"what's", "what is"),
    (r"some's", "someones"),
    (r"every's", "everyones"),
    (r"any's", "anyones"),
    (r"s'thing", "something"),
    (r"might'have", "might have"),
    (r"who's", "who is"),
    (r"is't" , "is it"),
    (r"did't", "did_not"),
    (r"\d", " "), # replace numbers with " ")
    (r"((?P<pre>\w+)([^a-zA-Z/_\s]+)(?P<suf>\w+))", r"\g<pre>\g<suf>"), # replace "all-star" with "allstar" but let "do_not" be "do_not"
    (r"(\w+)(\W+)(\w*)", r"\1 \3 "), # replace  "experience." with "experience" or "free****" with "free" or "horror/surviv" with "horror surviv"
    (r"(u\+)", ""), # replace rest of unicode characters
    (r"  ", " ")
]

regex_replacer.replace(sentence_entities) 
"""
