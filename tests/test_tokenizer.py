from lib.preprocessing.tokenizer import tokenize
from lib.preprocessing.clean_text import clean_text

# text = "I ❤️ absolutely LOVE this <b>moovie</b>! Don't miss it! Check it out: https://example.com 10/10"
text = "Check out https://openai.com :) #AIrocks"
tokens = tokenize(text)
# cleaned_text = clean_text(text)
# tokens = tokenize(cleaned_text)
print(tokens)
