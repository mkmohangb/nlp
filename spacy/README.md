
Notes from the [Advanced NLP with Spacy](https://course.spacy.io/en) course
- The course is split into 4 chapters.
- Chapter 1
    - Starts with the basics of Documents, Spans & Tokens.
      ```
      import spacy
      nlp = spacy.blank("en")
      doc = nlp("Hello World!")
      for token in doc:
          print(token.text)
      span = doc[1:3]
      print(span.text)
      ```
        ![image](https://github.com/mkmohangb/nlp/assets/2610866/cf24d122-4413-42db-bf34-fb112bd39f5e)

    - Token attributes
      - ```token.i, token.text```
      -  lexical attributes( they refer to the entry in the vocabulary and don't depend on the token's context )
          ```token.is_alpha, token.is_punct, token.like_num```
