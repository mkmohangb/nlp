
Notes from the [Advanced NLP with Spacy](https://course.spacy.io/en) course
- The course is split into 4 chapters.
- **Chapter 1**
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

    - Trained pipelines
      - predict linguistic attributes in context like part of speech tags, syntactic dependencies, named entities.
      - pipeline package includes binary weights, vocabulary, meta information & config file. 
      - e.g. ```en_core_web_sm``` - small english pipeline that supports all core capabilities and is trained on web text.
      - ```python -m spacy download en_core_web_sm```
      - ```nlp = spacy.load("en_core_web_sm")```
      - ```token.pos_, token.dep_, token.head.text```
      - ```spacy.explain('nsubj')```
      - ```doc.ents``` let's you access the named entities predicted by the NER model - iterator of Span objects.
      - ```for ent in doc.ents: print(ent.text, ent.label_)```
     
    - Rule based matching
      - match patterns - list of dictionaries, one per token.
        ```
            from spacy.matcher import Matcher
            matcher = Matcher(nlp.vocab)
            pattern = [{"TEXT": "iPhone"}, {"TEXT": "X"}]
            matcher.add("IPHONE_PATTERN", [pattern])
            doc = nlp("Upcoming iPhone X release date leaked")
            matches = matcher(doc)
            for match_id, start, end in matches:
                matched_span = doc[start:end]
                print(matched_span.text)
        ```
      - {"OP": "!"} - negation: match 0 times, "?" - 0 or 1 time, "+" - 1 or more times, "*" = 0 or more times
