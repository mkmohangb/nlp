Fine-tune GLiNER model

```python train_gliner.py train small```

**[18-May-24]** Trained for 10 epochs and evaluated on test set of CoNLL2003. Metric is F1 score.
- small  -  92.44%
- medium -  93.06%
- large  -  93.14% (5 epochs)

```
/workspace/nlp/ner# du -sh small medium large
583M    small
745M    medium
1.7G    large
```
