from datasets import load_dataset
from gliner import GLiNER
import torch
from trainer import GlinerTrainer

def merge_bi(ner_tags: list):
    # {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}
    ner_list = []
    start_idx = -1
    ner_tag = -1
    idx2text = {1 :'Person', 3: 'Organization', 5: 'Location', 7: 'Miscellaneous'}
    for i, _ in enumerate(ner_tags):
        if ner_tags[i] in [1, 3, 5, 7]:
            # handle the sequence 1, 1 or 3, 3
            if start_idx != -1:
                ner_list.append([ start_idx, i - 1, ner_tag ])
            start_idx = i
            ner_tag = idx2text[ner_tags[i]]
        # end of I-tag, add to ner list
        elif ner_tags[i] == 0 and start_idx != -1:
            ner_list.append([ start_idx, i - 1, ner_tag ])
            start_idx = -1
    # sentence ended on a I-tag
    if start_idx != -1:
        ner_list.append([ start_idx, len(ner_tags) - 1, ner_tag ])

    return ner_list

def convert_to_json(ds):
    json_data = []
    ner_list = list(map(lambda x: merge_bi(x), ds['ner_tags']))

    for i, token in enumerate(ds['tokens']):
        sample ={"tokenized_text" : token, "ner" : ner_list[i],
             "label": ['Person', 'Organization', 'Location', 'Miscellaneous']}
        json_data.append(sample)

    return json_data


def load_conll2003_dataset():
    ds = load_dataset("conll2003")
    assert ds.keys() == {'train', 'validation', 'test'}

    train_json = convert_to_json(ds['train'])
    val_json = convert_to_json(ds['validation'])
    test_json = convert_to_json(ds['test'])

    return train_json, val_json, test_json

def load_model(size):
    model = GLiNER.from_pretrained(f"urchade/gliner_{size}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model

if __name__ == "__main__":
    import sys
    action = sys.argv[1]
    size = sys.argv[2]
    train_json, val_json, test_json = load_conll2003_dataset()
    print(len(train_json))
    print(len(val_json))
    print(len(test_json))
    if action == "train":
        model = load_model(size)
        eval_data = {
        "entity_types": ["Person", 'Location', 'Organization', 'Miscellaneous'],
        "samples": val_json
        }
        trainer = GlinerTrainer(model,
                            train_data = train_json,
                            batch_size = 4,
                            grad_accum_every = 16,
                            lr_encoder = 1e-5,
                            lr_others = 5e-5,
                            freeze_token_rep = False,
                            val_every_step = 1000,
                            val_data = eval_data,
                            checkpoint_every_epoch = 15, # Or checkpoint_every_step if you use steps
                            max_types=25,
                )


        trainer.train(num_epochs=10)
        trainer.model.save_pretrained(size, repo_id=f"usernameandme/gliner-{size}", push_to_hub=True)
    else:
        print(len(test_json))
        eval_data = {
        "entity_types": ["Person", 'Location', 'Organization', 'Miscellaneous'],
        "samples": test_json
        }
        model = GLiNER.from_pretrained(size, local_files_only=True)
        results, f1 = model.evaluate(eval_data["samples"], flat_ner=True, threshold=0.5,
                                                        batch_size=12,
                                                        entity_types=eval_data["entity_types"])
        print(f"{results}")
