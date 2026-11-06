from transformers import AutoTokenizer
from datasets import load_dataset

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


def load_glue_dataset(config=None, **kwargs):
    model_name, _ = config.model.type.split('@')
    task_name, _ = config.data.type.split('@')

    # download the dataset.
    from federatedscope.glue.cenmag.offline import enabled, load_local_mnli
    if enabled(config):
        if task_name != 'mnli':
            raise ValueError('This offline asset manifest provides MNLI only')
        datasets = load_local_mnli(config)
    else:
        datasets = load_dataset("glue", task_name, cache_dir=config.data.root)

    # Labels
    is_regression = task_name == "stsb"
    if not is_regression:
        label_list = datasets["train"].features["label"].names
        num_labels = len(label_list)
        config.data.label_list = label_list
    else:
        num_labels = 1
    config.data.num_labels = num_labels

    # Preprocessing the datasets
    sentence1_key, sentence2_key = task_to_keys[task_name]

    # load tokenizer
    tokenizer_kwargs = {'cache_dir': config.llm.cache.model}
    if enabled(config):
        model_name = str(config.offline.model_dir)
        tokenizer_kwargs['local_files_only'] = True
    tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding='max_length', max_length=config.llm.tok_len, truncation=True)
        return result

    datasets = datasets.map(preprocess_function, batched=True, load_from_cache_file=True)
    datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    train_dataset = datasets["train"]
    if task_name == "mnli":
        eval_dataset = datasets["validation_matched" if config.data.matched else "validation_mismatched"]
        # test_dataset = datasets["test_matched" if config.data.matched else "test_mismatched"]
    else:
        eval_dataset = datasets["validation"]
        # test_dataset = datasets["test"]

    dataset = (train_dataset, eval_dataset, [])

    return dataset, config
