from .glue import read_tsv, create_glue_dataset
from .multi_domain import create_single_domain_dataset, create_multi_domain_mnli_examples, \
    create_multi_domain_dataset

glue_tasks = ['mnli', 'mnli-mm', 'qqp', 'qnli', 'sst-2', 'cola', 'sts-b', 'mrpc', 'rte', 'wnli', 'ax']
squad_tasks = ['squad1.1', 'squad2.0']
multi_choice_tasks = ['swag', 'race']
multi_domain_tasks = ['mnli', 'amazon_review']
all_tasks = glue_tasks + squad_tasks + multi_choice_tasks + multi_domain_tasks

all_mnli_domains = ['fiction', 'government', 'slate', 'telephone', 'travel']
all_amazon_review_domains = ['books', 'dvd', 'electronics', 'kitchen']
mnli_domains_to_ids = {k: v for v, k in enumerate(all_mnli_domains)}
amazon_review_domains_to_ids = {k: v for v, k in enumerate(all_amazon_review_domains)}

glue_labels = {
    'mrpc': ['0', '1'],
    'mnli': ['contradiction', 'entailment', 'neutral'],
    'mnli-mm': ['contradiction', 'entailment', 'neutral'],
    'ax': ['contradiction', 'entailment', 'neutral'],
    'cola': ['0', '1'],
    'sst-2': ['0', '1'],
    'sts-b': [None],
    'qqp': ['0', '1'],
    'qnli': ['entailment', 'not_entailment'],
    'rte': ['entailment', 'not_entailment'],
    'wnli': ['0', '1']
}

glue_num_classes = {
    'mrpc': 2,
    'mnli': 3,
    'mnli-mm': 3,
    'ax': 3,
    'cola': 2,
    'sst-2': 2,
    'sts-b': 1,
    'qqp': 2,
    'qnli': 2,
    'rte': 2,
    'wnli': 2,
}

amazon_review_labels = {
    'positive',
    'negative',
}

amazon_review_split = {
    'books': [1631, 170, 199],
    'dvd': [1621, 194, 185],
    'electronics': [1615, 172, 213],
    'kitchen': [1613, 184, 203],
}

multi_domain_num_classes = {
    'mnli': 3,
    'amazon_review': 2,
}
