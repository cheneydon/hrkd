
def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def compute_multi_domain_metrics(task_name, preds, labels):
    if task_name == 'mnli':
        return {'acc': simple_accuracy(preds, labels)}
    elif task_name == 'amazon_review':
        return {'acc': simple_accuracy(preds, labels)}


all_multi_domain_select_metrics = {
    'mnli': 'acc',
    'amazon_review': 'acc',
}
