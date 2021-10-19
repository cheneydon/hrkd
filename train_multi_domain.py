import os
import numpy as np
import torch
import torch.nn as nn
import logging
import argparse
import time
import datetime
from apex import amp
from apex.parallel import DistributedDataParallel as DDP
from tensorboardX import SummaryWriter
from models import select_config, select_model, MetaGraph
from tokenizers import select_tokenizer
from datasets import mnli_domains_to_ids, amazon_review_domains_to_ids, all_mnli_domains, \
    all_amazon_review_domains
from metrics import all_multi_domain_select_metrics, compute_multi_domain_metrics
from utils import AverageMeter, set_seeds, setup_logger, calc_params, reduce_tensor, save_checkpoint, \
    load_pretrain_state_dict, load_resume_state_dict, create_optimizer, create_scheduler, \
    create_single_domain_dataset, create_multi_domain_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--distributed', action='store_true', help='distributed mode')
parser.add_argument('--fp16', action='store_true', help='mixed precision training mode')
parser.add_argument('--opt_level', default='O1', type=str, help='fp16 optimization level')
parser.add_argument('--gpu_devices', default='4,5,6,7', type=str, help='available gpu devices')
parser.add_argument('--seed', default=42, type=int, help='seed')

parser.add_argument('--lowercase', action='store_true', help='whether to do lowercase')
parser.add_argument('--do_test', action='store_true', help='whether to do test')
parser.add_argument('--use_graph', action='store_true', help='whether to use graph')
parser.add_argument('--hierarchical', action='store_true', help='whether to use hierarchical graph')
parser.add_argument('--model_name', default='bert_base', type=str, help='model name')
parser.add_argument('--task', default='mnli', type=str, help='task name')
parser.add_argument('--domain', default='telephone', type=str, help='domain name')
parser.add_argument('--data_dir', default='', type=str, help='task dataset directory')
parser.add_argument('--vocab_path', default='', type=str, help='path to pretrained vocabulary file')
parser.add_argument('--merge_path', default='', type=str, help='path to pretrained merge file (for roberta)')
parser.add_argument('--max_seq_len', default=128, type=int, help='max length of input sequences')
parser.add_argument('--max_query_len', default=64, type=int, help='max length of input questions (for squad) or question-answer pairs (for multi-choice tasks)')
parser.add_argument('--train_ratio', default=0.9, type=float, help='train ratio for MNLI')
parser.add_argument('--dev_ratio', default=0.1, type=float, help='dev ratio for MNLI')

parser.add_argument('--start_epoch', default=1, type=int, help='start epoch (default is 1)')
parser.add_argument('--total_epochs', default=3, type=int, help='total epochs')
parser.add_argument('--batch_size', default=8, type=int, help='batch size for each domain')
parser.add_argument('--lr', default=3e-5, type=float, help='initial learning rate')
parser.add_argument('--optim_type', default='adamw', type=str, help='optimizer type')
parser.add_argument('--sched_type', default='step', type=str, help='lr scheduler type')
parser.add_argument('--warmup_proportion', default=0.1, type=float, help='proportion of warmup steps')
parser.add_argument('--momentum', default=0.9, type=float, help='SGD momentum')
parser.add_argument('--weight_decay', default=0.01, type=float, help='weight decay')
parser.add_argument('--max_grad_norm', default=1.0, type=float, help='max gradient norm')
parser.add_argument('--disp_freq', default=50, type=int, help='display step frequency')
parser.add_argument('--val_freq', default=50, type=int, help='validate step frequency')
parser.add_argument('--ckpt_keep_num', default=1, type=int, help='max number of checkpoint files to keep')
parser.add_argument('--num_workers', default=4, type=int, help='num workers of dataloader')

parser.add_argument('--pretrain_path', default='', type=str, help='path to pretrained state dict')
parser.add_argument('--resume_path', default='', type=str, help='path to resume checkpoint')
parser.add_argument('--cache_dir', default='./cache', type=str, help='cache directory to save processed dataset')
parser.add_argument('--exp_dir', default='./exp/tmp', type=str, help='experiment directory')
parser.add_argument('--local_rank', default=0, type=int, help='DDP local rank')
parser.add_argument('--world_size', default=1, type=int, help='DDP world size')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices


def main():
    args.exp_dir = os.path.join(args.exp_dir, datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    setup_logger(args.exp_dir)
    if args.local_rank == 0:
        logging.info(args)
        args.writer = SummaryWriter(log_dir=args.exp_dir)  # tensorboard

    args.use_gpu = False
    if args.gpu_devices and torch.cuda.is_available():
        args.use_gpu = True
    if args.use_gpu and args.local_rank == 0:
        logging.info('Currently using GPU: {}'.format(args.gpu_devices))
    elif not args.use_gpu and args.local_rank == 0:
        logging.info('Currently using CPU')
    set_seeds(args.seed, args.use_gpu)

    if args.use_gpu and args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        logging.info('Training in distributed mode (process {}/{})'.format(args.local_rank + 1, args.world_size))

    # Load model and tokenizer
    args.config = select_config(args.model_name, args.lowercase)
    model = select_model(args.model_name, args.lowercase, args.task, return_hid=args.use_graph)
    args.tokenizer = select_tokenizer(
        args.model_name, args.lowercase, 'mnli', args.vocab_path, args.max_seq_len, args.max_query_len)
    if args.use_graph:
        num_domains = len(all_mnli_domains) if args.task == 'mnli' else len(all_amazon_review_domains)
        args.meta_graph = MetaGraph(args.config.hidden_size, num_layers=1, num_domains=num_domains, hierarchical=args.hierarchical)

    if args.use_gpu:
        model = model.cuda()
        if args.use_graph:
            args.meta_graph = args.meta_graph.cuda()
    if args.local_rank == 0:
        logging.info('Model size: {:.2f}M'.format(calc_params(model) / 1e6))

    if args.task == 'mnli':
        args.all_domains = all_mnli_domains
        args.domains_to_ids = mnli_domains_to_ids
        fit_domain_id = mnli_domains_to_ids[args.domain]
    elif args.task == 'amazon_review':
        args.all_domains = all_amazon_review_domains
        args.domains_to_ids = amazon_review_domains_to_ids
        fit_domain_id = amazon_review_domains_to_ids[args.domain]

    # Create dataset
    if args.do_test:
        if args.task == 'amazon_review':
            _, _, _, all_test_loaders = create_multi_domain_dataset(
                args.model_name, args.task, None, args.data_dir, args.tokenizer, args.max_seq_len,
                args.batch_size * 4, args.use_gpu, args.distributed, 'test', args.local_rank, args.cache_dir,
                args.train_ratio, args.dev_ratio, args.num_workers)
            test_loader = all_test_loaders[args.domains_to_ids[args.domain]]
        else:
            _, _, _, test_loader = create_single_domain_dataset(
                args.model_name, args.task, args.domain, args.data_dir, args.tokenizer, args.max_seq_len,
                args.batch_size * 4, args.use_gpu, args.distributed, 'test', args.local_rank, args.cache_dir,
                args.train_ratio, args.dev_ratio, args.num_workers)
        optimizer = None
        scheduler = None
    else:
        _, _, _, train_loader = create_multi_domain_dataset(
            args.model_name, args.task, fit_domain_id, args.data_dir, args.tokenizer, args.max_seq_len,
            args.batch_size, args.use_gpu, args.distributed, 'train', args.local_rank, args.cache_dir,
            args.train_ratio, args.dev_ratio, args.num_workers)
        _, _, _, all_dev_loaders = create_multi_domain_dataset(
            args.model_name, args.task, None, args.data_dir, args.tokenizer, args.max_seq_len,
            args.batch_size * 4, args.use_gpu, args.distributed, 'dev', args.local_rank, args.cache_dir,
            args.train_ratio, args.dev_ratio, args.num_workers)

        # Create optimization tools
        num_sched_steps = len(train_loader) * args.total_epochs
        num_warmup_steps = int(num_sched_steps * args.warmup_proportion)
        optimizer = create_optimizer(model, args.optim_type, args.lr, args.weight_decay, args.momentum)
        criterion = nn.CrossEntropyLoss()
        scheduler = create_scheduler(optimizer, args.sched_type, num_sched_steps, num_warmup_steps)
        if args.use_gpu:
            criterion = criterion.cuda()
        optim_tools = [optimizer, criterion, scheduler]

    if args.use_gpu:
        if args.fp16:
            amp.register_half_function(torch, 'einsum')
            if optimizer is not None:
                model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level)
            if args.local_rank == 0:
                logging.info('Using fp16 training mode')
        if args.distributed:
            model = DDP(model, delay_allreduce=True)
            if args.use_graph:
                args.meta_graph = DDP(args.meta_graph, delay_allreduce=True)
        else:
            model = nn.DataParallel(model)
            if args.use_graph:
                args.meta_graph = nn.DataParallel(args.meta_graph)

    if args.pretrain_path:
        if os.path.exists(args.pretrain_path):
            load_pretrain_state_dict(model, args.pretrain_path, args.use_gpu)
            if args.local_rank == 0:
                logging.info('Loaded pretrained state dict from \'{}\''.format(args.pretrain_path))
        else:
            if args.local_rank == 0:
                logging.info('No pretrained state dict found in \'{}\''.format(args.pretrain_path))

    if args.resume_path:
        if os.path.exists(args.resume_path):
            checkpoint = load_resume_state_dict(model, args.resume_path, args.use_gpu, optimizer, scheduler)
            args.start_epoch = checkpoint['epoch'] + 1
            if args.local_rank == 0:
                logging.info('Loaded checkpoint from \'{}\''.format(args.resume_path))
                logging.info('Start epoch: {}\tMetrics: {}'.format(args.start_epoch, checkpoint['metrics']))
        else:
            if args.local_rank == 0:
                logging.info('No checkpoint found in \'{}\''.format(args.resume_path))

    try:
        if args.do_test:
            validate(model, test_loader, verbose=True)
        else:
            train(model, optim_tools, train_loader, all_dev_loaders)
    except KeyboardInterrupt:
        print('Keyboard interrupt (process {}/{})'.format(args.local_rank + 1, args.world_size))


def train(model, optim_tools, train_loader, all_dev_loaders):
    st_time = time.time()
    if args.local_rank == 0:
        logging.info('==> Start training')

    best_results = {
        'best_score': 0,
        'best_sel_metrics': None,
        'best_metrics': None,
        'best_epoch': None,
        'best_step': None,
    }
    best_domain_scores = {k: 0 for k in args.all_domains}
    for epoch in range(args.start_epoch, args.total_epochs + 1):
        train_epoch(model, epoch, optim_tools, train_loader, all_dev_loaders, best_results, best_domain_scores)

        if args.local_rank == 0:
            logging.info('-' * 50)
            state = {'state_dict': model.state_dict(),
                     'epoch': epoch}
            ckpt_name = 'ckpt_ep' + str(epoch) + '.bin'
            save_checkpoint(state, args.exp_dir, ckpt_name, args.ckpt_keep_num)
            logging.info('Best total score {} found in epoch {} step {}'
                         .format(best_results['best_score'], best_results['best_epoch'], best_results['best_step']))
            logging.info('Best sel metrics (total): {}'.format(best_results['best_sel_metrics']))
            logging.info('Best sel metrics (individual) {}'.format(best_domain_scores))
            logging.info('Student state dict has been saved to \'{}\''.format(os.path.join(args.exp_dir, ckpt_name)))
            logging.info('-' * 50)

    if args.local_rank == 0:
        elapsed = round(time.time() - st_time)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        logging.info('Finished, total training time (h:m:s): {}'.format(elapsed))


def train_epoch(model, epoch, optim_tools, train_loader, all_dev_loaders, best_results, best_domain_scores):
    model.train()

    optimizer, criterion, scheduler = optim_tools
    losses, train_time, data_time = [AverageMeter() for _ in range(3)]
    st_time = time.time()

    def _update_losses(all_losses, loss, data_size):
        if args.distributed:
            loss = reduce_tensor(loss.detach(), args.world_size)
        all_losses.update(loss.item(), data_size)

    for batch_idx, data in enumerate(train_loader):
        data_time.update(time.time() - st_time)

        all_domain_outputs, all_domain_labels = [], []
        for domain_id, domain_data in enumerate(data):
            if args.use_gpu:
                domain_data = [dat.cuda() for dat in domain_data]
            cur_domain_ids, cur_tok_ids, cur_seg_ids, cur_pos_ids, cur_attn_mask, cur_labels = domain_data
            outputs = model(cur_tok_ids, cur_seg_ids, cur_pos_ids, cur_attn_mask, domain_id)
            all_domain_outputs.append(outputs)
            all_domain_labels.append(cur_labels)

        loss = calc_loss(criterion, all_domain_outputs, all_domain_labels)
        optimizer.zero_grad()
        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()

        _update_losses(losses, loss, data[0][0].size(0))
        if args.use_gpu:
            torch.cuda.synchronize()
        train_time.update(time.time() - st_time)

        if args.local_rank == 0 and \
                (batch_idx == 0 or (batch_idx + 1) % args.disp_freq == 0 or batch_idx + 1 == len(train_loader)):
            lr = scheduler.get_lr()[0]
            logging.info('Epoch: [{}/{}][{}/{}]\t'
                         'LR: {:.2e}\t'
                         'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
                         'Train time: {train_time.val:.4f}s ({train_time.avg:.4f}s)\t'
                         'Load data time: {data_time.val:.4f}s ({data_time.avg:.4f}s)'
                         .format(epoch, args.total_epochs, batch_idx + 1, len(train_loader), lr,
                                 loss=losses, train_time=train_time, data_time=data_time))

        if args.local_rank == 0 and \
                (batch_idx == 0 or (batch_idx + 1) % (args.val_freq * 10) == 0 or batch_idx + 1 == len(train_loader)):
            if args.use_graph and args.hierarchical:
                all_domain_layer_weights = model.module.all_task_layer_weights if args.use_gpu else model.all_task_layer_weights
                softmax_weights = [torch.softmax(weights, dim=-1).detach().cpu() for weights in all_domain_layer_weights]

                logging.info('-' * 50)
                logging.info('Layer weights:')
                for cur_domain_weights in softmax_weights:
                    logging.info(cur_domain_weights)
                logging.info('-' * 50)

        if ((batch_idx + 1) % args.val_freq == 0 or batch_idx + 1 == len(train_loader)):
            total_score, all_sel_metrics, all_metrics = validate(model, all_dev_loaders)

            # Check total score
            if total_score > best_results['best_score']:
                best_results['best_score'] = total_score
                best_results['best_sel_metrics'] = all_sel_metrics
                best_results['best_metrics'] = all_metrics
                best_results['best_epoch'] = epoch
                best_results['best_step'] = batch_idx + 1

                if args.local_rank == 0:
                    state = {'state_dict': model.state_dict(),
                             'all_sel_metrics': all_sel_metrics,
                             'all_metrics': all_metrics,
                             'epoch': epoch,
                             'step': batch_idx + 1}
                    save_path = os.path.join(args.exp_dir, 'best_model.bin')
                    torch.save(state, save_path)
                    logging.info('Best total score found: {}'.format(total_score))

            # Check the score of each task individually
            for domain in args.all_domains:
                cur_metrics = all_metrics[domain]
                cur_score = all_sel_metrics[domain]
                cur_best_score = best_domain_scores[domain]
                if cur_score > cur_best_score:
                    best_domain_scores[domain] = cur_score
                    if args.local_rank == 0:
                        state = {'state_dict': model.state_dict(),
                                 'sel_metric': cur_score,
                                 'metrics': cur_metrics,
                                 'epoch': epoch,
                                 'step': batch_idx + 1}
                        save_path = os.path.join(args.exp_dir, 'best_model_' + domain + '.bin')
                        torch.save(state, save_path)
                        logging.info('Best {} score found: {}'.format(domain, cur_score))

            model.train()
        st_time = time.time()


def validate(model, data_loader, verbose=False):
    model.eval()

    val_time = AverageMeter()
    st_time = time.time()

    if args.do_test:
        all_metrics = {}
        with torch.no_grad():
            for batch_idx, data in enumerate(data_loader):
                if args.use_gpu:
                    data = [data_.cuda() for data_ in data]

                if args.task == 'amazon_review':
                    _, token_ids, segment_ids, position_ids, attn_mask, labels = data
                else:
                    token_ids, segment_ids, position_ids, attn_mask, labels = data

                domain_id = args.domains_to_ids[args.domain]
                outputs = model(token_ids, segment_ids, position_ids, attn_mask, domain_id)
                if args.use_graph:
                    outputs = outputs[0]
                preds, labels = outputs.detach().cpu().numpy(), labels.detach().cpu().numpy()
                preds = np.argmax(preds, axis=1)
                metrics = compute_multi_domain_metrics(args.task, preds, labels)

                # Average metrics
                if args.distributed:
                    for k, v in metrics.items():
                        metrics[k] = reduce_tensor(torch.tensor(metrics[k], dtype=torch.float64).cuda(), args.world_size).cpu().numpy()

                for k, v in metrics.items():
                    if all_metrics.get(k) is None:
                        all_metrics[k] = AverageMeter()
                    all_metrics[k].update(metrics[k], token_ids.size(0))
                avg_metrics = {k: v.avg for k, v in all_metrics.items()}

                if args.use_gpu:
                    torch.cuda.synchronize()
                val_time.update(time.time() - st_time)

                if verbose and args.local_rank == 0 and \
                        (batch_idx == 0 or (batch_idx + 1) % args.disp_freq == 0 or batch_idx + 1 == len(data_loader)):
                    logging.info('Iter: [{}/{}]\tVal time: {:.4f}s\tMetrics: {}'
                                 .format(batch_idx + 1, len(data_loader), val_time.avg, avg_metrics))
                st_time = time.time()

            sel_metric = avg_metrics[all_multi_domain_select_metrics[args.task]]
            return sel_metric, avg_metrics

    else:
        all_domain_sel_metrics = {}
        all_domain_metrics = {}

        all_data_loaders = data_loader
        with torch.no_grad():
            for domain_id, (domain, data_loader) in enumerate(zip(args.all_domains, all_data_loaders)):
                all_metrics = {}
                for batch_idx, data in enumerate(data_loader):
                    if args.use_gpu:
                        data = [data_.cuda() for data_ in data]
                    domain_ids, token_ids, segment_ids, position_ids, attn_mask, labels = data

                    outputs = model(token_ids, segment_ids, position_ids, attn_mask, domain_id)
                    if args.use_graph:
                        outputs = outputs[0]
                    preds, labels = outputs.detach().cpu().numpy(), labels.detach().cpu().numpy()
                    preds = np.argmax(preds, axis=1)
                    metrics = compute_multi_domain_metrics(args.task, preds, labels)

                    # Average metrics
                    if args.distributed:
                        for k, v in metrics.items():
                            metrics[k] = reduce_tensor(torch.tensor(metrics[k], dtype=torch.float64).cuda(), args.world_size).cpu().numpy()

                    for k, v in metrics.items():
                        if all_metrics.get(k) is None:
                            all_metrics[k] = AverageMeter()
                        all_metrics[k].update(metrics[k], token_ids.size(0))
                    avg_metrics = {k: v.avg for k, v in all_metrics.items()}

                    if args.use_gpu:
                        torch.cuda.synchronize()
                    val_time.update(time.time() - st_time)

                    if verbose and args.local_rank == 0 and \
                            (batch_idx == 0 or (batch_idx + 1) % args.disp_freq == 0 or batch_idx + 1 == len(data_loader)):
                        logging.info('Iter: [{}/{}]\tVal time: {:.4f}s\tMetrics: {}'
                                     .format(batch_idx + 1, len(data_loader), val_time.avg, avg_metrics))
                    st_time = time.time()

                sel_metric = avg_metrics[all_multi_domain_select_metrics[args.task]]
                all_domain_sel_metrics[domain] = sel_metric
                all_domain_metrics[domain] = avg_metrics
                if args.local_rank == 0:
                    args.writer.add_scalar('sel_metric/{}'.format(domain), sel_metric)

            total_score = sum([metric for metric in all_domain_sel_metrics.values()]) / len(all_domain_sel_metrics)
            if args.local_rank == 0:
                args.writer.add_scalar('sel_metric/total', total_score)
            return total_score, all_domain_sel_metrics, all_domain_metrics


def calc_loss(criterion, all_domain_outputs, all_domain_labels):
    total_loss = 0

    if args.use_graph:
        args.meta_graph.train()

        all_domain_meta_feats = []
        for domain_id, domain in enumerate(args.all_domains):
            if args.hierarchical:
                cur_pred_logits, cur_attn_outputs, cur_ffn_outputs, cur_weighted_ffn_outputs = all_domain_outputs[domain_id]
                all_domain_meta_feats.append(cur_weighted_ffn_outputs)
            else:
                cur_pred_logits, cur_attn_outputs, cur_ffn_outputs = all_domain_outputs[domain_id]
                all_domain_meta_feats.append(torch.stack(cur_ffn_outputs, dim=0).mean(dim=0))

        all_domain_meta_feats = torch.stack(all_domain_meta_feats, dim=0)  # (n_domain, bs, seq_len, hid_sz)
        all_domain_meta_feats = all_domain_meta_feats.mean(1).mean(1)  # (n_domain, hid_sz)
        all_domain_meta_feats = all_domain_meta_feats.unsqueeze(0)  # (1, n_domain, hid_sz)
        all_domain_ratios = args.meta_graph(all_domain_meta_feats)[0]  # (n_domain)

        for domain_id, domain in enumerate(args.all_domains):
            cur_pred_logits, cur_attn_outputs, cur_ffn_outputs = all_domain_outputs[domain_id][:3]
            cur_labels = all_domain_labels[domain_id]
            cur_loss = criterion(cur_pred_logits, cur_labels)

            cur_ratio = all_domain_ratios[domain_id]
            total_loss += cur_ratio * cur_loss

            if args.local_rank == 0:
                args.writer.add_scalar('ratio/{}'.format(domain), cur_ratio)
                args.writer.add_scalar('loss/{}'.format(domain), cur_loss)
    else:
        for domain_id, domain in enumerate(args.all_domains):
            cur_outputs = all_domain_outputs[domain_id]
            cur_labels = all_domain_labels[domain_id]
            cur_loss = criterion(cur_outputs, cur_labels)
            total_loss += cur_loss

    return total_loss


if __name__ == '__main__':
    main()
