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
from models import select_config, select_model
from tokenizers import select_tokenizer
from metrics import all_multi_domain_select_metrics, compute_multi_domain_metrics
from utils import AverageMeter, set_seeds, setup_logger, calc_params, reduce_tensor, soft_cross_entropy, save_checkpoint, load_pretrain_state_dict, \
    load_resume_state_dict, create_optimizer, create_scheduler, create_single_domain_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--distributed', action='store_true', help='distributed mode')
parser.add_argument('--fp16', action='store_true', help='mixed precision training mode')
parser.add_argument('--opt_level', default='O1', type=str, help='fp16 optimization level')
parser.add_argument('--gpu_devices', default='2', type=str, help='available gpu devices')
parser.add_argument('--seed', default=42, type=int, help='seed')

parser.add_argument('--lowercase', action='store_true', help='whether to do lowercase')
parser.add_argument('--do_test', action='store_true', help='whether to do test')
parser.add_argument('--temperature', default=1, type=float, help='temperature for soft cross entropy loss')
parser.add_argument('--hidden_ratio', default=1, type=float, help='ratio for hidden loss')
parser.add_argument('--pred_ratio', default=1, type=float, help='ratio for prediction loss')
parser.add_argument('--ffn_expr', default=[], nargs='+', help='feed-forward network expression')
parser.add_argument('--teacher_model_name', default='bert_base', type=str, help='teacher model name')
parser.add_argument('--student_model_name', default='tiny_bert', type=str, help='student model name')

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
parser.add_argument('--total_epochs', default=10, type=int, help='total epochs')
parser.add_argument('--batch_size', default=8, type=int, help='batch size')
parser.add_argument('--lr', default=5e-5, type=float, help='initial learning rate')
parser.add_argument('--optim_type', default='adamw', type=str, help='optimizer type')
parser.add_argument('--sched_type', default='step', type=str, help='lr scheduler type')
parser.add_argument('--warmup_proportion', default=0.1, type=float, help='proportion of warmup steps')
parser.add_argument('--momentum', default=0.9, type=float, help='SGD momentum')
parser.add_argument('--weight_decay', default=0.01, type=float, help='weight decay')
parser.add_argument('--max_grad_norm', default=1.0, type=float, help='max gradient norm')
parser.add_argument('--disp_freq', default=50, type=int, help='display step frequency')
parser.add_argument('--val_freq', default=50, type=int, help='validate step frequency')
parser.add_argument('--ckpt_keep_num', default=1, type=int, help='max number of checkpoint files to keep')
parser.add_argument('--num_workers', default=1, type=int, help='num workers of dataloader')

parser.add_argument('--teacher_pretrain_path', default='', type=str, help='path to pretrained teacher state dict')
parser.add_argument('--student_pretrain_path', default='', type=str, help='path to pretrained student state dict')
parser.add_argument('--student_resume_path', default='', type=str, help='path to student resume checkpoint')
parser.add_argument('--cache_dir', default='./cache', type=str, help='cache directory to save processed dataset')
parser.add_argument('--exp_dir', default='./exp/tmp/', type=str, help='experiment directory')
parser.add_argument('--local_rank', default=0, type=int, help='DDP local rank')
parser.add_argument('--world_size', default=1, type=int, help='DDP world size')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices


def main():
    args.exp_dir = os.path.join(args.exp_dir, datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    setup_logger(args.exp_dir)
    if args.local_rank == 0:
        logging.info(args)

    args.use_gpu = False
    if args.gpu_devices and torch.cuda.is_available():
        args.use_gpu = True
    if args.use_gpu:
        if args.local_rank == 0:
            logging.info('Currently using GPU: {}'.format(args.gpu_devices))
    else:
        if args.local_rank == 0:
            logging.info('Currently using CPU')
    set_seeds(args.seed, args.use_gpu)

    if args.use_gpu and args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        logging.info('Training in distributed mode (process {}/{})'.format(args.local_rank + 1, args.world_size))

    # Load model and tokenizer
    teacher_config = select_config(args.teacher_model_name, args.lowercase)
    student_config = select_config(args.student_model_name, args.lowercase)
    teacher_model = select_model(args.teacher_model_name, args.lowercase, args.task, return_hid=True)
    student_model = select_model(args.student_model_name, args.lowercase, args.task, return_hid=True)
    args.tokenizer = select_tokenizer(
        args.teacher_model_name, args.lowercase, args.task, args.vocab_path, args.max_seq_len, args.max_query_len)
    args.teacher_interval = teacher_config.num_layers // student_config.num_layers
    args.num_student_layers = student_config.num_layers
    args.student_config = student_config

    if args.use_gpu:
        teacher_model, student_model = teacher_model.cuda(), student_model.cuda()
    if args.local_rank == 0:
        logging.info('Teacher model size: {:.2f}M'.format(calc_params(teacher_model) / 1e6))
        logging.info('Student model size: {:.2f}M'.format(calc_params(student_model) / 1e6))
        logging.info('Student model config: {}'.format(args.student_config.__dict__))

    # Create dataset
    load_from_multi_domain = True if args.task == 'amazon_review' else False
    if args.do_test:
        _, _, _, test_loader = create_single_domain_dataset(
            args.student_model_name, args.task, args.domain, args.data_dir, args.tokenizer, args.max_seq_len,
            args.batch_size * 4, args.use_gpu, args.distributed, 'test', args.local_rank, args.cache_dir,
            args.train_ratio, args.dev_ratio, args.num_workers, load_from_multi_domain)
        optimizer = None
        scheduler = None
    else:
        _, _, _, train_loader = create_single_domain_dataset(
            args.student_model_name, args.task, args.domain, args.data_dir, args.tokenizer, args.max_seq_len,
            args.batch_size, args.use_gpu, args.distributed, 'train', args.local_rank, args.cache_dir,
            args.train_ratio, args.dev_ratio, args.num_workers, load_from_multi_domain)
        _, _, _, dev_loader = create_single_domain_dataset(
            args.student_model_name, args.task, args.domain, args.data_dir, args.tokenizer, args.max_seq_len,
            args.batch_size * 4, args.use_gpu, args.distributed, 'dev', args.local_rank, args.cache_dir,
            args.train_ratio, args.dev_ratio, args.num_workers, load_from_multi_domain)

        # Create optimization tools
        args.num_sched_steps = len(train_loader) * args.total_epochs
        args.num_warmup_steps = int(args.num_sched_steps * args.warmup_proportion)
        optimizer = create_optimizer(student_model, args.optim_type, args.lr, args.weight_decay, args.momentum)
        scheduler = create_scheduler(optimizer, args.sched_type, args.num_sched_steps, args.num_warmup_steps)

    # Enable fp16/distributed training
    if args.use_gpu:
        if args.fp16:
            amp.register_half_function(torch, 'einsum')
            if optimizer is not None:
                student_model, optimizer = amp.initialize(student_model, optimizer, opt_level=args.opt_level)
            if args.local_rank == 0:
                logging.info('Using fp16 training mode')
        if args.distributed:
            teacher_model = DDP(teacher_model, delay_allreduce=True)
            student_model = DDP(student_model, delay_allreduce=True)
        else:
            teacher_model = nn.DataParallel(teacher_model)
            student_model = nn.DataParallel(student_model)

    # Load model weights
    ckpt_path = args.teacher_pretrain_path
    if ckpt_path:
        if os.path.exists(ckpt_path):
            load_pretrain_state_dict(teacher_model, ckpt_path, args.use_gpu, is_finetune=True)
            if args.local_rank == 0:
                logging.info('Loaded teacher pretrained state dict from \'{}\''.format(ckpt_path))
        else:
            if args.local_rank == 0:
                logging.info('Teacher pretrained state dict is not found in \'{}\''.format(ckpt_path))

    ckpt_path = args.student_pretrain_path
    if ckpt_path:
        if os.path.exists(ckpt_path):
            load_pretrain_state_dict(student_model, ckpt_path, args.use_gpu)
            if args.local_rank == 0:
                logging.info('Loaded student pretrained state dict from \'{}\''.format(ckpt_path))
        else:
            if args.local_rank == 0:
                logging.info('Student pretrained state dict is not found in \'{}\''.format(ckpt_path))

    ckpt_path = args.student_resume_path
    if ckpt_path:
        if os.path.exists(ckpt_path):
            checkpoint = load_resume_state_dict(student_model, ckpt_path, args.use_gpu, optimizer, scheduler)
            args.start_epoch = checkpoint['epoch'] + 1
            if args.local_rank == 0:
                logging.info('Loaded student resume checkpoint from \'{}\''.format(ckpt_path))
                logging.info('Start epoch: {}\tMetrics: {}'.format(args.start_epoch, checkpoint['metrics']))
        else:
            if args.local_rank == 0:
                logging.info('No checkpoint found in \'{}\''.format(ckpt_path))

    try:
        if args.do_test:
            validate(student_model, test_loader, verbose=True)
        else:
            train(teacher_model, student_model, optimizer, scheduler, train_loader, dev_loader)
    except KeyboardInterrupt:
        print('Keyboard interrupt (process {}/{})'.format(args.local_rank + 1, args.world_size))


def train(teacher_model, student_model, optimizer, scheduler, train_loader, dev_loader):
    st_time = time.time()
    if args.local_rank == 0:
        logging.info('==> Start training')

    best_results = {
        'best_sel_metric': 0,
        'best_metrics': None,
        'best_epoch': None,
        'best_step': None,
    }
    for epoch in range(args.start_epoch, args.total_epochs + 1):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
        train_epoch(teacher_model, student_model, epoch, optimizer, scheduler, train_loader, dev_loader, best_results)
        sel_metric, metrics = validate(student_model, dev_loader)

        if args.local_rank == 0:
            logging.info('-' * 50)
            state = {'state_dict': student_model.state_dict(),
                     'optimizer': optimizer.state_dict(),
                     'scheduler': scheduler.state_dict(),
                     'metrics': metrics,
                     'epoch': epoch}
            ckpt_name = 'ckpt_ep' + str(epoch) + '.bin'
            save_checkpoint(state, args.exp_dir, ckpt_name, args.ckpt_keep_num)
            logging.info('Best select metric {} found in epoch {} step {}'
                         .format(best_results['best_sel_metric'], best_results['best_epoch'], best_results['best_step']))
            logging.info('Best metrics: {}, current metrics: {}'.format(best_results['best_metrics'], metrics))
            logging.info('Student state dict has been saved to \'{}\''.format(os.path.join(args.exp_dir, ckpt_name)))
            logging.info('-' * 50)

    if args.local_rank == 0:
        elapsed = round(time.time() - st_time)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        logging.info('Finished, total training time (h:m:s): {}'.format(elapsed))


def train_epoch(teacher_model, student_model, epoch, optimizer, scheduler, train_loader, dev_loader, best_results):
    teacher_model.eval()
    student_model.train()

    losses, train_time, data_time = [AverageMeter() for _ in range(3)]
    attn_losses, ffn_losses, pred_losses = [AverageMeter() for _ in range(3)]
    st_time = time.time()

    def _update_losses(all_losses, loss, data_size):
        if args.distributed:
            loss = reduce_tensor(loss.detach(), args.world_size)
        all_losses.update(loss.item(), data_size)

    for batch_idx, data in enumerate(train_loader):
        data_time.update(time.time() - st_time)
        if args.use_gpu:
            data = [data_.cuda() for data_ in data]

        if args.task == 'amazon_review':
            _, token_ids, segment_ids, position_ids, attn_mask, labels = data
        else:
            token_ids, segment_ids, position_ids, attn_mask, labels = data

        with torch.no_grad():
            teacher_outputs = teacher_model(token_ids, segment_ids, position_ids, attn_mask)
        student_outputs = student_model(token_ids, segment_ids, position_ids, attn_mask)
        loss, attn_loss, ffn_loss, pred_loss = calc_distil_losses(teacher_outputs, student_outputs)

        optimizer.zero_grad()
        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()

        _update_losses(losses, loss, token_ids.size(0))
        _update_losses(attn_losses, attn_loss, token_ids.size(0))
        _update_losses(ffn_losses, ffn_loss, token_ids.size(0))
        _update_losses(pred_losses, pred_loss, token_ids.size(0))

        if args.use_gpu:
            torch.cuda.synchronize()
        train_time.update(time.time() - st_time)

        if args.local_rank == 0 and \
                (batch_idx == 0 or (batch_idx + 1) % args.disp_freq == 0 or batch_idx + 1 == len(train_loader)):
            lr = scheduler.get_lr()[0]
            logging.info('Epoch: [{}/{}][{}/{}]\t'
                         'LR: {:.2e}\t'
                         'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
                         'Attn, ffn and pred loss: {attn_loss.val:.4f} {ffn_loss.val:.4f} {pred_loss.val:.4f} '
                         '({attn_loss.avg:.4f} {ffn_loss.avg:.4f} {pred_loss.avg:.4f})\t'
                         'Train time: {train_time.val:.4f}s ({train_time.avg:.4f}s)\t'
                         'Load data time: {data_time.val:.4f}s ({data_time.avg:.4f}s)'
                         .format(epoch, args.total_epochs, batch_idx + 1, len(train_loader), lr,
                                 loss=losses, attn_loss=attn_losses, ffn_loss=ffn_losses, pred_loss=pred_losses,
                                 train_time=train_time, data_time=data_time))

        if (batch_idx + 1) % args.val_freq == 0 or batch_idx + 1 == len(train_loader):
            sel_metric, metrics = validate(student_model, dev_loader)
            if sel_metric > best_results['best_sel_metric']:
                best_results['best_sel_metric'] = sel_metric
                best_results['best_metrics'] = metrics
                best_results['best_epoch'] = epoch
                best_results['best_step'] = batch_idx + 1

                if args.local_rank == 0:
                    state = {'state_dict': student_model.state_dict(),
                             'metrics': best_results['best_metrics'],
                             'epoch': best_results['best_epoch'],
                             'step': best_results['best_step'],
                             }
                    save_path = os.path.join(args.exp_dir, 'best_model.bin')
                    torch.save(state, save_path)
                    logging.info('Best metric found: {}'.format(best_results['best_sel_metric']))

            student_model.train()
        st_time = time.time()


def validate(model, data_loader, verbose=False):
    model.eval()

    val_time = AverageMeter()
    all_metrics = {}
    st_time = time.time()

    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader):
            if args.use_gpu:
                data = [data_.cuda() for data_ in data]

            if args.task == 'amazon_review':
                _, token_ids, segment_ids, position_ids, attn_mask, labels = data
            else:
                token_ids, segment_ids, position_ids, attn_mask, labels = data
            outputs = model(token_ids, segment_ids, position_ids, attn_mask)
            preds, labels = outputs[0].detach().cpu().numpy(), labels.detach().cpu().numpy()
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


def calc_distil_losses(teacher_outputs, student_outputs):
    teacher_pred_logits, teacher_attn_outputs, teacher_ffn_outputs = teacher_outputs
    student_pred_logits, student_attn_outputs, student_ffn_outputs = student_outputs

    def _replace_attn_mask(attn_output):
        replace_values = torch.zeros_like(attn_output)
        if args.use_gpu:
            replace_values = replace_values.cuda()
        attn_output = torch.where(attn_output <= -1e2, replace_values, attn_output)
        return attn_output

    mse_loss = nn.MSELoss()
    attn_loss, ffn_loss = 0, 0
    ffn_loss += mse_loss(teacher_ffn_outputs[0], student_ffn_outputs[0])
    for layer_id in range(args.num_student_layers):
        teacher_layer_id = (layer_id + 1) * args.teacher_interval - 1
        attn_loss += mse_loss(_replace_attn_mask(teacher_attn_outputs[teacher_layer_id]),
                              _replace_attn_mask(student_attn_outputs[layer_id]))
        ffn_loss += mse_loss(teacher_ffn_outputs[teacher_layer_id + 1], student_ffn_outputs[layer_id + 1])

    hidden_loss = attn_loss + ffn_loss
    pred_loss = soft_cross_entropy(student_pred_logits, teacher_pred_logits, args.temperature)
    total_loss = args.hidden_ratio * hidden_loss + args.pred_ratio * pred_loss
    return total_loss, attn_loss, ffn_loss, pred_loss


if __name__ == '__main__':
    main()
