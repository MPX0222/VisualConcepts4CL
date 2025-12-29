import os
import ast
import copy
import math
from typing import Tuple, Any
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.distributions.multivariate_normal import MultivariateNormal
import wandb
from tqdm import tqdm
from random import sample
from methods.Base import Base
from model.CLIP_Concept_Net import CLIP_Concept_Net
from model.backbone.clip import clip
from ReplayBank import ReplayBank
from utils.train_utils import *
from utils.functions import *


class CLIP_Concept(Base):
    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.memory_bank = ReplayBank(config, logger) if self.config.memory_size or self.config.memory_per_class else None

        self.desc_num = config.desc_num
        self.class_covs = None
        self.class_to_idx = None
        self.cur_class_names = []
        self.new_class_names = []
        self.cur_text_tokens = None
        self.new_text_tokens = None
        self.prompt_template = config.prompt_template if config.prompt_template is not None else "a photo of a {}."
        with open(config.desc_path, "r") as f:
            desc = f.read()
        self.all_descs = ast.literal_eval(desc)
        if config.increment_type != 'CIL':
            raise ValueError('This is a class incremental method!')

    def get_task_descs_and_indices(self, class_descs, classes):
        unique_descs_set = set()
        for cls in classes:
            unique_descs_set.update(class_descs[cls])
        unique_descs = list(unique_descs_set)

        desc_2_idx = {desc: idx for idx, desc in enumerate(unique_descs)}

        mask = torch.zeros((len(classes), len(unique_descs)), dtype=torch.float32).cuda()

        for i, cls in enumerate(classes):
            for desc in class_descs[cls]:
                if desc in desc_2_idx:
                    mask[i, desc_2_idx[desc]] = 1

        return unique_descs, mask

    def prepare_task_data(self, data_manager, task_id, is_train=True):
        if self.class_to_idx is None:
            self.class_to_idx = data_manager.class_to_idx
            self.class_descs = data_manager.class_descs
            self.idx_to_class = dict((value, key) for key, value in self.class_to_idx.items())
        if is_train:
            if task_id > 0 and self.memory_bank is not None:
                self.train_dataset = data_manager.get_dataset(source='train', mode='train',
                                                              indices=np.arange(self.known_classes, self.cur_classes),
                                                              appendent=self.memory_bank.get_memory())
            else:
                self.train_dataset = data_manager.get_dataset(source='train', mode='train',
                                                              indices=np.arange(self.known_classes, self.cur_classes))
            self.train_loader = DataLoader(self.train_dataset, batch_size=self.config.batch_size, shuffle=True,
                                           num_workers=self.config.num_workers)
            self.logger.info("train data num of task {}: {}".format(task_id + 1, len(self.train_dataset.samples)))

        self.test_dataset = data_manager.get_dataset(source='test', mode='test', indices=np.arange(0, self.cur_classes))

        self.test_loader = DataLoader(self.test_dataset, batch_size=self.config.batch_size, shuffle=False,
                                      num_workers=self.config.num_workers)
        self.openset_test_dataset = data_manager.get_openset_dataset(source='test', mode='test',
                                                                     known_indices=np.arange(0, self.cur_classes))
        self.openset_test_loader = DataLoader(self.openset_test_dataset, batch_size=self.config.batch_size,
                                              shuffle=False,
                                              num_workers=self.config.num_workers)

        self.new_class_names = [self.idx_to_class[i] for i in range(self.known_classes, self.cur_classes)]
        self.cur_class_names = [self.idx_to_class[i] for i in range(0, self.cur_classes)]
        self.new_descs, self.new_mask = self.get_task_descs_and_indices(self.class_descs, self.new_class_names)
        self.cur_descs, self.cur_mask = self.get_task_descs_and_indices(self.class_descs, self.cur_class_names)

        self.logger.info("test data num of task {}: {}".format(task_id + 1, len(self.test_dataset.samples)))
        self.logger.info('Cur Task classnames: {}'.format(self.cur_class_names))

    def prepare_model(self, task_id, checkpoint=None):
        if self.model is None:
            self.model = CLIP_Concept_Net(self.config, self.logger)
            self.model.model_init()
        self.model.update_model(task_id)
        self.model.freeze_fe()
        self.model.stage1_param()
        if checkpoint is not None:
            assert task_id == checkpoint["task_id"]
            model_state_dict = checkpoint["state_dict"]
            self.model.load_state_dict(model_state_dict)
            if checkpoint["class_means"] is not None:
                self.class_means = checkpoint["class_means"]
            self.logger.info("checkpoint loaded!")
        self.model.show_trainable_params()

        self.new_text_tokens = clip.tokenize(self.new_descs)
        self.cur_text_tokens = clip.tokenize(self.cur_descs)
        self.model = self.model.cuda()

    def incremental_train(self, data_manager, task_id):
        wandb.define_metric("overall/task_id")
        wandb.define_metric("overall/*", step_metric="overall/task_id")

        params = []
        for name, param in self.model.backbone.visual.transformer.resblocks[-2:].named_parameters():
            if param.requires_grad:
                params.append({'params': param, 'lr': self.config.lr*0.01})

        other_params = []
        for name, param in self.model.named_parameters():
            if not name.startswith('backbone') and param.requires_grad:
                other_params.append(param)

        if other_params:
            params.append({'params': other_params, 'lr': self.config.lr})
        optimizer = optim.AdamW(params,lr=self.config.lr, weight_decay=self.config.weight_decay)
        scheduler = get_scheduler(optimizer, self.config)
        hard_loss = get_loss_func(self.config)
        soft_loss = None
        if len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) > 1:
            self.model = nn.DataParallel(self.model)
        self.train_model(self.train_loader, self.test_loader, hard_loss, soft_loss, optimizer, scheduler,
                         task_id=task_id, epochs=self.config.epochs)
        if len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) > 1:
            self.model = self.model.module

        if self.config.ca_epoch > 0:
            self.compute_mean_cov(data_manager)
            self.logger.info("class means and covs computed!")

            if task_id > 0:
                self.stage2_training(task_id)
                self.logger.info("stage 2 training finished!")

    def train_model(self, train_loader, test_loader, hard_loss, soft_loss, optimizer, scheduler, task_id, epochs):
        wandb.define_metric("task " + str(task_id + 1) + "/" + "epoch")
        wandb.define_metric("task " + str(task_id + 1) + "/*",
                            step_metric="task" + str(task_id + 1) + "/" + "epoch")

        for epoch in range(epochs):
            train_preds, train_targets, train_loss = self.epoch_train(self.model, train_loader, hard_loss, soft_loss,
                                                                          optimizer, task_id, epoch)
            if scheduler is not None:
                scheduler.step()

            test_preds, test_targets, test_loss = self.epoch_test(self.model, test_loader, hard_loss, soft_loss,
                                                                  task_id)

            train_overall_acc, _ = calculate_acc(train_preds.cpu().detach().numpy(),
                                                                   train_targets.cpu().detach().numpy(),
                                                                   self.cur_classes, self.config.increment_steps)
            test_overall_acc, _ = calculate_acc(test_preds.cpu().detach().numpy(),
                                                                 test_targets.cpu().detach().numpy(),
                                                                 self.cur_classes, self.config.increment_steps)

            wandb.log({
                "task " + str(task_id + 1) + "/" + "epoch": epoch + 1,
                "task " + str(task_id + 1) + "/" + "train_overall_acc": train_overall_acc,
                "task " + str(task_id + 1) + "/" + "test_overall_acc": test_overall_acc,
                "task " + str(task_id + 1) + "/" + "train_loss": train_loss["all_loss"],
                "task " + str(task_id + 1) + "/" + "test_loss": test_loss["all_loss"]
            })

            self.logger.info("task_id: {}, epoch: {}/{}".format(task_id + 1, epoch + 1, epochs))
            self.logger.info(
                "train_overall_acc: {:.2f}, test_overall_acc: {:.2f}".format(train_overall_acc, test_overall_acc))
            self.logger.info("train_losses: {}".format(train_loss))
            self.logger.info("test_losses: {}".format(test_loss))

    def epoch_train(self, model, train_loader, hard_loss, soft_loss, optimizer, task_id, epoch):
        losses = 0.
        ce_losses, local_losses, attn_losses = 0., 0., 0.
        model.train()
        for idx, (inputs, targets, _) in enumerate(train_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            with autocast():
                out = model(inputs, text_tokens=self.new_text_tokens.cuda(), train=True, task_id=task_id, labels=targets-self.known_classes)
                logits = out["logits"]  # [B, class_num, 3]

                B = logits.shape[0]
                final_logits = logits
                attn_scores = out["attn"]

                mask = self.new_mask[targets-self.known_classes].cuda()
                mask = mask / mask.sum(dim=-1, keepdim=True)
                # print(mask)

                log_probs = F.log_softmax(attn_scores, dim=-1)

                attn_loss = -(mask * log_probs).sum(dim=-1).mean()
                ce_loss = hard_loss(final_logits[:, self.known_classes:], targets-self.known_classes)

                loss = ce_loss + self.config.lambd * attn_loss
                ce_losses += ce_loss.item()
                attn_losses += attn_loss.item()

                # 预测结果
                preds = torch.max(final_logits[:, self.known_classes:], dim=1)[1] + self.known_classes

            if idx == 0:
                all_preds = preds
                all_targets = targets
            else:
                all_preds = torch.cat((all_preds, preds))
                all_targets = torch.cat((all_targets, targets))

            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            # losses += loss.item()

            optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
            losses += loss.item()

        train_loss = {'all_loss': losses / len(train_loader), 'loss_clf': ce_losses / len(train_loader)}

        return all_preds, all_targets, train_loss

    def epoch_test(self, model, test_loader, hard_loss, soft_loss, task_id):
        losses = 0.
        ce_losses, local_losses, text_losses = 0., 0., 0.
        model.eval()
        with torch.no_grad():
            for idx, (inputs, targets, _) in enumerate(test_loader):
                inputs, targets = inputs.cuda(), targets.cuda()
                out = model(inputs, text_tokens=self.cur_text_tokens.cuda(), task_id=task_id, labels=targets)
                logits = out["logits"]  # [B, class_num, 3]

                ce_loss = hard_loss(logits, targets)
                ce_losses += ce_loss.item()
                loss = ce_loss
                preds = torch.max(logits, dim=1)[1]  # [B]
                
                losses += loss.item()
                if idx == 0:
                    all_preds = preds
                    all_targets = targets
                else:
                    all_preds = torch.cat((all_preds, preds))
                    all_targets = torch.cat((all_targets, targets))

            test_loss = {'all_loss': losses / len(test_loader), 'loss_clf': ce_losses / len(test_loader)}

        return all_preds, all_targets, test_loss

    def predict(self, model, test_loader, task_id):
        model.eval()
        with torch.no_grad():
            for idx, (inputs, targets, _) in enumerate(test_loader):
                inputs, targets = inputs.cuda(), targets.cuda()
                out = model(inputs, text_tokens=self.cur_text_tokens.cuda(), task_id=task_id)
                logits = out["logits"].softmax(dim=-1)
                attn = out["attn"]
                scores, preds = torch.max(logits, dim=1)  # [B]

                if idx == 0:
                    all_preds = preds
                    all_scores = scores
                    all_targets = targets
                    all_attn = attn
                else:
                    all_preds = torch.cat((all_preds, preds))
                    all_scores = torch.cat((all_scores, scores))
                    all_targets = torch.cat((all_targets, targets))
                    all_attn = torch.cat((all_attn, attn))

            return all_preds, all_scores, all_targets

    def compute_mean_cov(self, data_manager, check_diff=False, oracle=False):
        if hasattr(self, 'class_means') and self.class_means is not None and not check_diff:
            ori_classes = self.class_means.shape[0]
            assert ori_classes == self.known_classes
            cur_class_means = torch.zeros((self.cur_classes, self.model.output_dim))
            cur_class_means[:self.known_classes] = self.class_means
            self.class_means = cur_class_means
            cur_class_cov = torch.zeros((self.cur_classes, self.model.output_dim, self.model.output_dim))
            cur_class_cov[:self.known_classes] = self.class_covs
            self.class_covs = cur_class_cov
        elif not check_diff:
            self.class_means = torch.zeros((self.cur_classes, self.model.output_dim))
            self.class_covs = torch.zeros((self.cur_classes, self.model.output_dim, self.model.output_dim))

        if check_diff or oracle:
            old_class_dataset = data_manager.get_dataset(source='train', mode='test',
                                                         indices=np.arange(0, self.known_classes))
            for class_idx in range(0, self.known_classes):
                vectors, _, _ = extract_vectors(self.config, self.model, old_class_dataset, class_idx)
                vectors = vectors.type(torch.float64)
                old_class_mean = torch.mean(vectors, dim=0)
                old_class_cov = torch.cov(vectors.T).detach().cpu() + torch.eye(old_class_mean.shape[-1]) * 1e-5
                if oracle:
                    self.class_means[class_idx, :] = old_class_mean
                    self.class_covs[class_idx, ...] = old_class_cov
                if check_diff:
                    log_info = "cls {} sim: {}".format(class_idx, torch.cosine_similarity(
                        self.class_means[class_idx, :].unsqueeze(0),
                        old_class_mean.unsqueeze(0)).item())
                    self.logger.info(log_info)

        new_class_dataset = data_manager.get_dataset(source='train', mode='test',
                                                     indices=np.arange(self.known_classes, self.cur_classes))
        for class_idx in range(self.known_classes, self.cur_classes):
            vectors, _, _ = extract_vectors(self.config, self.model, new_class_dataset, class_idx)
            vectors = vectors.type(torch.float64)
            new_class_mean = torch.mean(vectors, dim=0)
            new_class_cov = torch.cov(vectors.T).detach().cpu() + torch.eye(new_class_mean.shape[-1]) * 1e-4
            self.class_means[class_idx, :] = new_class_mean
            self.class_covs[class_idx, ...] = new_class_cov

    def stage2_training(self, task_id):
        self.model.show_trainable_params()
        optimizer2 = optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.config.ca_lr,
                                 weight_decay=self.config.weight_decay)
        scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer2, T_max=self.config.ca_epoch)

        # self.model.eval()
        for epoch in range(self.config.ca_epoch):
            sampled_data = []
            sampled_label = []
            num_sampled_pcls = self.config.num_sampled_pcls

            for c_id in range(self.cur_classes):
                cls_mean = self.class_means[c_id].cuda()  # torch.from_numpy(self._class_means[c_id]).to(self._device)
                cls_cov = self.class_covs[c_id].cuda()

                m = MultivariateNormal(cls_mean.float(), cls_cov.float())

                sampled_data_pcls = m.sample(sample_shape=(num_sampled_pcls,))
                sampled_data.append(sampled_data_pcls)
                sampled_label.extend([c_id] * num_sampled_pcls)

            inputs = torch.cat(sampled_data, dim=0).float().cuda()
            targets = torch.tensor(sampled_label).long().cuda()

            sf_indexes = torch.randperm(inputs.size(0))
            inputs = inputs[sf_indexes]
            targets = targets[sf_indexes]
            assert inputs.shape[0] % num_sampled_pcls == 0
            for i in range(inputs.shape[0] // num_sampled_pcls):
                inp = inputs[i * num_sampled_pcls:(i + 1) * num_sampled_pcls]
                tgt = targets[i * num_sampled_pcls:(i + 1) * num_sampled_pcls]
                with autocast():
                    outputs = self.model.forward_with_vectors(inp, self.cur_text_tokens.cuda(), labels=tgt)
                    logits = outputs['logits']

                if self.config.ca_logit_norm > 0:
                    per_task_norm = []
                    prev_t_size = 0
                    cur_t_size = 0
                    for _ti in range(task_id + 1):
                        cur_t_size += self.config.increment_steps[_ti]
                        temp_norm = torch.norm(logits[:, prev_t_size:cur_t_size], p=2, dim=-1, keepdim=True) + 1e-7
                        per_task_norm.append(temp_norm)
                        prev_t_size += self.config.increment_steps[_ti]
                    per_task_norm = torch.cat(per_task_norm, dim=-1)
                    norms = per_task_norm.mean(dim=-1, keepdim=True)

                    decoupled_logits = torch.div(logits[:, :self.cur_classes], norms) / self.config.ca_logit_norm
                    loss = F.cross_entropy(decoupled_logits, tgt)
                else:
                    B = logits.shape[0]
                    final_logits = logits
                    attn_scores = outputs["attn"]  # [B, new_class*3]

                    mask = self.cur_mask[tgt].cuda()
                    mask = mask / mask.sum(dim=-1, keepdim=True)

                    log_probs = F.log_softmax(attn_scores, dim=-1)
                    attn_loss = -(mask * log_probs).sum(dim=-1).mean()

                    ce_loss = F.cross_entropy(final_logits, tgt)
                    loss = ce_loss + self.config.lambd * attn_loss
                # optimizer2.zero_grad()
                # loss.backward()
                # optimizer2.step()
                # losses += loss.item()

                optimizer2.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer2)
                self.scaler.update()

                preds = torch.max(final_logits, dim=-1)[1]
                if i == 0:
                    all_preds = preds
                    all_targets = tgt
                else:
                    all_preds = torch.cat((all_preds, preds))
                    all_targets = torch.cat((all_targets, tgt))
            # print(all_preds.shape)
            train_overall_acc, _ = calculate_acc(all_preds.detach().cpu().numpy(), all_targets.detach().cpu().numpy(),
                                                 self.cur_classes, self.config.increment_steps)
            self.logger.info("stage2 train acc: {}".format(train_overall_acc))

            scheduler2.step()



