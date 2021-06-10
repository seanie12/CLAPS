import os
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from transformers import T5ForConditionalGeneration
from transformers.optimization import get_linear_schedule_with_warmup

from dataset import get_dist_loader, get_loader
from models import AdvContrastiveQG
from optimizers import Adafactor
from train_utils import (cal_running_avg_loss, eta, progress_bar, time_since,
                         user_friendly_time)


class Trainer(object):
    def __init__(self, args):
        self.args = args

        self.train_loader = None
        self.sampler = None

        self.model = None
        self.optimizer = None
        self.model_dir = args.model_dir

    def make_model_env(self, gpu, ngpus_per_node):
        self.args.gpu = self.args.devices[gpu]

        # For multiprocessing distributed training, rank needs to be the
        # global rank among all the processes
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.args.rank = self.args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=self.args.dist_backend, init_method=self.args.dist_url,
                                world_size=self.args.world_size, rank=self.args.rank)
        self.model = AdvContrastiveQG(self.args)

        
        torch.cuda.set_device(self.args.gpu)
        self.model.cuda(self.args.gpu)
        self.args.batch_size = int(self.args.batch_size / ngpus_per_node)
        self.args.workers = (self.args.workers +
                             ngpus_per_node - 1) // ngpus_per_node

        self.train_loader, self.sampler = get_dist_loader(self.args.train_features,
                                                          shuffle=True,
                                                          args=self.args)
        param = self.model.parameters()
        self.optimizer = Adafactor(param,
                                   relative_step=False,
                                   scale_parameter=False,
                                   lr=self.args.lr)
        t_total = len(self.train_loader) * self.args.num_epochs
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                         num_warmup_steps=0,
                                                         num_training_steps=t_total)
        self.model = DistributedDataParallel(self.model, device_ids=[self.args.gpu],
                                             find_unused_parameters=True)

        cudnn.benchmark = True

    def process_batch(self, batch):
        (input_ids, attention_mask,
         decoder_input_ids, decoder_attention_mask, lm_labels) = batch
        src_length = torch.sum(attention_mask, 1)
        max_src_length = torch.max(src_length)

        trg_length = torch.sum(decoder_attention_mask, 1)
        max_trg_length = torch.max(trg_length)

        input_ids = input_ids[:, :max_src_length]
        attention_mask = attention_mask[:, :max_src_length]

        decoder_input_ids = decoder_input_ids[:, :max_trg_length]
        decoder_attention_mask = decoder_attention_mask[:, :max_trg_length]
        lm_labels = lm_labels[:, :max_trg_length]

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "lm_labels": lm_labels
        }
        return inputs

    
    # train and evaluate
    def train(self, model_path=None):
        best_loss = 1e12
        n_iter, running_avg_loss = 0, 0.0
        batch_nb = len(self.train_loader)
        for epoch in range(self.args.start_epoch, self.args.num_epochs + 1):
            start = time.time()
            self.model.train()
            self.sampler.set_epoch(epoch)
            for batch_idx, batch in enumerate(self.train_loader, start=1):
                batch = self.process_batch(batch)

                # assign tensor to gpu
                inputs = dict()
                for k, v in batch.items():
                    inputs[k] = v.to(self.args.gpu)
                inputs["adv"] = True
                nll, cont_loss = self.model(**inputs)
                
                loss = nll + cont_loss
                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                # compute running average
                batch_loss = nll.detach().item()
                running_avg_loss = cal_running_avg_loss(batch_loss,
                                                        running_avg_loss)
                n_iter += 1
                msg = "{}/{} {} - ETA : {} - loss: {:.4f}, cont: {:.4f}".format(
                    batch_idx, batch_nb,
                    progress_bar(batch_idx, batch_nb),
                    eta(start, batch_idx, batch_nb),
                    running_avg_loss, cont_loss.item())
                print(msg, end="\r")
                dist.barrier()
            # evaluate model on validation set
            if self.args.rank == 0:
                val_loss = self.evaluate(msg)
                if val_loss < best_loss:
                    best_loss = val_loss
                    self.save_model(epoch, val_loss)

                print("Epoch {} took {} - Final loss : {:.4f} - val loss : "
                      "{:.4f}".format(epoch, user_friendly_time(time_since(start)),
                                      running_avg_loss,
                                      val_loss))

    def evaluate(self, msg):
        val_loader = get_loader(self.args.val_features,
                                shuffle=False,
                                args=self.args)
        val_batch_nb = len(val_loader)
        val_losses = []
        self.model.eval()
        for i, batch, in enumerate(val_loader, start=1):
            batch = self.process_batch(batch)
            inputs = dict()
            for k, v in batch.items():
                inputs[k] = v.to(self.args.gpu)

            with torch.no_grad():
                loss = self.model(**inputs)[0]
            msg2 = "{} =>   Evaluating : {}/{}".format(msg, i, val_batch_nb)
            print(msg2, end="\r")
            val_losses.append(loss.item())
            val_loss = np.mean(val_losses)

        return val_loss

    # save model
    def save_model(self, epoch, loss):
        model_to_save = self.model.module if hasattr(
            self.model, "module") else self.model

        ckpt = {
            "args": self.args,
            "state_dict": model_to_save.t5_model.state_dict(),
        }
        model_save_path = os.path.join(self.model_dir,
                                       "{}_{:.4f}".format(epoch, loss))
        torch.save(ckpt, model_save_path)
