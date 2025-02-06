print("Starting the imports...")
import argparse
import os
import random

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_only
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader

import MIDI
from midi_model import MIDIModel
from midi_tokenizer import MIDITokenizer
tokenizer = MIDITokenizer()
import sys 

EXTENSION = [".mid", ".midi"]
# removes an error... 
torch.set_float32_matmul_precision('medium')
# helps with memory management 
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"



def file_ext(fname):
    return os.path.splitext(fname)[1].lower()


class MidiDataset(Dataset):
    def __init__(self, midi_list, tokenizer: MIDITokenizer, max_len=2048, min_file_size=3000, max_file_size=384000,
                 aug=True, check_quality=False, rand_start=True):

        self.tokenizer = tokenizer
        self.midi_list = midi_list
        self.max_len = max_len
        self.min_file_size = min_file_size
        self.max_file_size = max_file_size
        self.aug = aug
        self.check_quality = check_quality
        self.rand_start = rand_start

    def __len__(self):
        return len(self.midi_list)

    def load_midi(self, index):
        path = self.midi_list[index]
        try:
            with open(path, 'rb') as f:
                datas = f.read()
            if len(datas) > self.max_file_size:  # large midi file will spend too much time to load
                raise ValueError("file too large")
            elif len(datas) < self.min_file_size:
                raise ValueError("file too small")
            mid = MIDI.midi2score(datas)
            if max([0] + [len(track) for track in mid[1:]]) == 0:
                raise ValueError("empty track")
            mid = self.tokenizer.tokenize(mid)
            if self.check_quality and not self.tokenizer.check_quality(mid)[0]:
                raise ValueError("bad quality")
            if self.aug:
                mid = self.tokenizer.augment(mid)
        except Exception:
            mid = self.load_midi(random.randint(0, self.__len__() - 1))
        return mid

    def __getitem__(self, index):
        mid = self.load_midi(index)
        mid = np.asarray(mid, dtype=np.int16)
        # if mid.shape[0] < self.max_len:
        #     mid = np.pad(mid, ((0, self.max_len - mid.shape[0]), (0, 0)),
        #                  mode="constant", constant_values=self.tokenizer.pad_id)
        if self.rand_start:
            start_idx = random.randrange(0, max(1, mid.shape[0] - self.max_len))
            start_idx = random.choice([0, start_idx])
        else:
            max_start = max(1, mid.shape[0] - self.max_len)
            start_idx = (index*(max_start//8)) % max_start
        mid = mid[start_idx: start_idx + self.max_len]
        mid = mid.astype(np.int64)
        mid = torch.from_numpy(mid)
        return mid


def collate_fn(batch):
    max_len = max([len(mid) for mid in batch])
    batch = [F.pad(mid, (0, 0, 0, max_len - mid.shape[0]), mode="constant", value=tokenizer.pad_id) for mid in batch]
    batch = torch.stack(batch)
    return batch


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


class TrainMIDIModel(MIDIModel):
    def __init__(self, tokenizer: MIDITokenizer, n_layer=12, n_head=16, n_embd=1024, n_inner=4096, flash=False,
                 lr=2e-4, weight_decay=0.01, warmup=1e3, max_step=1e6, sample_seq=False, sample_dir="./sample/"):
        super(TrainMIDIModel, self).__init__(tokenizer=tokenizer, n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                                             n_inner=n_inner, flash=flash)
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup = warmup
        self.max_step = max_step
        self.sample_seq = sample_seq
        self.sample_dir = sample_dir # write samples here

    def configure_optimizers(self):
        param_optimizer = list(self.named_parameters())
        no_decay = ['bias', 'norm']  # no decay for bias and Norm
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': self.weight_decay},
            {
                'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        optimizer = optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.lr,
            betas=(0.9, 0.99),
            eps=1e-08,
        )
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.warmup,
            num_training_steps=self.max_step,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1
            }
        }

    def training_step(self, batch, batch_idx):
        x = batch[:, :-1].contiguous()  # (batch_size, midi_sequence_length, token_sequence_length)
        y = batch[:, 1:].contiguous()
        hidden = self.forward(x)
        if self.sample_seq:  # to reduce vram
            rand_idx = [-1] + random.sample(list(range(y.shape[1] - 2)), min(127, (y.shape[1] - 2) // 2))
            hidden = hidden[:, rand_idx]
            y = y[:, rand_idx]
        hidden = hidden.reshape(-1, hidden.shape[-1])
        y = y.reshape(-1, y.shape[-1])  # (batch_size*midi_sequence_length, token_sequence_length)
        x = y[:, :-1]
        logits = self.forward_token(hidden, x)
        loss = F.cross_entropy(
            logits.view(-1, self.tokenizer.vocab_size),
            y.view(-1),
            reduction="mean",
            ignore_index=self.tokenizer.pad_id
        )
        self.log("train/loss", loss)
        self.log("train/lr", self.lr_schedulers().get_last_lr()[0])
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[:, :-1].contiguous()  # (batch_size, midi_sequence_length, token_sequence_length)
        y = batch[:, 1:].contiguous()
        hidden = self.forward(x)
        hidden = hidden.reshape(-1, hidden.shape[-1])
        y = y.reshape(-1, y.shape[-1])  # (batch_size*midi_sequence_length, token_sequence_length)
        x = y[:, :-1]
        logits = self.forward_token(hidden, x)
        loss = F.cross_entropy(
            logits.view(-1, self.tokenizer.vocab_size),
            y.view(-1),
            reduction="mean",
            ignore_index=self.tokenizer.pad_id
        )
        self.log("val/loss", loss, sync_dist=True)
        return loss

    def on_validation_start(self):
        # mac version
        torch.mps.empty_cache()
        # torch.cuda.empty_cache()

    def on_validation_end(self):
        @rank_zero_only
        def gen_example():
            # generate randomly from the model 
            mid = self.generate()
            mid = self.tokenizer.detokenize(mid)
            img = self.tokenizer.midi2img(mid)
            img.save(f"{self.sample_dir}{self.global_step}_0.png")
            with open(f"{self.sample_dir}{self.global_step}_0.mid", 'wb') as f:
                f.write(MIDI.score2midi(mid))

            # generate a continuation from a randomly chosen fole 
            prompt = val_dataset.load_midi(random.randint(0, len(val_dataset) - 1))
            prompt = np.asarray(prompt, dtype=np.int16)
            ori = prompt[:512]
            prompt = prompt[:256].astype(np.int64)
            mid = self.generate(prompt, max_len=64) # 64 is a short gen
            mid = self.tokenizer.detokenize(mid)
            img = self.tokenizer.midi2img(mid)
            img.save(f"{self.sample_dir}{self.global_step}_1.png")
            img = self.tokenizer.midi2img(self.tokenizer.detokenize(ori))
            img.save(f"{self.sample_dir}{self.global_step}_1_ori.png")
            with open(f"{self.sample_dir}{self.global_step}_1.mid", 'wb') as f:
                f.write(MIDI.score2midi(mid))
        
        # print(f"Generation at validation step disabled for memory saving...")

        # try:
        gen_example()
        # except Exception as e:
        #     print(e)
        
        torch.mps.empty_cache()
        
        # torch.cuda.empty_cache()


def get_midi_list(path):
    all_files = {
        os.path.join(root, fname)
        for root, _dirs, files in os.walk(path)
        for fname in files
    }
    all_midis = sorted(
        fname for fname in all_files if file_ext(fname) in EXTENSION
    )
    return all_midis


if __name__ == '__main__':
    print("Parsing the args")
    parser = argparse.ArgumentParser()
    # model args
    parser.add_argument(
        "--resume", type=str, default="", help="resume training from ckpt"
    )
    parser.add_argument(
        "--ckpt", type=str, default="", help="load ckpt"
    )

    # dataset args
    parser.add_argument(
        "--data", type=str, required=True, help="dataset path"
    )
    parser.add_argument(
        "--data-val-split",
        type=int,
        default=128,
        help="split length for validation",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=2048,
        help="max seq length for training",
    )
    parser.add_argument(
        "--quality", action="store_true", default=False, help="check dataset quality"
    )

    # training args
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="weight decay")
    parser.add_argument("--warmup-step", type=int, default=1e2, help="warmup step")
    parser.add_argument("--max-step", type=int, default=1e6, help="max training step")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="gradient clip val")
    parser.add_argument(
        "--sample-seq", action="store_true", default=False, help="sample midi seq to reduce vram"
    )
    parser.add_argument(
        "--batch-size-train", type=int, default=1, help="batch size for training"
    )
    parser.add_argument(
        "--batch-size-val", type=int, default=1, help="batch size for val"
    )
    parser.add_argument(
        "--workers-train",
        type=int,
        default=1,
        help="workers num for training dataloader",
    )
    parser.add_argument(
        "--workers-val",
        type=int,
        default=1,
        help="workers num for validation dataloader",
    )
    parser.add_argument(
        "--acc-grad", type=int, default=2, help="gradient accumulation"
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="gpu",
        choices=["cpu", "gpu", "tpu", "ipu", "hpu", "auto"],
        help="accelerator",
    )
    parser.add_argument("--devices", type=int, default=-1, help="devices num")
    parser.add_argument(
        "--fp32", action="store_true", default=False, help="disable mix precision"
    )
    parser.add_argument(
        "--disable-benchmark", action="store_true", default=False, help="disable cudnn benchmark"
    )
    parser.add_argument(
        "--log-step", type=int, default=1, help="log training loss every n steps"
    )
    parser.add_argument(
        "--val-step", type=int, default=500, help="valid and save every n steps"
    )
    parser.add_argument(
        "--sample-dir", type=str, default="samples/", help="save samples to this directory"
    )

    opt = parser.parse_args()
    print(opt)

    if not os.path.exists("lightning_logs"):
        os.mkdir("lightning_logs")
    if not os.path.exists("sample"):
        os.mkdir("sample")
    if not os.path.exists(opt.sample_dir):
        os.mkdir(opt.sample_dir)

    # sys.exit()

    pl.seed_everything(opt.seed)
    print("---load dataset---")
    tokenizer = MIDITokenizer()

    assert os.path.exists(opt.data), f"Data folder not found: {opt.data}"
    midi_list = get_midi_list(opt.data)
    assert len(midi_list) > 0, f"Failed to find any midi files in data folder {opt.data}"
    print(f"got a list of midi files {len(midi_list)}")
    random.shuffle(midi_list)
    full_dataset_len = len(midi_list)
    train_dataset_len = full_dataset_len - opt.data_val_split
    assert train_dataset_len > 0, f"No training data - total dataset {full_dataset_len} but you want {opt.data_val_split}"
    train_midi_list = midi_list[:train_dataset_len]
    val_midi_list = midi_list[train_dataset_len:]
    train_dataset = MidiDataset(train_midi_list, tokenizer, max_len=opt.max_len, aug=True, check_quality=opt.quality, rand_start=True)
    val_dataset = MidiDataset(val_midi_list, tokenizer, max_len=opt.max_len, aug=False, check_quality=opt.quality, rand_start=False)
    print(f"Got training set {len(train_dataset)} val dataset {len(val_dataset)}")

    train_dataloader = DataLoader(
        train_dataset,
        # batch_size=opt.batch_size_train,
        batch_size=2,
        shuffle=True,
        persistent_workers=True,
        num_workers=2, 
        # num_workers=opt.workers_train,
        pin_memory=True,
        collate_fn=collate_fn
    )
    val_dataloader = DataLoader(
        val_dataset,
        # batch_size=opt.batch_size_val,
        batch_size=1,
        shuffle=False,
        persistent_workers=True,
        num_workers=opt.workers_val,
        pin_memory=True,
        collate_fn=collate_fn
    )
    print(f"train: {len(train_dataset)}  val: {len(val_dataset)}. LOADING MODEL")
    model = TrainMIDIModel(tokenizer, flash=True, lr=opt.lr, weight_decay=opt.weight_decay,
                           warmup=opt.warmup_step, max_step=opt.max_step, sample_seq=opt.sample_seq, sample_dir=opt.sample_dir)
    
    # saves memory but makes it slower apparently 
    # model.gradient_checkpointing_enable()
    # 
    print(f"Model loaded")
    if opt.ckpt:
        ckpt = torch.load(opt.ckpt, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        model.load_state_dict(state_dict, strict=False)
    print("---start train---")
    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
        filename="epoch={epoch},loss={val/loss:.4f}",
    )
    
    callbacks = [checkpoint_callback]

    trainer = Trainer(
        # precision=32 if opt.fp32 else 16,
        precision=32, 
        accumulate_grad_batches=opt.acc_grad,
        gradient_clip_val=opt.grad_clip,
        accelerator="mps",
        devices=opt.devices,
        max_steps=opt.max_step,
        benchmark=not opt.disable_benchmark,
        log_every_n_steps=1,
#        strategy="ddp",
        strategy="auto", # for mac
        # val_check_interval=None, # within an epoch, how often to save
        # check_val_every_n_epoch=1, # save every n epochs
        check_val_every_n_epoch=None, # for small datasets
        val_check_interval=opt.val_step,

        callbacks=callbacks,
    )
    ckpt_path = opt.resume
    if ckpt_path == "":
        ckpt_path = None
    else:
        print(f"---loading from checkpoint---\n{ckpt_path}\m")
    print("---start train---")
    # trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=ckpt_path)
    trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=ckpt_path)
