from model.CMGAN.actor import TSCNet, TSCNetSmall
from model.critic import Discriminator
import os
from data import dataset as dataloader
import torch.nn.functional as F
import torch
from utils import batch_pesq, power_compress, power_uncompress, original_pesq, K_way_CrossEntropy
import logging
from torchinfo import summary
import argparse
from collections import OrderedDict


import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import wandb

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=120, help="number of epochs of training")
parser.add_argument("--parallel", action='store_true', help="Set this falg to run parallel gpu training.")
parser.add_argument("--gpu", action='store_true', help="Set this falg to run single gpu training.")
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--exp", type=str, default='default', help='Experiment name')
parser.add_argument("--suffix", type=str, default='', help='Experiment suffix name')

parser.add_argument("-pt", "--ckpt", type=str, required=False, default=None,
                        help="Path to saved cmgan checkpoint for resuming training.")
parser.add_argument("--small", action='store_true', required=False, 
                    help="set this flag to train small cmgan.")
parser.add_argument("--mag_only", action='store_true', required=False, 
                    help="set this flag to train using magnitude only.")
parser.add_argument("--pretrain_init", action='store_true', required=False, 
                    help="set this flag to init model with pretrainied weights.")
parser.add_argument("--wandb", action='store_true', required=False, 
                    help="set this flag to log using wandb.")
parser.add_argument("--log_interval", type=int, default=500)
parser.add_argument("--accum_grad", type=int, default=4)
parser.add_argument("--decay_epoch", type=int, default=12, help="epoch from which to start lr decay")
parser.add_argument("--init_lr", type=float, default=5e-4, help="initial learning rate")
parser.add_argument("--cut_len", type=int, default=16000*2, help="cut length, default is 2 seconds in denoise "
                                                                 "and dereverberation")
parser.add_argument("--data_dir", type=str, required=True,
                    help="dir of VCTK+DEMAND dataset")
parser.add_argument("--save_model_dir", type=str, required=True,
                    help="dir of saved model")
parser.add_argument("--loss_weights", nargs='+', type=float, default=[0.3, 0.7, 0.01, 1],
                    help="weights of RI components, magnitude, time loss, and Metric Disc")
args = parser.parse_args()
logging.basicConfig(level=logging.INFO)


def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


class Trainer:
    def __init__(self, train_ds, test_ds, batchsize, log_wandb=False, parallel=False, gpu_id=None, accum_grad=1, resume_pt=None):
        
        self.n_fft = 400
        self.hop = 100
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.ACCUM_GRAD = accum_grad
    
        self.model = TSCNet(num_channel=64, 
                            num_features=self.n_fft // 2 + 1, 
                            gpu_id=gpu_id)
        self.batchsize = batchsize
        
        self.log_wandb = log_wandb
        self.gpu_id = gpu_id
        self.discriminator = Discriminator(ndf=16)

        self.ce_loss = K_way_CrossEntropy()

        if gpu_id is not None:
            self.model = self.model.to(gpu_id)
            self.discriminator = self.discriminator.to(gpu_id)

        #optimizers and schedulers
        self.optimizer = torch.optim.AdamW(filter(lambda layer:layer.requires_grad,self.model.parameters()), 
                                           lr=args.init_lr)
        self.optimizer_disc = torch.optim.AdamW(
            self.discriminator.parameters(), lr=2 * args.init_lr
        )
        self.scheduler_G = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=args.decay_epoch, gamma=0.5
        )
        self.scheduler_D = torch.optim.lr_scheduler.StepLR(
            self.optimizer_disc, step_size=args.decay_epoch, gamma=0.5
        )

        self.start_epoch = 0
        if resume_pt is not None:
            if not resume_pt.endswith('.pt'):
                raise ValueError("Incorrect path to the checkpoint..")
            try:
                name = resume_pt[:-3]
                epoch = name.split('_')[-1]
                self.start_epoch = int(epoch)
            except Exception:
                self.start_epoch = 0
            self.load_checkpoint(resume_pt)

        if parallel:
            self.model = DDP(self.model, device_ids=[gpu_id])
            self.discriminator = DDP(self.discriminator, device_ids=[gpu_id])
        
        if log_wandb:
            wandb.login()
            wandb.init(project=args.exp)

    def forward_generator_step(self, clean, noisy):

        # Normalization
        c = torch.sqrt(noisy.size(-1) / torch.sum((noisy**2.0), dim=-1))
        noisy, clean = torch.transpose(noisy, 0, 1), torch.transpose(clean, 0, 1)
        noisy, clean = torch.transpose(noisy * c, 0, 1), torch.transpose(
            clean * c, 0, 1
        )

        noisy_spec = torch.stft(
            noisy,
            self.n_fft,
            self.hop,
            window=torch.hamming_window(self.n_fft).to(self.gpu_id),
            onesided=True,
        )
        clean_spec = torch.stft(
            clean,
            self.n_fft,
            self.hop,
            window=torch.hamming_window(self.n_fft).to(self.gpu_id),
            onesided=True,
        )


        noisy_spec = power_compress(noisy_spec).permute(0, 1, 3, 2)
        clean_spec = power_compress(clean_spec)
        clean_real = clean_spec[:, 0, :, :].unsqueeze(1)
        clean_imag = clean_spec[:, 1, :, :].unsqueeze(1)

        est_real, est_imag = self.model(noisy_spec)
        
        est_real, est_imag = est_real.permute(0, 1, 3, 2), est_imag.permute(0, 1, 3, 2)
        est_mag = torch.sqrt(est_real**2 + est_imag**2)
        clean_mag = torch.sqrt(clean_real**2 + clean_imag**2)

        est_spec_uncompress = power_uncompress(est_real, est_imag).squeeze(1)
        est_audio = torch.istft(
            est_spec_uncompress,
            self.n_fft,
            self.hop,
            window=torch.hamming_window(self.n_fft).to(self.gpu_id),
            onesided=True,
        )

        return {
            "est_real": est_real,
            "est_imag": est_imag,
            "est_mag": est_mag,
            "clean_real": clean_real,
            "clean_imag": clean_imag,
            "clean_mag": clean_mag,
            "est_audio": est_audio, 
        }
    
    def load_checkpoint(self, path):
        try:
            state_dict = torch.load(path, map_location=torch.device(self.gpu_id))
            self.model.load_state_dict(state_dict['generator_state_dict'])
            self.discriminator.load_state_dict(state_dict['discriminator_state_dict'])
            self.optimizer.load_state_dict(state_dict['optimizer_G_state_dict'])
            self.optimizer_disc.load_state_dict(state_dict['optimizer_D_state_dict'])
            self.scheduler_G.load_state_dict(state_dict['scheduler_G_state_dict'])
            self.scheduler_D.load_state_dict(state_dict['scheduler_D_state_dict'])
            print(f"Loaded checkpoint saved at {path} starting at epoch {self.start_epoch}")
            del state_dict
            
        except Exception as e:
            state_dict = torch.load(path, map_location=torch.device(self.gpu_id))
            
            gen_state_dict = OrderedDict()
            for name, params in state_dict['generator_state_dict'].items():
                name = name[7:]
                gen_state_dict[name] = params        
            self.model.load_state_dict(gen_state_dict)
            del gen_state_dict
            
            disc_state_dict = OrderedDict()
            for name, params in state_dict['discriminator_state_dict'].items():
                name = name[7:]
                disc_state_dict[name] = params
            self.discriminator.load_state_dict(disc_state_dict)
            del disc_state_dict
            
            self.optimizer.load_state_dict(state_dict['optimizer_G_state_dict'])
            self.optimizer_disc.load_state_dict(state_dict['optimizer_D_state_dict'])
            self.scheduler_G.load_state_dict(state_dict['scheduler_G_state_dict'])
            self.scheduler_D.load_state_dict(state_dict['scheduler_D_state_dict'])
            
            print(f"Loaded checkpoint saved at {path} starting at epoch {self.start_epoch}")
            del state_dict
    
    def save_model(self, path_root, exp, epoch, pesq):
        """
        Save model at path_root
        """
        checkpoint_prefix = f"{exp}_PESQ_{pesq}_epoch_{epoch}.pt"
        path = os.path.join(path_root, exp)
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, checkpoint_prefix)
        if self.gpu_id == 0:
            save_dict = {'generator_state_dict':self.model.state_dict(), 
                        'discriminator_state_dict':self.discriminator.state_dict(),
                        'optimizer_G_state_dict':self.optimizer.state_dict(),
                        'optimizer_D_state_dict':self.optimizer_disc.state_dict(),
                        'scheduler_G_state_dict':self.scheduler_G.state_dict(),
                        'scheduler_D_state_dict':self.scheduler_D.state_dict(),
                        'epoch':epoch,
                        'pesq':pesq
                        }
            
            torch.save(save_dict, path)
            print(f"checkpoint:{checkpoint_prefix} saved at {path}")

    def calculate_generator_loss(self, generator_outputs):

        predict_fake_metric = self.discriminator(
            generator_outputs["clean_mag"], generator_outputs["est_mag"]
        )
        predict_fake_metric = torch.argmax(predict_fake_metric, dim=-1)


        gen_loss_GAN = F.mse_loss(
            predict_fake_metric.flatten(), generator_outputs["one_labels"].float()
        )

        loss_mag = F.mse_loss(
            generator_outputs["est_mag"], generator_outputs["clean_mag"]
        ) 

        loss_ri = F.mse_loss(
            generator_outputs["est_real"], generator_outputs["clean_real"]
        ) + F.mse_loss(generator_outputs["est_imag"], generator_outputs["clean_imag"])

        time_loss = torch.mean(
            torch.abs(generator_outputs["est_audio"] - generator_outputs["clean"])
        )

        loss = (
            args.loss_weights[0] * loss_ri #0.3
            + args.loss_weights[1] * loss_mag #0.7
            + args.loss_weights[2] * time_loss #1
            + args.loss_weights[3] * gen_loss_GAN #0.01
        )
        
        return loss

    def calculate_discriminator_loss(self, generator_outputs):

        length = generator_outputs["est_audio"].size(-1)
        est_audio_list = list(generator_outputs["est_audio"].detach().cpu().numpy())
        clean_audio_list = list(generator_outputs["clean"].cpu().numpy()[:, :length])
        pesq_mask, pesq_score = batch_pesq(clean_audio_list, est_audio_list)

        if self.gpu_id is not None:
            pesq_score = pesq_score.to(self.gpu_id)
            pesq_mask = pesq_mask.to(self.gpu_id)
      
        # The calculation of PESQ can be None due to silent part
        if pesq_score is not None:
            predict_enhance_metric = self.discriminator(
                generator_outputs["clean_mag"], generator_outputs["est_mag"].detach()
            )
            predict_max_metric = self.discriminator(
                generator_outputs["clean_mag"], generator_outputs["clean_mag"]
            )
            discrim_loss_metric = F.mse_loss(
                predict_max_metric.flatten(), generator_outputs["one_labels"]
            ) + F.mse_loss(predict_enhance_metric.flatten() * pesq_mask, pesq_score * pesq_mask)
        else:
            discrim_loss_metric = None
            pesq_score = torch.tensor([0.0])

        return discrim_loss_metric, pesq_score.mean()

    def train_step(self, step, batch):
        # Trainer generator
        clean = batch[0].to(self.gpu_id)
        noisy = batch[1].to(self.gpu_id)
        one_labels = torch.ones(clean.shape[0]).to(self.gpu_id)
        
        #Run generator
        generator_outputs = self.forward_generator_step(
            clean,
            noisy,
        )
        generator_outputs["one_labels"] = one_labels
        generator_outputs["clean"] = clean

        loss = self.calculate_generator_loss(generator_outputs)
        #print(f'Check Loss:{loss.sum()}, {torch.isnan(loss).any()}, {torch.isinf(loss).any()}')
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            return None, None
     
        loss = loss / self.ACCUM_GRAD

        self.optimizer.zero_grad()
        loss.backward()
        if step % self.ACCUM_GRAD == 0 or step == len(self.train_ds):
            self.optimizer.step()

        # Train Discriminator
        discrim_loss_metric, pesq = self.calculate_discriminator_loss(generator_outputs)

        if discrim_loss_metric is not None:
            discrim_loss_metric = discrim_loss_metric / self.ACCUM_GRAD
            self.optimizer_disc.zero_grad()
            discrim_loss_metric.backward()
            if step % self.ACCUM_GRAD == 0 or step == len(self.train_ds):
                self.optimizer_disc.step()
        else:
            discrim_loss_metric = torch.tensor([0.0])

        wandb.log({
            'step_gen_loss':loss,
            'step_disc_loss':discrim_loss_metric,
            'step_train_pesq':original_pesq(pesq)
        })
        print(f"G_LOSS:{loss} | D_LOSS:{discrim_loss_metric}")

        return loss.item(), discrim_loss_metric.item()

    @torch.no_grad()
    def test_step(self, batch):

        clean = batch[0].to(self.gpu_id)
        noisy = batch[1].to(self.gpu_id)
        one_labels = torch.ones(clean.shape[0]).to(self.gpu_id)

        generator_outputs = self.forward_generator_step(
            clean,
            noisy,
        )
        generator_outputs["one_labels"] = one_labels
        generator_outputs["clean"] = clean

        loss = self.calculate_generator_loss(generator_outputs)

        discrim_loss_metric, pesq = self.calculate_discriminator_loss(generator_outputs)
        if discrim_loss_metric is None:
            discrim_loss_metric = torch.tensor([0.0])
        

        return loss, discrim_loss_metric, pesq

    def test(self):
        self.model.eval()
        self.discriminator.eval()
        gen_loss_total = 0.0
        disc_loss_total = 0.0
        val_pesq = 0.0
        for idx, batch in enumerate(self.test_ds):
            step = idx + 1
            try:
                loss, disc_loss, pesq = self.test_step(batch)
            except Exception as e:
                print(e)
                continue
            
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                continue
            gen_loss_total += loss
            disc_loss_total += disc_loss
            val_pesq += pesq
        gen_loss_avg = gen_loss_total / step
        disc_loss_avg = disc_loss_total / step
        val_pesq = val_pesq / step

        template = "GPU: {}, Generator loss: {}, Discriminator loss: {}"
        logging.info(template.format(self.gpu_id, gen_loss_avg, disc_loss_avg))

        return gen_loss_avg, disc_loss_avg, val_pesq

    def train(self):
        scheduler_G = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=args.decay_epoch, gamma=0.5
        )
        scheduler_D = torch.optim.lr_scheduler.StepLR(
            self.optimizer_disc, step_size=args.decay_epoch, gamma=0.5
        )
        for epoch in range(self.start_epoch+1, args.epochs):
            self.model.train()
            self.discriminator.train()
            for idx, batch in enumerate(self.train_ds):
                clean, noisy, _ = batch
                if torch.isnan(clean).any() or torch.isnan(noisy).any():
                    continue
                if torch.isinf(clean).any() or torch.isinf(noisy).any():
                    continue
                step = idx + 1
                try:
                    loss, disc_loss = self.train_step(step, batch)
                except Exception as e:
                    import traceback
                    print(traceback.format_exc())
                    continue
                template = "GPU: {}, Epoch {}, Step {}, loss: {}, disc_loss: {}"
                if (step % args.log_interval) == 0:
                    logging.info(
                        template.format(self.gpu_id, epoch, step, loss, disc_loss)
                    )
                
            gen_loss, disc_loss, val_pesq = self.test()
            wandb.log({
                'val_gen_loss':gen_loss,
                'val_disc_loss':disc_loss,
                'val_pesq':original_pesq(val_pesq),
                'Epoch':epoch
            })
            
            self.save_model(path_root=args.save_model_dir,
                            exp=f"{args.exp}_{args.suffix}",
                            epoch=epoch,
                            pesq=original_pesq(val_pesq))
            scheduler_G.step()
            scheduler_D.step()


def main(rank: int, world_size: int, args):
    ddp_setup(rank, world_size)
    if rank == 0:
        print(args)
        available_gpus = [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ]
        print(available_gpus)

    train_ds, test_ds = dataloader.load_data(
            args.data_dir, args.batch_size, 1, args.cut_len, gpu=False,
        )


    print(f"Train:{len(train_ds)}, Validation:{len(test_ds)}")
    trainer = Trainer(train_ds=train_ds, 
                      test_ds=test_ds, 
                      batchsize=args.batch_size, 
                      parallel=args.parallel, 
                      gpu_id=rank,
                      accum_grad=args.accum_grad, 
                      resume_pt=args.ckpt,
                      log_wandb=args.wandb)
    trainer.train()
    destroy_process_group()


if __name__ == "__main__":
    ARGS = args

    output = f"{ARGS.save_model_dir}/{ARGS.exp}_{ARGS.suffix}"
    os.makedirs(output, exist_ok=True)

    world_size = torch.cuda.device_count()
    print(f"World size:{world_size}")
    if ARGS.parallel:
        mp.spawn(main, args=(world_size, ARGS), nprocs=world_size)
    else:
        if ARGS.gpu:
            main(0, world_size, ARGS)
        else:
            main(None, world_size, ARGS)