
import torch
import torch.nn as nn

from dataset.dataset import load_data
from models.reward_model import JNDModel, power_compress
import argparse
import os
from tqdm import tqdm
from sklearn.metrics import classification_report

def ARGS():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--root", type=str, required=True,
                        help="Root directory to JND Dataset.")
    parser.add_argument("-c", "--comp", type=str, required=True,
                        help="Root directory to JND Dataset comparision lists.")
    parser.add_argument("-pt", "--ckpt", type=str, required=False, default=None,
                        help="Path to saved cmgan checkpoint for resuming training.")
    parser.add_argument("--gpu", action='store_true',
                        help="Set this flag for single gpu training.")
    parser.add_argument("--norm", type=str, required=False, default='sbn',
                        help="option, choose between 'ln(layernorm) / sbn(batchnorm)'")
    parser.add_argument("--enc", type=int, required=False, default=1,
                        help="encoding option, choose between 1 or 2")
    parser.add_argument("--loss", type=str, required=False, default='featureloss',
                        help="option, choose between featureloss/attentionloss")
    parser.add_argument("--cut_len", type=int, default=40000, help="cut length")
    return parser
  

class Evaluation:
    def __init__(self, checkpoint, gpu_id=None):
        self.model = JNDModel(in_channels=2,
                              out_dim=2, 
                              n_layers=7, 
                              keep_prob=0.7, 
                              norm_type=args.norm, 
                              sum_till=7, 
                              gpu_id=gpu_id,
                              enc_type=args.enc,
                              loss_type=args.loss)
        self.n_fft = 400
        self.hop = 100
        
        if gpu_id == None:
            dev = 'cpu'
        else:
            dev = gpu_id
            self.model = self.model.to(gpu_id)
        state_dict = self.load(checkpoint, dev)
        print(self.model)
        print(state_dict['model_state_dict'])
        self.model.load_state_dict(state_dict['model_state_dict'])
        print(f"Loaded save checkpoint at epoch:{state_dict['epoch']} with val_acc:{state_dict['val_acc']}")
        self.model.eval()
        self.criterion = nn.CrossEntropyLoss(reduction='mean')
        self.gpu_id=gpu_id
        
    def get_specs(self, clean, noisy):
        """
        Create spectrograms from input waveform.
        ARGS:
            clean : clean waveform (batch * cut_len)
            noisy : noisy waveform (batch * cut_len)

        Return
            noisy_spec : (b * 2 * f * t) noisy spectrogram
            clean_spec : (b * 2 * f * t) clean spectrogram
        """
        # Normalization
        c = torch.sqrt(noisy.size(-1) / torch.sum((noisy**2.0), dim=-1))
        noisy, clean = torch.transpose(noisy, 0, 1), torch.transpose(clean, 0, 1)
        noisy, clean = torch.transpose(noisy * c, 0, 1), torch.transpose(
            clean * c, 0, 1
        )
        
        win = torch.hamming_window(self.n_fft)
        if self.gpu_id is not None:
            win = win.to(self.gpu_id)

        noisy_spec = torch.stft(
            noisy,
            self.n_fft,
            self.hop,
            window=win,
            onesided=True,
        )
        clean_spec = torch.stft(
            clean,
            self.n_fft,
            self.hop,
            window=win,
            onesided=True,
        )

        noisy_spec = power_compress(noisy_spec)
        clean_spec = power_compress(clean_spec)

        return noisy_spec, clean_spec
        
    def load(self, path, device):
        if device == 'cpu':
            dev = torch.device('cpu')
        else:
            dev = torch.device(f'cuda:{device}')
        state_dict = torch.load(path, map_location=dev)
        return state_dict
    
    def forward_step(self, batch):
        wav_in, wav_out, labels = batch
        class_probs = self.model(wav_in, wav_out)
        loss = self.criterion(class_probs, labels)
        return loss, class_probs
        
    def predict(self, dataset):
        PREDS=[]
        LABELS=[]
    
        with torch.no_grad():
            for batch in tqdm(dataset):
                wav_in, wav_out, labels = batch
                if self.gpu_id is not None:
                    wav_in = wav_in.to(self.gpu_id)
                    wav_out = wav_out.to(self.gpu_id)
                    labels = labels.to(self.gpu_id)

                wav_in, wav_out = self.get_specs(wav_in, wav_out)
                batch = (wav_in, wav_out, labels)
                _, probs = self.forward_step(batch)
                y_preds = torch.argmax(probs, dim=-1)
                labels = torch.argmax(labels, dim=-1)
                print(f"PREDS:{y_preds}")
                print(f"LABELS:{labels}")
                print(f"ACC:{self.accuracy(y_preds, labels)}")

                y_preds = y_preds.detach().cpu().numpy().tolist()
                labels = labels.detach().cpu().numpy().tolist()
                PREDS.extend(y_preds)
                LABELS.extend(labels)
            
        return PREDS, LABELS
    
    def accuracy(self, y_pred, y_true):
        score = (y_pred == y_true).float()
        return score.mean()
    
    def score(self, labels, preds):
        print(classification_report(labels, preds))


def main(args):
    train_ds, test_ds = load_data(root=args.root, 
                                      path_root=args.comp,
                                      batch_size=16, 
                                      n_cpu=1,
                                      split_ratio=0.85, 
                                      cut_len=args.cut_len,
                                      resample=True,
                                      parallel=False)

    if args.gpu:
        eval = Evaluation(args.ckpt, 0)
    else:
        eval = Evaluation(args.ckpt, 'None')

    preds, labels = eval.predict(test_ds)
    eval.score(preds, labels)

if __name__=='__main__':
    args = ARGS().parse_args()
    main(args)