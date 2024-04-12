import numpy as np
from model.actor import TSCNet, TSCNetSmall 
from natsort import natsorted
import os
from compute_metrics import compute_metrics
from utils import *
import torchaudio
import soundfile as sf
import argparse
from tqdm import tqdm


@torch.no_grad()
def enhance_one_track(
    model, audio_path, saved_dir, cut_len, n_fft=400, hop=100, save_tracks=False
):
    name = os.path.split(audio_path)[-1]
    noisy, sr = torchaudio.load(audio_path)
    assert sr == 16000
    noisy = noisy.cuda()

    c = torch.sqrt(noisy.size(-1) / torch.sum((noisy**2.0), dim=-1))
    noisy = torch.transpose(noisy, 0, 1)
    noisy = torch.transpose(noisy * c, 0, 1)

    length = noisy.size(-1)
    frame_num = int(np.ceil(length / 100))
    padded_len = frame_num * 100
    padding_len = padded_len - length
    noisy = torch.cat([noisy, noisy[:, :padding_len]], dim=-1)
    if padded_len > cut_len:
        batch_size = int(np.ceil(padded_len / cut_len))
        while 100 % batch_size != 0:
            batch_size += 1
        noisy = torch.reshape(noisy, (batch_size, -1))

    noisy_spec = torch.stft(
        noisy, n_fft, hop, window=torch.hamming_window(n_fft).cuda(), onesided=True
    )
    noisy_spec = power_compress(noisy_spec).permute(0, 1, 3, 2)
    est_real, est_imag = model(noisy_spec)
    est_real, est_imag = est_real.permute(0, 1, 3, 2), est_imag.permute(0, 1, 3, 2)

    est_spec_uncompress = power_uncompress(est_real, est_imag).squeeze(1)
    est_audio = torch.istft(
        est_spec_uncompress,
        n_fft,
        hop,
        window=torch.hamming_window(n_fft).cuda(),
        onesided=True,
    )
    est_audio = est_audio / c
    est_audio = torch.flatten(est_audio)[:length].cpu().numpy()
    assert len(est_audio) == length, f"est_audio:{len(est_audio)} ref_audio:{length}"
    if save_tracks:
        saved_path = os.path.join(saved_dir, name)
        sf.write(saved_path, est_audio, sr)

    return est_audio, length


def evaluation(model_path, noisy_dir, clean_dir, cutlen, save_tracks, saved_dir, pre, small, dist):
    n_fft = 400
    if not dist:
        dist = None
    else:
        dist = 'Normal'
    if small:
        model = TSCNetSmall(num_channel=64, 
                            num_features=n_fft // 2 + 1, 
                            distribution=dist,
                            gpu_id=0,
                            eval=True).cuda()
    else:
        model = TSCNet(num_channel=64, 
                        num_features=n_fft // 2 + 1, 
                        distribution=dist,
                        gpu_id=0, 
                        eval=True).cuda()
    
    if pre:
        try:
            model.load_state_dict(torch.load(model_path)['generator_state_dict'])
        except KeyError as e:
            model.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path)['actor_state_dict'])

    if not os.path.exists(saved_dir):
        os.mkdir(saved_dir)
    
    model = model.eval()

    audio_list = os.listdir(noisy_dir)
    audio_list = natsorted(audio_list)
    num = len(audio_list)
    print(f"Parsed {num} audios in {noisy_dir}...")
    metrics_total = np.zeros(7)
    for audio in tqdm(audio_list):
        try:
            with torch.no_grad():
                noisy_path = os.path.join(noisy_dir, audio)
                clean_path = os.path.join(clean_dir, audio)
                #est_audio, length = enhance_one_track(
                #    model, noisy_path, saved_dir, 16000 * 10, n_fft, n_fft // 4, save_tracks
                #)
                est_audio, length = enhance_one_track(
                    model, noisy_path, saved_dir, cutlen, n_fft, n_fft // 4, save_tracks
                )
                clean_audio, sr = sf.read(clean_path)
                assert sr == 16000
                metrics = compute_metrics(clean_audio, est_audio, sr, 0)
                metrics = np.array(metrics)
                metrics_total += metrics
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            continue

    metrics_avg = metrics_total / num
    print(
        "pesq: ",
        metrics_avg[0],
        "csig: ",
        metrics_avg[1],
        "cbak: ",
        metrics_avg[2],
        "covl: ",
        metrics_avg[3],
        "ssnr: ",
        metrics_avg[4],
        "stoi: ",
        metrics_avg[5],
        "si-sdr: ",
        metrics_avg[6],
    )


parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default='./best_ckpt/ckpt_80',
                    help="the path where the model is saved")
parser.add_argument("--test_dir", type=str, default='dir to your VCTK-DEMAND test dataset',
                    help="noisy tracks dir to be enhanced")
parser.add_argument("--save_tracks", type=str, default=True, help="save predicted tracks or not")
parser.add_argument("--out_dist", action='store_true', help="toggle to test models that output nomral dist.")
parser.add_argument("--pre", action='store_true', help="toggle to test pretrained models")
parser.add_argument("--small", action='store_true', help="toggle to test small cmgan models")
parser.add_argument("--cutlen", type=int, default=16 * 16000, help="length of signal to be passed to model. ")
parser.add_argument("--save_dir", type=str, default='./saved_tracks_best', help="where enhanced tracks to be saved")

args = parser.parse_args()


if __name__ == "__main__":
    noisy_dir = os.path.join(args.test_dir, "noisy")
    clean_dir = os.path.join(args.test_dir, "clean")
    evaluation(args.model_path, noisy_dir, clean_dir, args.cutlen, args.save_tracks, args.save_dir, args.pre, args.small, args.out_dist)
