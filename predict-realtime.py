from argparse import ArgumentParser
import time
import yaml
import numpy as np
from numpy.lib.stride_tricks import as_strided
import torch
import torch.nn as nn
import sounddevice as sd
from multiprocessing import Pool, Manager, TimeoutError

d_len = 0
q_out =None
in_buffer = np.zeros(0, np.float32)
model = None
p = None
xx = None
yy = None


class Replicator(nn.Module):
    def __init__(self) -> None:
        super(Replicator,self).__init__()
        self.lstm0 = nn.LSTM(input_size=1, hidden_size=6, batch_first=True, num_layers=2)
      
    def forward(self,x):
        tx = torch.from_numpy(x.astype(np.float32)).clone()
        tx,_ = self.lstm0(tx)
        return tx


model = Replicator()
model.share_memory()
model.load_state_dict(torch.load("best_result.pth"))
model.eval()


def sliding_window(x, window, slide):
    n_slide = (len(x) - window) // slide
    remain = (len(x) - window) % slide
    clopped = x[:-remain]
    return as_strided(clopped, shape=(n_slide + 1, window), strides=(slide * 4, 4))

def processing(indata, q , model):

    d_len = len(indata)

    padded = np.concatenate((
        np.zeros(prepad, np.float32),
        indata))

    x = sliding_window(padded, input_timesteps, output_timesteps)
    x = x[:, :, np.newaxis]
    xx = x.copy()
    yy = model(xx.astype(np.float32))
    yy = yy.detach().numpy()

    time.sleep(0.0)

    q.put(yy[:, -output_timesteps:, -1:].reshape(-1)[:d_len])

    return

def callback(in_data, out_data, frames, time, status):

    print("\r\033[1Aq_size", q_out.qsize())
    out_data[: ,0] = q_out.get()

    global in_buffer
    global model

    in_buffer = np.concatenate([in_buffer, in_data[: ,0]])

    if in_buffer.shape[0] == block_size:

        try:
            res= p.apply_async(processing, args=(in_buffer, q_out ,model ,))  # Create new process
            res.get(timeout=0)
        except TimeoutError as err:
            pass

        in_buffer = np.zeros(0, np.float32)  # Empty the input buffer


def main():

    args = parse_args()

    with open(args.config_file) as fp:
        config = yaml.safe_load(fp)

    parser = ArgumentParser()

    global input_timesteps
    global output_timesteps
    global batch_size
    global block_size
    global prepad
    global q_out
    global p


    input_timesteps = config["input_timesteps"]
    output_timesteps = config["output_timesteps"]
    batch_size = config["batch_size"]

    block_size = output_timesteps * batch_size
    prepad = input_timesteps - output_timesteps


    CHUNK= block_size
    RATE=48000

    p = Pool(processes=4)
    m = Manager()
    q_out = m.Queue(maxsize=0)

    time.sleep(1)
    print("prepare start")

    prefill = np.zeros(block_size, np.float32)

    for i in range(4):
        prefill = np.zeros(block_size, np.float32)
        p.apply(processing, args=(prefill, q_out, model ,))
        
    print("prepare done. wait 1sec")
    time.sleep(1)

    
    try:
        with sd.Stream(device=1,
                samplerate=RATE, blocksize=CHUNK,
                dtype=np.float32,
                channels=1,
                callback=callback,
                prime_output_buffers_using_stream_callback=True):
            print('#' * 80)
            print('press Return to quit')
            print('#' * 80)
            print(" ")
            input()
    except KeyboardInterrupt:
        parser.exit(message='')
        p.close()
    except Exception as e:
        parser.exit(message=type(e).__name__ + ': ' + str(e))
    



def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--config_file", "-c", default="./config.yml",
        help="configuration file (*.yml)")
    parser.add_argument(
        "--input_file", "-i",
        help="input wave file (48kHz/mono, *.wav)")
    parser.add_argument(
        "--output_file", "-o", default="./predicted-by-nnp.wav",
        help="output wave file (48kHz/mono, *.wav)")
    parser.add_argument(
        "--model_file", "-m",
        help="input model file (*.h5)")
    return parser.parse_args()

if __name__ == '__main__':
    main()
