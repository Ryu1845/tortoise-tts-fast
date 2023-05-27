import os
from glob import glob
from typing import Dict, List

import librosa
import librosa.util as librosa_util
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from librosa.util import pad_center, tiny
from scipy.io.wavfile import read
from scipy.signal import get_window
from torch.autograd import Variable


def window_sumsquare(
    window,
    n_frames,
    hop_length=200,
    win_length=800,
    n_fft=800,
    dtype=np.float32,
    norm=None,
):
    """
    # from librosa 0.6
    Compute the sum-square envelope of a window function at a given hop length.

    This is used to estimate modulation effects induced by windowing
    observations in short-time fourier transforms.

    Parameters
    ----------
    window : string, tuple, number, callable, or list-like
        Window specification, as in `get_window`

    n_frames : int > 0
        The number of analysis frames

    hop_length : int > 0
        The number of samples to advance between frames

    win_length : [optional]
        The length of the window function.  By default, this matches `n_fft`.

    n_fft : int > 0
        The length of each analysis frame.

    dtype : np.dtype
        The data type of the output

    Returns
    -------
    wss : np.ndarray, shape=`(n_fft + hop_length * (n_frames - 1))`
        The sum-squared envelope of the window function
    """
    if win_length is None:
        win_length = n_fft

    n = n_fft + hop_length * (n_frames - 1)
    x = np.zeros(n, dtype=dtype)

    # Compute the squared window at the desired length
    win_sq = get_window(window, win_length, fftbins=True)
    win_sq = librosa_util.normalize(win_sq, norm=norm) ** 2
    win_sq = librosa_util.pad_center(win_sq, n_fft)

    # Fill the envelope
    for i in range(n_frames):
        sample = i * hop_length
        x[sample : min(n, sample + n_fft)] += win_sq[: max(0, min(n_fft, n - sample))]
    return x


class STFT(torch.nn.Module):
    """adapted from Prem Seetharaman's https://github.com/pseeth/pytorch-stft"""

    def __init__(
        self, filter_length=800, hop_length=200, win_length=800, window="hann"
    ):
        super(STFT, self).__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.forward_transform = None
        scale = self.filter_length / self.hop_length
        fourier_basis = np.fft.fft(np.eye(self.filter_length))

        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack(
            [np.real(fourier_basis[:cutoff, :]), np.imag(fourier_basis[:cutoff, :])]
        )

        forward_basis = torch.FloatTensor(fourier_basis[:, None, :])
        inverse_basis = torch.FloatTensor(
            np.linalg.pinv(scale * fourier_basis).T[:, None, :]
        )

        if window is not None:
            assert filter_length >= win_length
            # get window and zero center pad it to filter_length
            fft_window = get_window(window, win_length, fftbins=True)
            fft_window = pad_center(fft_window, size=filter_length)
            fft_window = torch.from_numpy(fft_window).float()

            # window the bases
            forward_basis *= fft_window
            inverse_basis *= fft_window

        self.register_buffer("forward_basis", forward_basis.float())
        self.register_buffer("inverse_basis", inverse_basis.float())

    def transform(self, input_data):
        num_batches = input_data.size(0)
        num_samples = input_data.size(1)

        self.num_samples = num_samples

        # similar to librosa, reflect-pad the input
        input_data = input_data.view(num_batches, 1, num_samples)
        input_data = F.pad(
            input_data.unsqueeze(1),
            (int(self.filter_length / 2), int(self.filter_length / 2), 0, 0),
            mode="reflect",
        )
        input_data = input_data.squeeze(1)

        forward_transform = F.conv1d(
            input_data,
            Variable(self.forward_basis, requires_grad=False),
            stride=self.hop_length,
            padding=0,
        )

        cutoff = int((self.filter_length / 2) + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]

        magnitude = torch.sqrt(real_part**2 + imag_part**2)
        phase = torch.autograd.Variable(torch.atan2(imag_part.data, real_part.data))

        return magnitude, phase

    def inverse(self, magnitude, phase):
        recombine_magnitude_phase = torch.cat(
            [magnitude * torch.cos(phase), magnitude * torch.sin(phase)], dim=1
        )

        inverse_transform = F.conv_transpose1d(
            recombine_magnitude_phase,
            Variable(self.inverse_basis, requires_grad=False),
            stride=self.hop_length,
            padding=0,
        )

        if self.window is not None:
            window_sum = window_sumsquare(
                self.window,
                magnitude.size(-1),
                hop_length=self.hop_length,
                win_length=self.win_length,
                n_fft=self.filter_length,
                dtype=np.float32,
            )
            # remove modulation effects
            approx_nonzero_indices = torch.from_numpy(
                np.where(window_sum > tiny(window_sum))[0]
            )
            window_sum = torch.autograd.Variable(
                torch.from_numpy(window_sum), requires_grad=False
            )
            window_sum = window_sum.cuda() if magnitude.is_cuda else window_sum
            inverse_transform[:, :, approx_nonzero_indices] /= window_sum[
                approx_nonzero_indices
            ]

            # scale by hop ratio
            inverse_transform *= float(self.filter_length) / self.hop_length

        inverse_transform = inverse_transform[:, :, int(self.filter_length / 2) :]
        inverse_transform = inverse_transform[:, :, : -int(self.filter_length / 2) :]

        return inverse_transform

    def forward(self, input_data):
        self.magnitude, self.phase = self.transform(input_data)
        reconstruction = self.inverse(self.magnitude, self.phase)
        return reconstruction


BUILTIN_VOICES_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "../voices"
)


def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    if data.dtype == np.int32:
        norm_fix = 2**31
    elif data.dtype == np.int16:
        norm_fix = 2**15
    elif data.dtype == np.float16 or data.dtype == np.float32:
        norm_fix = 1.0
    else:
        raise NotImplementedError(f"Provided data dtype not supported: {data.dtype}")
    return (torch.FloatTensor(data.astype(np.float32)) / norm_fix, sampling_rate)


def check_audio(audio, audiopath: str):
    # Check some assumptions about audio range. This should be automatically fixed in load_wav_to_torch, but might not be in some edge cases, where we should squawk.
    # '2' is arbitrarily chosen since it seems like audio will often "overdrive" the [-1,1] bounds.
    if torch.any(audio > 2) or not torch.any(audio < 0):
        print(f"Error with {audiopath}. Max={audio.max()} min={audio.min()}")
    audio.clip_(-1, 1)


def read_audio_file(audiopath: str):
    if audiopath[-4:] == ".wav":
        audio, lsr = load_wav_to_torch(audiopath)
    elif audiopath[-4:] == ".mp3":
        audio, lsr = librosa.load(audiopath, sr=None)
        audio = torch.FloatTensor(audio)
    else:
        assert False, f"Unsupported audio format provided: {audiopath[-4:]}"

    # Remove any channel data.
    if len(audio.shape) > 1:
        if audio.shape[0] < 5:
            audio = audio[0]
        else:
            assert audio.shape[1] < 5
            audio = audio[:, 0]

    return audio, lsr


def load_required_audio(audiopath: str):
    audio, lsr = read_audio_file(audiopath)

    audios = [
        torchaudio.functional.resample(audio, lsr, sampling_rate)
        for sampling_rate in (22050, 24000)
    ]
    for audio in audios:
        check_audio(audio, audiopath)

    return [audio.unsqueeze(0) for audio in audios]


def load_audio(audiopath, sampling_rate):
    audio, lsr = read_audio_file(audiopath)

    if lsr != sampling_rate:
        audio = torchaudio.functional.resample(audio, lsr, sampling_rate)
    check_audio(audio, audiopath)

    return audio.unsqueeze(0)


TACOTRON_MEL_MAX = 2.3143386840820312
TACOTRON_MEL_MIN = -11.512925148010254


def denormalize_tacotron_mel(norm_mel):
    return ((norm_mel + 1) / 2) * (
        TACOTRON_MEL_MAX - TACOTRON_MEL_MIN
    ) + TACOTRON_MEL_MIN


def normalize_tacotron_mel(mel):
    return 2 * ((mel - TACOTRON_MEL_MIN) / (TACOTRON_MEL_MAX - TACOTRON_MEL_MIN)) - 1


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression(x, C=1):
    """
    PARAMS
    ------
    C: compression factor used to compress
    """
    return torch.exp(x) / C


def get_voices(extra_voice_dirs: List[str] = []):
    dirs = [BUILTIN_VOICES_DIR] + extra_voice_dirs
    voices: Dict[str, List[str]] = {}
    for d in dirs:
        subs = os.listdir(d)
        for sub in subs:
            subj = os.path.join(d, sub)
            if os.path.isdir(subj):
                voices[sub] = (
                    list(glob(f"{subj}/*.wav"))
                    + list(glob(f"{subj}/*.mp3"))
                    + list(glob(f"{subj}/*.pth"))
                )
    return voices


def load_voice(voice: str, extra_voice_dirs: List[str] = []):
    if voice == "random":
        return None, None

    voices = get_voices(extra_voice_dirs)
    paths = voices[voice]
    if len(paths) == 1 and paths[0].endswith(".pth"):
        return None, torch.load(paths[0])
    else:
        conds = []
        for cond_path in paths:
            c = load_required_audio(cond_path)
            conds.append(c)
        return conds, None


def load_voices(voices: List[str], extra_voice_dirs: List[str] = []):
    latents = []
    clips = []
    for voice in voices:
        if voice == "random":
            if len(voices) > 1:
                print(
                    "Cannot combine a random voice with a non-random voice. Just using a random voice."
                )
            return None, None
        clip, latent = load_voice(voice, extra_voice_dirs)
        if latent is None:
            assert (
                len(latents) == 0
            ), "Can only combine raw audio voices or latent voices, not both. Do it yourself if you want this."
            clips.extend(clip)
        elif clip is None:
            assert (
                len(clips) == 0
            ), "Can only combine raw audio voices or latent voices, not both. Do it yourself if you want this."
            latents.append(latent)
    if len(latents) == 0:
        return clips, None
    else:
        latents_0 = torch.stack([l[0] for l in latents], dim=0).mean(dim=0)
        latents_1 = torch.stack([l[1] for l in latents], dim=0).mean(dim=0)
        latents = (latents_0, latents_1)
        return None, latents


class TacotronSTFT:
    def __init__(
        self,
        filter_length=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=80,
        sampling_rate=22050,
        mel_fmin=0.0,
        mel_fmax=8000.0,
    ):
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate
        self.stft_fn = STFT(filter_length, hop_length, win_length)
        from librosa.filters import mel as librosa_mel_fn

        mel_basis = librosa_mel_fn(
            sr=sampling_rate,
            n_fft=filter_length,
            n_mels=n_mel_channels,
            fmin=mel_fmin,
            fmax=mel_fmax,
        )
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)

    def spectral_normalize(self, magnitudes):
        output = dynamic_range_compression(magnitudes)
        return output

    def spectral_de_normalize(self, magnitudes):
        output = dynamic_range_decompression(magnitudes)
        return output

    def mel_spectrogram(self, y):
        """Computes mel-spectrograms from a batch of waves
        PARAMS
        ------
        y: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]

        RETURNS
        -------
        mel_output: torch.FloatTensor of shape (B, n_mel_channels, T)
        """
        assert torch.min(y.data) >= -10
        assert torch.max(y.data) <= 10
        y = torch.clip(y, min=-1, max=1)

        magnitudes, phases = self.stft_fn.transform(y)
        magnitudes = magnitudes.data
        mel_output = torch.matmul(self.mel_basis, magnitudes)
        mel_output = self.spectral_normalize(mel_output)
        return mel_output


def wav_to_univnet_mel(wav, do_normalization=False, device="cuda"):
    stft = TacotronSTFT(1024, 256, 1024, 100, 24000, 0, 12000)
    stft = stft.to(device)
    mel = stft.mel_spectrogram(wav)
    if do_normalization:
        mel = normalize_tacotron_mel(mel)
    return mel
