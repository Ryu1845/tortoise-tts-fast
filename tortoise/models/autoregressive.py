# AGPL: a notification must be added stating that changes have been made to that file.
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gpt2 import GPT2Config, GPT2Model


def null_position_embeddings(range, dim):
    return torch.zeros((range.shape[0], range.shape[1], dim), device=range.device)


def _p(t):
    return t and (len(t), len(t[0]), t[0][0].shape)  # kv_cache debug


class GPT2InferenceModel(nn.Module):
    def __init__(self, config, gpt, text_pos_emb, embeddings, norm, linear, kv_cache):
        super().__init__()
        self.config = config
        self.transformer = gpt
        self.text_pos_embedding = text_pos_emb
        self.embeddings = embeddings
        self.lm_head = nn.Sequential(norm, linear)
        self.kv_cache = kv_cache

    def store_mel_emb(self, mel_emb):
        self.cached_mel_emb = mel_emb

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)  # usually None
        if not self.kv_cache:
            past_key_values = None
        # only last token for inputs_ids if past is defined in kwargs
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        assert self.cached_mel_emb is not None
        assert inputs_embeds is None  # Not supported by this inference model.
        assert labels is None  # Training not supported by this inference model.

        # Create embedding
        mel_len = self.cached_mel_emb.shape[1]
        if input_ids.shape[1] != 1:
            text_inputs = input_ids[:, mel_len:]
            text_emb = self.embeddings(text_inputs)
            text_emb = text_emb + self.text_pos_embedding(text_emb)
            if self.cached_mel_emb.shape[0] != text_emb.shape[0]:
                mel_emb = self.cached_mel_emb.repeat_interleave(
                    text_emb.shape[0] // self.cached_mel_emb.shape[0], 0
                )
            else:  # this outcome only occurs once per loop in most cases
                mel_emb = self.cached_mel_emb
            emb = torch.cat([mel_emb, text_emb], dim=1)
        else:
            emb = self.embeddings(input_ids)
            emb = emb + self.text_pos_embedding.get_fixed_embedding(
                attention_mask.shape[1] - mel_len, attention_mask.device
            )

        transformer_outputs = self.transformer(
            inputs_embeds=emb,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states)

        return (lm_logits,) + transformer_outputs[1:]

    @staticmethod
    def _reorder_cache(past, beam_idx):
        """
        This function is used to re-order the :obj:`past_key_values` cache if
        :meth:`~transformers.PreTrainedModel.beam_search` or :meth:`~transformers.PreTrainedModel.beam_sample` is
        called. This is required to match :obj:`past_key_values` with the correct beam_idx at every generation step.
        """
        return tuple(
            tuple(
                past_state.index_select(0, beam_idx.to(past_state.device))
                for past_state in layer_past
            )
            for layer_past in past
        )

    def generate(
        self,
        input_ids,
        repetition_penalty=2.0,
        temperature=0.2,
        top_p=0.8,
        eos_token_id=None,
        pad_token_id=None,
        max_length=250,
        **kwargs,
    ):
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = (
            torch.tensor(eos_token_id).to(input_ids.device)
            if eos_token_id is not None
            else None
        )
        unfinished_sequences = torch.ones(
            input_ids.shape[0], dtype=torch.long, device=input_ids.device
        )
        while True:
            model_inputs = self.prepare_inputs_for_generation(input_ids)
            # forward pass to get next token
            logits = self(**model_inputs)[0]
            scores = logits[:, -1, :]

            # top p
            sorted_logits, sorted_indices = torch.sort(scores, descending=False)
            cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
            # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
            sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
            # scatter sorted tensors to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            scores = scores.masked_fill(indices_to_remove, -float("Inf"))

            # temperature
            scores = scores / temperature

            # repetition penalty
            score = torch.gather(scores, 1, input_ids)
            # if score < 0 then repetition penalty has to be multiplied to reduce the previous token probability
            score = torch.where(
                score < 0, score * repetition_penalty, score / repetition_penalty
            )
            scores.scatter_(1, input_ids, score)

            # sample
            probs = nn.functional.softmax(scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError(
                        "If `eos_token_id` is defined, make sure that `pad_token_id` is defined."
                    )
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (
                    1 - unfinished_sequences
                )
            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

            if eos_token_id_tensor is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1)
                    .ne(eos_token_id_tensor.unsqueeze(1))
                    .prod(dim=0)
                )

                # stop when each sentence is finished
                if unfinished_sequences.max() == 0:
                    break
            if input_ids.shape[-1] >= max_length:
                break
        return input_ids


class LearnedPositionEmbeddings(nn.Module):
    def __init__(self, seq_len, model_dim, init=0.02):
        super().__init__()
        self.emb = nn.Embedding(seq_len, model_dim)
        # Initializing this way is standard for GPT-2
        self.emb.weight.data.normal_(mean=0.0, std=init)

    def forward(self, x):
        sl = x.shape[1]
        return self.emb(torch.arange(0, sl, device=x.device))

    def get_fixed_embedding(self, ind, dev):
        return self.emb(torch.arange(0, ind, device=dev))[ind - 1 : ind]


class UnifiedVoice(nn.Module):
    def __init__(
        self,
        layers=8,
        model_dim=512,
        heads=8,
        max_text_tokens=120,
        max_mel_tokens=250,
        max_conditioning_inputs=1,
        mel_length_compression=1024,
        number_text_tokens=256,
        start_text_token=None,
        number_mel_codes=8194,
        start_mel_token=8192,
        stop_mel_token=8193,
        train_solo_embeddings=False,
        use_mel_codes_as_input=True,
        checkpointing=True,
        types=1,
    ):
        """
        Args:
            layers: Number of layers in transformer stack.
            model_dim: Operating dimensions of the transformer
            heads: Number of transformer heads. Must be divisible by model_dim. Recommend model_dim//64
            max_text_tokens: Maximum number of text tokens that will be encountered by model.
            max_mel_tokens: Maximum number of MEL tokens that will be encountered by model.
            max_conditioning_inputs: Maximum number of conditioning inputs provided to the model. If (1), conditioning input can be of format (b,80,s), otherwise (b,n,80,s).
            mel_length_compression: The factor between <number_input_samples> and <mel_tokens>. Used to compute MEL code padding given wav input length.
            number_text_tokens:
            start_text_token:
            stop_text_token:
            number_mel_codes:
            start_mel_token:
            stop_mel_token:
            train_solo_embeddings:
            use_mel_codes_as_input:
            checkpointing:
        """
        super().__init__()

        self.number_text_tokens = number_text_tokens
        self.start_text_token = (
            number_text_tokens * types if start_text_token is None else start_text_token
        )
        self.stop_text_token = 0
        self.number_mel_codes = number_mel_codes
        self.start_mel_token = start_mel_token
        self.stop_mel_token = stop_mel_token
        self.layers = layers
        self.heads = heads
        self.max_mel_tokens = max_mel_tokens
        self.max_text_tokens = max_text_tokens
        self.model_dim = model_dim
        self.max_conditioning_inputs = max_conditioning_inputs
        self.mel_length_compression = mel_length_compression
        self.text_embedding = nn.Embedding(
            self.number_text_tokens * types + 1, model_dim
        )
        self.mel_embedding = nn.Embedding(self.number_mel_codes, model_dim)

        max_mel_seq_len = self.max_mel_tokens + 2 + self.max_conditioning_inputs
        max_text_seq_len = self.max_text_tokens + 2
        gpt_config = GPT2Config(
            vocab_size=256,  # Unused.
            n_positions=max_mel_seq_len + max_text_seq_len,
            n_embd=model_dim,
            n_layer=layers,
            n_head=heads,
            use_cache=not checkpointing,
        )
        gpt = GPT2Model(gpt_config)
        # Override the built in positional embeddings
        del (
            gpt.wpe
        )  # TODO: figure out relevance in fixing exported model definition: Embedding(1012, 1024)
        gpt.wpe = functools.partial(null_position_embeddings, dim=model_dim)
        # Built-in token embeddings are unused.
        del gpt.wte
        self.gpt = gpt
        self.mel_pos_embedding = LearnedPositionEmbeddings(max_mel_seq_len, model_dim)
        self.text_pos_embedding = LearnedPositionEmbeddings(max_text_seq_len, model_dim)
        self.mel_layer_pos_embedding = None
        self.text_layer_pos_embedding = None
        self.mel_solo_embedding = 0
        self.text_solo_embedding = 0

        self.final_norm = nn.LayerNorm(model_dim)
        self.text_head = nn.Linear(model_dim, self.number_text_tokens * types + 1)
        self.mel_head = nn.Linear(model_dim, self.number_mel_codes)

        # Initialize the embeddings per the GPT-2 scheme
        embeddings = [self.text_embedding]
        if use_mel_codes_as_input:
            embeddings.append(self.mel_embedding)
        for module in embeddings:
            module.weight.data.normal_(mean=0.0, std=0.02)

    def post_init_gpt2_config(self, kv_cache=True):
        seq_length = self.max_mel_tokens + self.max_text_tokens + 2
        gpt_config = GPT2Config(
            vocab_size=self.max_mel_tokens,
            n_positions=seq_length,
            n_embd=self.model_dim,
            n_layer=self.layers,
            n_head=self.heads,
            use_cache=True,
        )
        self.inference_model = GPT2InferenceModel(
            gpt_config,
            self.gpt,
            self.mel_pos_embedding,
            self.mel_embedding,
            self.final_norm,
            self.mel_head,
            kv_cache=kv_cache,
        )
        # self.inference_model = PrunedGPT2InferenceModel(gpt_config, self.gpt, self.mel_pos_embedding, self.mel_embedding, self.final_norm, self.mel_head)
        self.gpt.wte = self.mel_embedding

    def build_aligned_inputs_and_targets(self, input, start_token, stop_token):
        inp = F.pad(input, (1, 0), value=start_token)
        tar = F.pad(input, (0, 1), value=stop_token)
        return inp, tar

    def set_mel_padding(self, mel_input_tokens, wav_lengths):
        """
        Given mel tokens that are derived from a padded audio clip and the actual lengths of each batch element in
        that audio clip, reformats the tokens with STOP_MEL_TOKEN in place of the zero padding. This is required
        preformatting to create a working TTS model.
        """
        # Set padding areas within MEL (currently it is coded with the MEL code for <zero>).
        mel_lengths = torch.div(
            wav_lengths, self.mel_length_compression, rounding_mode="trunc"
        )
        for b in range(len(mel_lengths)):
            actual_end = (
                mel_lengths[b] + 1
            )  # Due to the convolutional nature of how these tokens are generated, it would be best if the model predicts a token past the actual last token.
            if actual_end < mel_input_tokens.shape[-1]:
                mel_input_tokens[b, actual_end:] = self.stop_mel_token
        return mel_input_tokens

    def get_logits(
        self,
        speech_conditioning_inputs,
        first_inputs,
        first_head,
        second_inputs=None,
        second_head=None,
        get_attns=False,
        return_latent=False,
    ):
        emb = torch.cat(
            [speech_conditioning_inputs, first_inputs, second_inputs], dim=1
        )

        last_hidden_state = self.gpt(inputs_embeds=emb)[0]

        enc = last_hidden_state[
            :, 1:
        ]  # The first logit is tied to the speech_conditioning_input
        enc = self.final_norm(enc)

        if return_latent:
            return (
                enc[
                    :,
                    speech_conditioning_inputs.shape[
                        1
                    ] : speech_conditioning_inputs.shape[1]
                    + first_inputs.shape[1],
                ],
                enc[:, -second_inputs.shape[1] :],
            )

    def forward(
        self,
        speech_conditioning_latent,
        text_inputs,
        text_lengths,
        mel_codes,
        wav_lengths,
        types=None,
        text_first=True,
        raw_mels=None,
        return_attentions=False,
        return_latent=False,
        clip_inputs=True,
    ):
        """
        Forward pass that uses both text and voice in either text conditioning mode or voice conditioning mode
        (actuated by `text_first`).

        speech_conditioning_input: MEL float tensor, (b,1024)
        text_inputs: long tensor, (b,t)
        text_lengths: long tensor, (b,)
        mel_inputs:  long tensor, (b,m)
        wav_lengths: long tensor, (b,)
        raw_mels: MEL float tensor (b,80,s)

        If return_attentions is specified, only logits are returned.
        If return_latent is specified, loss & logits are not computed or returned. Only the predicted latents are returned.
        If clip_inputs is True, the inputs will be clipped to the smallest input size across each input modality.
        """
        if clip_inputs:
            # This model will receive micro-batches with a ton of padding for both the text and MELs. Ameliorate this by
            # chopping the inputs by the maximum actual length.
            max_text_len = text_lengths.max()
            text_inputs = text_inputs[:, :max_text_len]
            max_mel_len = wav_lengths.max() // self.mel_length_compression
            mel_codes = mel_codes[:, :max_mel_len]
            if raw_mels is not None:
                raw_mels = raw_mels[:, :, : max_mel_len * 4]
        mel_codes = self.set_mel_padding(mel_codes, wav_lengths)
        text_inputs = F.pad(text_inputs, (0, 1), value=self.stop_text_token)
        mel_codes = F.pad(mel_codes, (0, 1), value=self.stop_mel_token)

        conds = speech_conditioning_latent.unsqueeze(1)
        text_inputs, text_targets = self.build_aligned_inputs_and_targets(
            text_inputs, self.start_text_token, self.stop_text_token
        )
        text_emb = self.text_embedding(text_inputs) + self.text_pos_embedding(
            text_inputs
        )
        mel_codes, mel_targets = self.build_aligned_inputs_and_targets(
            mel_codes, self.start_mel_token, self.stop_mel_token
        )
        mel_inp = mel_codes
        mel_emb = self.mel_embedding(mel_inp)
        mel_emb = mel_emb + self.mel_pos_embedding(mel_codes)

        text_logits, mel_logits = self.get_logits(
            conds,
            text_emb,
            self.text_head,
            mel_emb,
            self.mel_head,
            return_latent=return_latent,
        )
        if return_latent:
            return mel_logits[
                :, :-2
            ]  # Despite the name, these are not logits. Strip off the two tokens added by this forward pass.

    def inference_speech(
        self,
        speech_conditioning_latent,
        text_inputs,
        input_tokens=None,
        num_return_sequences=1,
        max_generate_length=None,
        typical_sampling=False,
        typical_mass=0.9,
        **hf_generate_kwargs,
    ):
        text_inputs = F.pad(text_inputs, (0, 1), value=self.stop_text_token)
        text_inputs, text_targets = self.build_aligned_inputs_and_targets(
            text_inputs, self.start_text_token, self.stop_text_token
        )
        text_emb = self.text_embedding(text_inputs) + self.text_pos_embedding(
            text_inputs
        )

        conds = speech_conditioning_latent.unsqueeze(1)
        emb = torch.cat([conds, text_emb], dim=1)
        self.inference_model.store_mel_emb(emb)

        fake_inputs = torch.full(
            (
                emb.shape[0],
                conds.shape[1] + emb.shape[1],
            ),
            fill_value=1,
            dtype=torch.long,
            device=text_inputs.device,
        )
        fake_inputs[:, -1] = self.start_mel_token
        trunc_index = fake_inputs.shape[1]
        inputs = fake_inputs

        max_length = (
            trunc_index + self.max_mel_tokens - 1
            if max_generate_length is None
            else trunc_index + max_generate_length
        )
        gen = self.inference_model.generate(
            inputs,
            bos_token_id=self.start_mel_token,
            pad_token_id=self.stop_mel_token,
            eos_token_id=self.stop_mel_token,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            **hf_generate_kwargs,
        )
        return gen[:, trunc_index:]


if __name__ == "__main__":
    gpt = UnifiedVoice(
        model_dim=256,
        heads=4,
        train_solo_embeddings=True,
        use_mel_codes_as_input=True,
        max_conditioning_inputs=4,
    )
    l = gpt(
        torch.randn(2, 3, 80, 800),
        torch.randint(high=120, size=(2, 120)),
        torch.tensor([32, 120]),
        torch.randint(high=8192, size=(2, 250)),
        torch.tensor([250 * 256, 195 * 256]),
    )
    gpt.text_forward(
        torch.randn(2, 80, 800),
        torch.randint(high=50, size=(2, 80)),
        torch.tensor([32, 80]),
    )
