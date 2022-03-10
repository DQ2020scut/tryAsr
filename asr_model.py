from collections import  defaultdict
from typing import  List,Optional,Tuple

import torch

from torch.nn.utils.rnn import pad_sequence

from tryasr.transformer.cmvn import GlobalCMVN
from tryasr.transformer.ctc import CTC
from tryasr.transformer.decoder import (TransformerDecoder, BitransformerDecoder)
from tryasr.transformer.encoder import ConformerEncoder
from tryasr.transformer.encoder import TransformerEncoder
from tryasr.transformer.label_smoothing_loss import LabelSmoothingLoss
from tryasr.utils.cmvn import load_cmvn
from tryasr.utils.common import (IGNORE_ID, add_sos_eos, log_add,
                                remove_duplicates_and_blank, th_accuracy,
                                reverse_pad_list)
wenet.utils.mask import (make_pad_mask, mask_finished_preds,
                              mask_finished_scores, subsequent_mask)

class ASRModel(torch.nn.Module):
    # CTC-attention hybrid Encoder-Decoder model
    def __init__(
            self,
            vocab_size: int,
            encoder: TransformerEncoder,
            decoder: TransformerDecoder,
            ctc: CTC,
            ctc_weight: float = 0.5,
            ignore_id: int = IGNORE_ID,
            reverse_weight: float = 0.0,
            lsm_weight: float = 0.0,
            length_normalized_loss: bool = False,
                 ):
        assert 0.0 <= ctc_weight <= 1.0, ctc_weight

        super().__init__()
        # 注意eos is the same as sos (equivalent ID)
        self.sos = vocab_size - 1
        self.eos = vocab_size - 1
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.ctc_weight = ctc_weight
        self.reverse_weight = reverse_weight
        self.encoder = encoder
        self.decoder = decoder
        self.ctc = ctc
        self.criterion_att = LabelSmoothingLoss(
            size = vocab_size,
            padding_idx = ignore_id,
            smoothing = lsm_weight,
            normalize_length = length_normalized_loss,
        )

        def forward(
            self,
            speech: torch.Tensor,
            speech_lengths: torch.Tensor,
            text: torch.Tensor,
            text_lengths: torch.Tensor,
        ) -> Tuple[Optional[torch.Tensor],Optional[torch.Tensor],
                   Optional[torch.Tensor]]:
            '''
            Frontend + Encoder + Decoder + Calc loss

            Args:
                speech: (Batch, Length,...)
                speech_lengths: (Batch, )
                text: (Batch, Length)
                text_lengths: (Batch, )
            '''
            assert text_lengths.dim() == 1, text_lengths.shape
            # Check that batch_size is unified
            assert (speech.shape[0] == speech_lengths.shape[0] == text.shape[0] == text_lengths.shape[0]),\
                (speech.shape, speech_lengths.shape, text.shape, text_lengths.shape)

            # 1. Encoder
            encoder_out, encoder_mask = self.encoder(speech, speech_lengths)
            encoder_out_lens = encoder_mask.squeeze(1).sum(1)

            # 2a. Attention-decoder branch
            if self.ctc_weight != 1.0 :
                loss_att, acc_att = self._calc_att_loss(encoder_out, encoder_mask, text, text_lengths)
            else:
                loss_att = None
            # 2b. CTC branch
            if self.ctc_weight != 0.0:
                loss_ctc = self.ctc(encoder_out, encoder_out_lens, text, text_lengths)
            else:
                loss_ctc = None

            if loss_ctc is None:
                loss = loss_att
            elif loss_att is None:
                loss = loss_ctc
            else:
                loss = loss = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att
            return loss, loss_att, loss_ctc

    def _calc_att_loss(
            self,
            encoder_out: torch.Tensor,
            encoder_mask: torch.Tensor,
            ys_pad: torch.Tensor,
            ys_pad_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos,
                                            self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        #reverse the seq, used for right to left decoder
        r_ys_pad = reverse_pad_list(ys_pad, ys_pad_lens, float(self.ignore_id))
        r_ys_in_pad, r_ys_out_pad = add_sos_eos(r_ys_pad, self.sos, self.eos, self.ignore_id)

        # 1. Forward decoder
        decoder_out, r_decoder_out, _ =self.decoder(encoder_out, encoder_mask,
                                                    ys_in_pad, ys_in_lens,
                                                    r_ys_in_pad,
                                                    self.reverse_weight)
        # 2.Compute attention loss
        loss_att=self.criterion_att(decoder_out, ys_out_pad)
        r_loss_att = torch.tensor(0.0)
        if self.reverse_weight > 0.0:
            r_loss_att = self.criterion_att(r_decoder_out, r_ys_out_pad)
        loss_att = loss_att * (1 - self.reverse_weight) + r_loss_att * self.reverse_weight
        acc_att = th_accuracy(
            decoder_out.view(-1, self.vocab_size),
            ys_out_pad,
            ignore_label = self.ignore_id,
        )
        return loss_att, acc_att

    def _forward_encoder(
            self,
            speech: torch.Tensor,
            speech_lengths: torch.Tensor,
            decoding_chunk_size: int = -1,
            num_decoding_left_chunks: int = -1,
            simulate_streaming: bool = False,
    )->Tuple[torch.Tensor, torch.Tensor]:
        # Assume B = batch_size
        # 1. Encoder
        if simulate_streaming and decoding_chunk_size > 0:
            encoder_out, encoder_mask = self.encoder.forward_chunk_by_chunk(
                speech,
                decoding_chunk_size = decoding_chunk_size,
                num_decoding_left_chunks = num_decoding_left_chunks
            ) # (B, max_len, encoder_dim)
        else:
            encoder_out, encoder_mask = self.encoder(
                speech,
                speech_lengths,
                decoding_chunk_size = decoding_chunk_size,
                num_decoding_left_chunks = num_decoding_left_chunks
            ) # (B, max_len, encoder_dim)
        return encoder_out, encoder_mask

    def recognize(
            self,
            speech: torch.Tensor,
            speech_lengths: torch.Tensor,
            beam_size: int = 10,
            decoding_chunk_size: int = -1,
            num_decoding_left_chunks: int = -1,
            simulate_streaming: bool = False,
    )->torch.Tensor:
        """
        Apply beam search on attention decoder
        Args:
            speech(torch.Tensor): (batch, max_len, feat_dim)
            speech_lengths(torch.Tensor): (batch, )
            beam_size(int): beam size for beam search
            decoding_chunk_size(int): decoding chunk for dynamic chunk trained model.
                <0: for decoding ,use full chunk.
                >0: for decoding ,use fixed chunk size as set.
                0: used for training, it is prohibited here(这应该就是非流式的意思，我们不要这个)
            simulate_streaming(bool): whether do encoder forward in a streaming fashion
            (只要流式，所以用的时候应该设置成true)
        Returns:
            torch.Tensor, decoding result, (batch, mas_result_len)
        """
        assert speech.shape[0] == speech_lengths.shape[0]
        assert decoding_chunk_size != 0
        device = speech.device
        batch_size = speech.shape[0]
        # Assume B = batch_size and N = beam_size
        # 1. Encoder
        encoder_out, encoder_mask = self._forward_encoder(
            speech, speech_lengths, decoding_chunk_size,
            num_decoding_left_chunks, num_decoding_left_chunks,
            simulate_streaming) #(B, maxlen, encoder_dim)
        maxlen = encoder_out.size(1)
        encoder_dim = encoder_out.size(2)
        running_size = batch_size * beam_size
        encoder_out = encoder_out.unsqueeze(1).repeat(1, beam_size, 1, 1).view(
            running_size, maxlen, encoder_dim) #(B * N, max_len, encoder_dim)
        encoder_mask = encoder_mask.unsqueeze(1).repeat(
            1, beam_size, 1, 1).view(running_size, 1, maxlen) # (B * N, 1, maxlen)
        hyps = torch.ones([running_size, 1], dtype = torch.long,
                          device = device).fill_(self.sos) #(B * N, 1)
        scores = torch.tensor([0.0] + [-float('inf')] * (beam_size - 1),
                              dtype = torch.float)
        scores = scores.to(device).repeat([batch_size]).unsqueeze(1).to(device)#(B * N, 1)
        end_flag = torch.zeros_like(scores, dtype = torch.bool, device = device)
        cache: Optional[List[torch.Tensor]] = None
        # 2. Decoder forward step by step
        for i in range(1, maxlen + 1):
            # Stop if all batch and all beam produce eos
            if end_flag.sum() == running_size:
                break
            # 2.1 Forward decoder step
            hyps_mask =