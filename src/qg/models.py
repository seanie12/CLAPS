import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration


class AdvContrastiveQG(nn.Module):
    def __init__(self, args):
        super(AdvContrastiveQG, self).__init__()
        self.tau = args.tau
        self.pos_eps = args.pos_eps
        self.neg_eps = args.neg_eps

        self.t5_model = T5ForConditionalGeneration.from_pretrained(
            args.t5_model)
        self.projection = nn.Sequential(nn.Linear(args.hidden_size, args.hidden_size),
                                        nn.ReLU())

    def forward(self, input_ids, attention_mask,
                decoder_input_ids, decoder_attention_mask,
                lm_labels, adv=False):
        # input_ids: ids of article tokens
        # attention_mask: mask for input_ids 0 for PAD 1 o.w
        # decoder_input_ids: ids of summary tokens
        # decoder_attention_mask: mask for decoder_input_ids 0 for PAD 1 o.w
        # lm_labels: shift decoder_input_ids left

        encoder = self.t5_model.get_encoder()
        decoder = self.t5_model.get_decoder()

        encoder_outputs = encoder(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  inputs_embeds=None,
                                  head_mask=None
                                  )

        hidden_states = encoder_outputs[0]

        decoder_outputs = decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=None,
            past_key_value_states=None,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=None,
            use_cache=None,
        )
        sequence_output = decoder_outputs[0]
        # Rescale output before projecting on vocab
        # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
        sequence_output = sequence_output * (self.t5_model.model_dim ** -0.5)
        lm_logits = self.t5_model.lm_head(sequence_output)

        # Add hidden states and attention if they are here
        decoder_outputs = (lm_logits,) + decoder_outputs[1:]

        vocab_size = lm_logits.size(-1)

        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        nll = criterion(lm_logits.view(-1, vocab_size),
                        lm_labels.view(-1))

        if adv:
            proj_enc_h = self.projection(hidden_states)
            proj_dec_h = self.projection(sequence_output)
            avg_doc = self.avg_pool(proj_enc_h, attention_mask)
            avg_abs = self.avg_pool(proj_dec_h, decoder_attention_mask)

            cos = nn.CosineSimilarity(dim=-1)
            cont_crit = nn.CrossEntropyLoss()
            sim_matrix = cos(avg_doc.unsqueeze(1),
                             avg_abs.unsqueeze(0))
            perturbed_dec = self.generate_adv(sequence_output,
                                              lm_labels)  # [n,b,t,d] or [b,t,d]
            batch_size = input_ids.size(0)

            proj_pert_dec_h = self.projection(perturbed_dec)
            avg_pert = self.avg_pool(proj_pert_dec_h,
                                     decoder_attention_mask)

            adv_sim = cos(avg_doc, avg_pert).unsqueeze(1)  # [b,1]

            pos_dec_hidden = self.generate_cont_adv(hidden_states, attention_mask,
                                                    sequence_output, decoder_attention_mask,
                                                    lm_logits,
                                                    self.tau, self.pos_eps)

            avg_pos_dec = self.avg_pool(self.projection(pos_dec_hidden),
                                        decoder_attention_mask)

            pos_sim = cos(avg_doc, avg_pos_dec).unsqueeze(-1)  # [b,1]
            logits = torch.cat([sim_matrix, adv_sim], 1) / self.tau

            identity = torch.eye(batch_size, device=input_ids.device)
            pos_sim = identity * pos_sim
            neg_sim = sim_matrix.masked_fill(identity == 1, 0)
            new_sim_matrix = pos_sim + neg_sim
            new_logits = torch.cat([new_sim_matrix, adv_sim], 1)

            labels = torch.arange(batch_size,
                                  device=input_ids.device)

            cont_loss = cont_crit(logits, labels)
            new_cont_loss = cont_crit(new_logits, labels)

            cont_loss = 0.5 * (cont_loss + new_cont_loss)

            return nll, cont_loss

        else:
            return (nll, )

    def generate_adv(self, dec_hiddens, lm_labels):
        dec_hiddens = dec_hiddens.detach()

        dec_hiddens.requires_grad = True

        lm_logits = self.t5_model.lm_head(dec_hiddens)
        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        loss = criterion(lm_logits.view(-1, lm_logits.size(-1)),
                         lm_labels.view(-1))
        loss.backward()
        dec_grad = dec_hiddens.grad.detach()
        l2_norm = torch.norm(dec_grad, dim=-1)

        dec_grad /= (l2_norm.unsqueeze(-1) + 1e-12)

        perturbed_dec = dec_hiddens + self.neg_eps * dec_grad.detach()
        perturbed_dec = perturbed_dec  # [b,t,d]
        self.zero_grad()

        return perturbed_dec

    def generate_cont_adv(self, enc_hiddens, enc_mask,
                          dec_hiddens, dec_mask, lm_logits,
                          tau, eps):
        enc_hiddens = enc_hiddens.detach()
        dec_hiddens = dec_hiddens.detach()
        lm_logits = lm_logits.detach()
        dec_hiddens.requires_grad = True

        avg_enc = self.avg_pool(self.projection(enc_hiddens),
                                enc_mask)

        avg_dec = self.avg_pool(self.projection(dec_hiddens),
                                dec_mask)

        cos = nn.CosineSimilarity(dim=-1)
        logits = cos(avg_enc.unsqueeze(1), avg_dec.unsqueeze(0)) / tau

        cont_crit = nn.CrossEntropyLoss()
        labels = torch.arange(avg_enc.size(0),
                              device=enc_hiddens.device)
        loss = cont_crit(logits, labels)
        loss.backward()

        dec_grad = dec_hiddens.grad.detach()
        l2_norm = torch.norm(dec_grad, dim=-1)
        dec_grad /= (l2_norm.unsqueeze(-1) + 1e-12)

        perturb_dec_hidden = dec_hiddens + eps * dec_grad
        perturb_dec_hidden = perturb_dec_hidden.detach()
        perturb_dec_hidden.requires_grad = True
        perturb_logits = self.t5_model.lm_head(perturb_dec_hidden)

        true_probs = F.softmax(lm_logits, -1)
        true_probs = true_probs * dec_mask.unsqueeze(-1).float()

        perturb_log_probs = F.log_softmax(perturb_logits, -1)

        kl_crit = nn.KLDivLoss(reduction="sum")
        vocab_size = lm_logits.size(-1)

        kl = kl_crit(perturb_log_probs.view(-1, vocab_size),
                     true_probs.view(-1, vocab_size))
        kl = kl / torch.sum(dec_mask).float()
        kl.backward()

        kl_grad = perturb_dec_hidden.grad.detach()

        l2_norm = torch.norm(kl_grad, dim=-1)

        kl_grad /= (l2_norm.unsqueeze(-1) + 1e-12)

        perturb_dec_hidden = perturb_dec_hidden - eps * kl_grad

        return perturb_dec_hidden

    def avg_pool(self, hidden_states, mask):
        length = torch.sum(mask, 1, keepdim=True).float()
        mask = mask.unsqueeze(2)
        hidden = hidden_states.masked_fill(mask == 0, 0.0)
        avg_hidden = torch.sum(hidden, 1) / length

        return avg_hidden
