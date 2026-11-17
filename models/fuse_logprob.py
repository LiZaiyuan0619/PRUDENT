"""Answer log-probabilities of the pruned large model, for the confidence of the first answer.

install(lm) wraps lm.generate so that generate() also returns its raw logits (output_logits) while the caller,
InternVLChatModel.chat(), still receives the token ids; the last call's (sequences, logits) wait in lm._fuse_last.
stats() turns them into per-answer statistics. Recording the logits does not change greedy decoding.
"""
import torch


def install(lm):
    if getattr(lm, "_fuse_wrapped", False):
        return
    orig = lm.generate

    def generate_with_logits(*args, **kwargs):
        kwargs.update(return_dict_in_generate=True, output_logits=True)
        out = orig(*args, **kwargs)
        lm._fuse_last = (out.sequences, out.logits)
        return out.sequences

    lm.generate = generate_with_logits
    lm._fuse_wrapped = True


def stats(lm, eos_id):
    """-> dict over the answer tokens (before the first EOS): n, sum / mean / min log-prob, first_p and first_margin
    (probability of the first token, and its lead over the runner-up), eos_p (None if no EOS was generated)."""
    seq, logits = lm._fuse_last
    seq = seq[0]
    lp = torch.stack([torch.log_softmax(x[0].float(), -1) for x in logits])
    n = int((seq != eos_id).long().cumprod(0).sum())
    tok = lp[torch.arange(n, device=lp.device), seq[:n]]
    p0 = lp[0].exp()
    top2 = torch.topk(p0, 2).values
    return dict(n=n, sum=float(tok.sum()), mean=float(tok.mean()) if n else 0.0, min=float(tok.min()) if n else 0.0,
                first_p=float(p0[seq[0]]), first_margin=float(top2[0] - top2[1]),
                eos_p=None if n == len(seq) else float(lp[n, eos_id].exp()),
                argmax_ok=bool((lp.argmax(-1) == seq[:len(lp)]).all()))
