"""Keep the FL2VA DiT in VRAM across requests and stream Qwen decoder layers."""

from contextlib import nullcontext

import torch

from ..native_placement import CheckpointPlacement


_GIB = 1024 ** 3


def _weight_bytes(placement):
    return sum(t.numel() * t.element_size() for t in placement.cpu_state.values())


class _RetainedDiT:
    def __init__(self, placement):
        self.placement = placement
        self.keep = False

    @property
    def model(self):
        return self.placement.model

    @property
    def active(self):
        return self.placement.active

    def load(self):
        return self.placement.load()

    def release(self):
        if not self.keep:
            self.placement.release()


class _CachedText:
    def __init__(self, placement, dit):
        self.placement = placement
        self.dit = dit
        self.model = placement.model
        self.device = placement.device
        decoder = getattr(self.model, "language_model", self.model)
        self.layers = tuple(decoder.layers)
        # These modules run outside Qwen's decoder-layer loop. Retain them for both text and keyframe requests.
        self.fixed = [decoder.embed_tokens, decoder.norm, decoder.rotary_emb]
        if decoder is not self.model:
            self.fixed.append(self.model.visual)
        self.fixed_bytes = sum(t.numel() * t.element_size() for m in self.fixed for t in m.state_dict().values())
        self.layer_bytes = [sum(t.numel() * t.element_size() for t in layer.state_dict().values())
                            for layer in self.layers]
        self.budget = 0
        self.held_bytes = 0
        self.prefix_layers = 0
        self.hooks = []
        self.phase = None

    def set_budget(self, budget):
        self.budget = budget
        if self.held_bytes > budget:
            self.evict()

    def evict(self):
        self.phase = None
        if self.held_bytes:
            for hook in self.hooks:
                hook.remove()
            self.hooks.clear()
            # Assign immutable checkpoint tensors rather than copying the CUDA tensors back.
            self.model.load_state_dict(self.placement.cpu_state, strict=True, assign=True)
            self.held_bytes = 0
            self.prefix_layers = 0
        else:
            self.placement.release()

    def load(self):
        if not self.dit.active:
            return self.placement.load()
        if not self.held_bytes:
            self._cache_prefix()
        return self.model

    def release(self):
        try:
            self.placement.release()
        finally:
            self.phase = None

    def _cache_prefix(self):
        self.held_bytes = self.fixed_bytes
        try:
            for module in self.fixed:
                module.to(device=self.device, non_blocking=True)

            for layer, size in zip(self.layers, self.layer_bytes, strict=True):
                if self.held_bytes + size > self.budget:
                    break
                layer.to(device=self.device, non_blocking=True)
                self.held_bytes += size
                self.prefix_layers += 1

            for layer in self.layers[self.prefix_layers:]:
                placement = CheckpointPlacement(layer, self.device)
                self.hooks.extend((layer.register_forward_pre_hook(self._load_layer(placement)),
                                   layer.register_forward_hook(self._release_layer(placement), always_call=True)))
        except BaseException:
            self.evict()
            raise

    def _load_layer(self, placement):
        def load(_module, _args):
            with self.phase("placement") if self.phase is not None else nullcontext():
                placement.load()

        return load

    def _release_layer(self, placement):
        def release(_module, _args, _result):
            with self.phase("placement") if self.phase is not None else nullcontext():
                placement.release()

        return release


class ResidentVideoPlacement:
    """Reuse a resident DiT and as much of Qwen as the next request safely allows."""

    def __init__(self, text, dit, vae, audio_vae, *, hidden_size):
        self.device = dit.device
        self.hidden_size = hidden_size
        self.dit_bytes = _weight_bytes(dit)
        self.decoder_bytes = max(_weight_bytes(vae), _weight_bytes(audio_vae))
        self.dit = _RetainedDiT(dit)
        self.text = _CachedText(text, self.dit)
        self.disabled = False

    @property
    def resident(self):
        return self.dit.active

    @property
    def resident_text_layers(self):
        return self.text.prefix_layers

    def prepare(self, *, sequence_tokens, output_pixels, phase=None):
        if self.disabled:
            return
        free, _ = torch.cuda.mem_get_info(self.device)
        cached = torch.cuda.memory_reserved(self.device) - torch.cuda.memory_allocated(self.device)
        held = (self.dit_bytes if self.dit.active else 0) + self.text.held_bytes
        # Reserve per-stage peaks independently. Text layers, DiT activations and video decode do not overlap.
        workspace = max(max(self.text.layer_bytes) + 2 * _GIB,
                        sequence_tokens * self.hidden_size * 64,
                        self.decoder_bytes + output_pixels * 40)
        reserve = max(12 * _GIB, workspace + 4 * _GIB)
        budget = free + cached + held - self.dit_bytes - reserve
        if budget < self.text.fixed_bytes:
            self.evict()
            return
        self.text.set_budget(budget)
        self.text.phase = phase
        self.dit.keep = True

    def evict(self, *, disable=False):
        """Restore CPU weights and optionally suspend retention for later requests."""
        if disable:
            self.disabled = True
        self.text.evict()
        self.dit.keep = False
        self.dit.release()
