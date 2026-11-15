"""Checkpoint-backed CUDA placement for the native MiniMax-H3 video pipeline."""


class CheckpointPlacement:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.cpu_state = model.state_dict()
        self.active = False

    def load(self):
        if not self.active:
            self.active = True
            try:
                self.model.to(device=self.device)
            except BaseException:
                self._restore()
                raise
        return self.model

    def _restore(self):
        # Reference immutable CPU weights without copying their CUDA counterparts back.
        self.model.load_state_dict(self.cpu_state, strict=True, assign=True)
        self.active = False

    def release(self):
        if self.active:
            self._restore()
