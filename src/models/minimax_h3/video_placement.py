"""Keep immutable CPU weights when the MiniMax-H3 video baseline swaps components."""

from contextlib import contextmanager


@contextmanager
def checkpoint_offload(pipe):
    """Keep Diffusers' placement decisions and restore CPU tensors without a CUDA download."""
    managed_hooks = pipe._components_manager.model_hooks
    if not managed_hooks:
        raise ValueError("MiniMax-H3 checkpoint offload requires Diffusers component management")
    replacements = []
    try:
        for hook in managed_hooks:
            model = hook.model
            if any(parameter.device.type != "cpu" for parameter in model.parameters()):
                raise ValueError("Install MiniMax-H3 checkpoint offload before the first CUDA forward")
            state = model.state_dict()
            buffers = [(module, name, buffer) for module in model.modules()
                       for name, buffer in module.named_buffers(recurse=False)
                       if name in module._non_persistent_buffers_set]

            def restore(model=model, state=state, buffers=buffers):
                model.load_state_dict(state, strict=True, assign=True)
                for module, name, buffer in buffers:
                    setattr(module, name, buffer)

            replacements.append((hook, hook.__dict__.get("offload")))
            hook.offload = restore
        yield
    finally:
        for hook, prior in replacements:
            if prior is None:
                del hook.offload
            else:
                hook.offload = prior
