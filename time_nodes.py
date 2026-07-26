"""
Time nodes that replicate Nuke's temporal tools.

These nodes treat an IMAGE batch as a timeline: batch item ``i``
corresponds to timeline frame ``frame_start + i``. This mirrors how a
frame range read into ComfyUI as a batch maps onto Nuke's timeline.
"""

import logging

import torch

from .utils import NukeNodeBase, ensure_batch_dim

logger = logging.getLogger(__name__)


class NukeFrameHold(NukeNodeBase):
    """FrameHold - hold (freeze) frames of a batch, matching Nuke's FrameHold node.

    Batch-as-timeline model:
        The input IMAGE batch of B frames is interpreted as a contiguous
        frame range on a timeline. Batch item 0 is timeline frame
        ``frame_start``, item 1 is ``frame_start + 1``, and so on, up to
        item B-1 at timeline frame ``frame_start + B - 1``.

    For each output item i (timeline frame t = frame_start + i):
        - increment == 0:
            held = first_frame
            (the classic FrameHold: every output frame shows first_frame)
        - increment > 0:
            held = first_frame + increment * floor((t - first_frame) / increment)
            (frames advance in steps of ``increment`` starting at
            ``first_frame``; floor division handles t < first_frame by
            stepping backwards, before clamping)
        The held frame number is then clamped into the available range
        [frame_start, frame_start + B - 1], and:
            out[i] = in[held - frame_start]

    The output batch always has the same length as the input batch.

    Example:
        first_frame=1, increment=5, frame_start=1, batch of 15 frames
        (timeline frames 1..15) -> held frames:
            1, 1, 1, 1, 1, 6, 6, 6, 6, 6, 11, 11, 11, 11, 11
        i.e. the sequence freezes on frame 1 for 5 frames, then jumps to
        frame 6 for 5 frames, then frame 11 for the rest.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "Input frame batch, treated as a timeline: batch "
                            "item i is timeline frame frame_start + i."
                        ),
                    },
                ),
                "first_frame": (
                    "INT",
                    {
                        "default": 1,
                        "min": -100000,
                        "max": 100000,
                        "step": 1,
                        "tooltip": (
                            "Timeline frame number to hold. With increment=0 "
                            "every output frame shows this frame; with "
                            "increment>0 it is the frame where the "
                            "increment stepping starts."
                        ),
                    },
                ),
                "increment": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 100000,
                        "step": 1,
                        "tooltip": (
                            "0 = hold first_frame for the whole batch. "
                            "N > 0 = advance the held frame every N frames "
                            "(first_frame, first_frame+N, first_frame+2N, ...)."
                        ),
                    },
                ),
                "frame_start": (
                    "INT",
                    {
                        "default": 1,
                        "min": -100000,
                        "max": 100000,
                        "step": 1,
                        "tooltip": (
                            "Timeline frame number of the first batch item "
                            "(batch item 0). Held frame numbers are clamped "
                            "into [frame_start, frame_start + batch - 1]."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "frame_hold"
    CATEGORY = "Nuke/Time"

    def frame_hold(self, image, first_frame, increment, frame_start):
        """Hold frames of the input batch per the batch-as-timeline model.

        For output item i (timeline frame t = frame_start + i):
        held = first_frame if increment == 0 else
        first_frame + increment * floor((t - first_frame) / increment),
        clamped to [frame_start, frame_start + B - 1];
        out[i] = in[held - frame_start].
        """
        img = ensure_batch_dim(image)
        batch = img.shape[0]

        indices = []
        for i in range(batch):
            t = frame_start + i
            if increment == 0:
                held = first_frame
            else:
                # Python floor division floors toward negative infinity,
                # which is exactly the behavior we want for t < first_frame
                # (steps backwards before clamping).
                held = first_frame + increment * ((t - first_frame) // increment)
            held = max(frame_start, min(held, frame_start + batch - 1))
            indices.append(held - frame_start)

        index_tensor = torch.tensor(indices, dtype=torch.long, device=img.device)
        result = torch.index_select(img, 0, index_tensor)
        return (result,)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "NukeFrameHold": NukeFrameHold,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NukeFrameHold": "Nuke FrameHold",
}
