import numpy as np
def random_spans_noise_mask(
        length: int, mean_span_count: float, mean_noise_span_length: int
    ) -> np.array:

    orig_length = length

    num_noise_tokens = int(np.round(mean_span_count * mean_noise_span_length))
    # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
    # num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
    num_noise_tokens = min(max(num_noise_tokens, 1), max(int(length/1.5),1)) # modified
    num_noise_spans = int(np.round(num_noise_tokens / mean_noise_span_length))

    # avoid degeneracy by ensuring positive number of noise spans
    num_noise_spans = max(num_noise_spans, 1)
    num_nonnoise_tokens = length - num_noise_tokens
    # print(num_nonnoise_tokens, num_noise_spans, num_noise_tokens)
    
    # pick the lengths of the noise spans and the non-noise spans
    def _random_segmentation(num_items, num_segments):

        mask_indices = np.arange(num_items - 1) < (num_segments - 1)
        np.random.shuffle(mask_indices)
        first_in_segment = np.pad(mask_indices, [[1, 0]])
        segment_id = np.cumsum(first_in_segment)
        # count length of sub segments assuming that list is sorted
        _, segment_length = np.unique(segment_id, return_counts=True)
        return segment_length

    noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans)
    nonnoise_span_lengths = _random_segmentation(num_nonnoise_tokens, num_noise_spans)
    # print("noise_span_lengths: ",noise_span_lengths)
    # print("nonnoise_span_lengths: ",nonnoise_span_lengths)
    interleaved_span_lengths = np.reshape(
        np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1),
        [num_noise_spans * 2],
    )
    span_starts = np.cumsum(interleaved_span_lengths)[:-1]
    span_start_indicator = np.zeros((length,), dtype=np.int8)
    span_start_indicator[span_starts] = True
    span_num = np.cumsum(span_start_indicator)
    is_noise = np.equal(span_num % 2, 1)
    return is_noise[:orig_length].astype(np.int8)

from typing import List, Tuple, Any
def get_mask_start_end_ids(
        mask_indices: np.ndarray
    ) -> Tuple[List[int], List[int]]:
        """Return mask start and end ids."""
        start_indices = mask_indices - np.roll(mask_indices, 1, axis=-1) * mask_indices
        start_indices[:, 0] = mask_indices[:, 0]

        end_indices = mask_indices - np.roll(mask_indices, -1, axis=-1) * mask_indices
        end_indices[:, -1] = mask_indices[:, -1]

        return (
            np.nonzero(start_indices)[1].tolist(),
            np.nonzero(end_indices)[1].tolist(),
        )