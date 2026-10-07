import torch
from .alphabet import idx_to_char_v3, idx_to_char_v2

def greedy_decode(logits, input_lengths):

    print("=== DECODER DEBUG ===")
    print("idx_to_char_v2 id:", id(idx_to_char_v2))
    print("len idx_to_char_v2:", len(idx_to_char_v2))
    print("max idx:", max(idx_to_char_v2))
    print("idx 116:", idx_to_char_v2.get(116, "<BRAK>"))
    print("idx_to_char_v3 116:", idx_to_char_v3.get(116, "<BRAK>"))
    print("=====================")

    predictions = torch.argmax(logits, dim=2)

    decoded_text = []

    for i, sample in enumerate(predictions):

        input_length = int(input_lengths[i])
        sequence = sample[:input_length]

        duplicate_decode = []
        previous = -1

        for current in sequence:

            if current != previous:
                duplicate_decode.append(int(current))

            previous = current

        blank_decode = []

        for non_blank in duplicate_decode:
            if non_blank != 0:
                blank_decode.append(non_blank)

        pred_decoded = []

        for char_idx in blank_decode:
            pred_decoded.append(idx_to_char_v3[char_idx])

        decoded_text.append("".join(pred_decoded))

        

    return decoded_text