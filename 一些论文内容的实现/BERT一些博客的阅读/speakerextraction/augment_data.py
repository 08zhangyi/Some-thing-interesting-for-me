#!/usr/bin/env python
import ast


def data_augmentation(saved_file="label_honglou.txt"):
    speakers = []
    contexts = []
    combined_res = [] 
    # combine N speakers with M contexts to get N*M examples
    with open(saved_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            res = ast.literal_eval(line)
            speakers.append(res['speaker'])
            ctx = res['context']
            ctx_left = ctx[:res["istart"]]
            ctx_right = ctx[res["iend"]:]
            contexts.append([ctx_left, ctx_right, res["istart"]])
            
    uid = 0   
    len_truncate = 128
    for speaker in speakers:
        for ctx_left, ctx_right, istart in contexts:
            # do not create new sentence for contexts without original speaker
            if istart == -1: continue
            try:
                new_ctx = ctx_left + speaker + ctx_right
                new_iend = istart + len(speaker)
                new_speaker = speaker

                # truncate the input if sentence is longer than 128 words
                if len(new_ctx) > len_truncate:
                    truncated_ctx = new_ctx[-len_truncate:]
                    # if speaker is in the truncated_ctx
                    if (len(new_ctx)-istart)<len_truncate:
                        istart = istart - (len(new_ctx) - len_truncate)
                        new_iend = istart + len(speaker)
                        new_speaker = truncated_ctx[istart:new_iend]
                    else: # if speaker is not in the truncated_ctx
                        istart = -1
                        new_iend = 0
                        new_speaker = None
                    new_ctx = truncated_ctx
                res = {'uid': uid,
                       'context': new_ctx,
                       'speaker': new_speaker,
                       'istart': istart,
                       'iend': new_iend}
                combined_res.append(res)
                uid = uid + 1
            except:
                continue
    return combined_res


speakers = data_augmentation()
