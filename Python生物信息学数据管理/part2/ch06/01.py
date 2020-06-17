tracking = open('transcripts.tracking', 'r')
out_file = open('transcripts-filtered.tracking', 'w')
for track in tracking:
    columns = track.strip().split('\t')
    wildtype = columns[4:7].count('-')
    treatment = columns[7:10].count('-')
    if wildtype < 2 or treatment < 2:
        out_file.write(track)
tracking.close()
out_file.close()

output_file = open('transcripts-filtered.tracking', 'w')
for track in open('transcripts.tracking'):
    columns = track.strip().split('\t')
    wt = 0
    t = 0
    if columns[4] != '-': wt += 1
    if columns[5] != '-': wt += 1
    if columns[6] != '-': wt += 1
    if columns[7] != '-': t += 1
    if columns[8] != '-': t += 1
    if columns[9] != '-': t += 1
    if wt > 1 or t > 1:
        output_file.write(track)
output_file.close()