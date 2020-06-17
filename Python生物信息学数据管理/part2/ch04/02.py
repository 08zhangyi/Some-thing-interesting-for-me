seq = "MGSNKSKPKDASQRRRSLEPAENVHGAGGGAFPASQTPSKPASADGHRGPSAAFAPAAAE"
if 'GGG' in seq and 'RRR' in seq:
    print('GGG is at position: ', seq.find('GGG'))
    print('RRR is at position: ', seq.find('RRR'))
if 'WWW' in seq or 'AAA' in seq:
    print('Either WWW or AAA occur in the sequence')
if 'AAA' in seq and not 'PPP' in seq:
    print('AAA occurs in the sequence but not PPP')

bases = ['A', 'C', 'T', 'G']
seq = 'GGACXCAGXXGATT'
seqlist = [base for base in seq if base in bases]
print(seqlist)