import re

genome_seq = open('genome.txt').read()
sites = []
for line in open('TFBS.txt'):
    fields = line.split()
    tf = fields[0]
    site = fields[1]
    sites.append((tf, site))
for tf, site in sites:
    tfbs_regexp = re.compile(site)
    all_matches = tfbs_regexp.findall(genome_seq)
    matches = tfbs_regexp.finditer(genome_seq)
    if all_matches:
        print(tf, ':')
        for tfbs in matches:
            print('\t', tfbs.group(), tfbs.start(), tfbs.end())
