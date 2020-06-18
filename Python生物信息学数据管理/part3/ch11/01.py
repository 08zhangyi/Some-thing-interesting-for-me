class Pea:
    def __init__(self, genotype):
        self.genotype = genotype

    def get_phenotype(self):
        if "G" in self.genotype:
            return "yellow"
        else:
            return "green"

    def create_offspring(self, other):
        offspring = []
        new_genotype = ""
        for haplo1 in self.genotype:
            for haplo2 in other.genotype:
                new_genotype = haplo1 + haplo2
                offspring.append(Pea(new_genotype))
        return offspring

    def __repr__(self):
        return self.get_phenotype() + ' [%s]' % self.genotype


class PeaStrain:
    def __init__(self, peas):
        self.peas = peas

    def __repr__(self):
        return 'strain with %i peas'%(len(self.peas))


yellow = Pea('GG')
green = Pea('gg')
strain = PeaStrain([yellow, green])
print(strain)


class CommentedPea(Pea):

    def __init__(self, genotype, comment):
        Pea.__init__(self, genotype)
        self.comment = comment

    def __repr__(self):
        return  '%s [%s] (%s)' % (self.get_phenotype(), self.genotype, self.comment)


yellow1 = CommentedPea('GG', 'homozygote')
yellow2 = CommentedPea('Gg', 'heterozygote')
print(yellow1)