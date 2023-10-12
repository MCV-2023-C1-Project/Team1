import argparse

from argparse import HelpFormatter
from operator import attrgetter


class SortingHelpFromatter(HelpFormatter):
    def add_arguments(self, actions):
        actions = sorted(actions, key=attrgetter("option_strings"))
        super(SortingHelpFromatter, self).add_arguments(actions)

class Parser(argparse.ArgumentParser):

    def __init__(self):

        super().__init__(
            description="Options for museum painting retrieval ",
            formatter_class=SortingHelpFromatter
        )

        super().add_argument("-qf", "--queryfile", default=False, required=False ,help="File with the gt of the querys")
        super().add_argument("-q", "--querys", type=str, required=True ,help="Folder with the querys")
        super().add_argument("--update", required=False, action="store_true", help="Check the descritptors database and update if there is a new image to compute his descriptor")
        super().add_argument( "--overwrite", action="store_true", help="Compute and overwrite all the descriptor BBDD")
        super().add_argument("-m", "--method", choices=["gray_hist","norm-rg","cummulative","multitile", "pyramidal", "multiresolution"], required=True, type=str, help= "Methods to compute the descriptors")
        super().add_argument("-s", "--similarity", choices=["cosine", "l1", "euc", "chi", "hellkdis", "jensen", "histint"], required=True, type=str, help= "Methods to compute the similarity")
        super().add_argument("-k", "--k", default=10, help="@K to compute the retrieval")
        super().add_argument("-nt", "--tiles", default=6, help="tiles to compute piramidal slicing")
        super().add_argument("-br", "--background_removal", action="store_true", help="to enable the background removal")
        super().add_argument("-st", "--steps", default=3, help="Steps for the pyramidal representation")

    def parse(self):
        return super().parse_args()

