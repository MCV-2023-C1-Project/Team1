# noinspection PyInterpreter
from utils import utils
from preprocessing import pipelines as pipes
from common.options import Parser
from common.config import *
from metrics.retrieval_distances import *
from metrics.retrieval_metrics import *

from toolz.functoolz import pipe

## TODO Crear sa funció de Evaluar i ja estaria sa part sensera des cuadros (retrieval)
def main():
    METHODS = {"gray_hist":pipes.generate_grayscale_histogram_descriptors,
               "norm-rg":pipes.generate_normalized_rg_histogram_descriptors,
               "cummulative":pipes.generate_cummulative_histogram_descriptors,
               "multitile":pipes.generate_multi_tile_histogram_descriptors}

    PREPROCESSING = {"gray_hist":[str, utils.read_img , utils.convert2gray],
                    "norm-rg":[str, utils.read_img],
                    "cummulative":[str, utils.read_img , utils.normalize_min_max, utils.convert2luv],
                    "multitile":[str, utils.read_img, utils.convert2luv]}

    SIMILARITY = {"cosine": cos_sim,
                  "l1": l1norm,
                  "euc": euclDis,
                  "chi": chi2_distance,
                  "hellkdis":hellkdis,
                  "jensen": jensensim,
                  "histint": histogram_intersection
                  }
    args = Parser().parse()
    ## Reading querys
    if args.queryfile != False:
        querys_gt = utils.read_pickle(args.queryfile)
        gt = [value[0] for value in querys_gt]

        print(querys_gt)

    # SOME UTILS
    QUERYS = sorted(utils.read_bbdd(args.querys))
    images_path = sorted(utils.read_bbdd(BBDD))


    ## GETTING THE DESCRIPTORS OF OUR BBDD IF THEY ARE COMPUTED
    filename = args.method+".pkl"
    MDESCRIPTOR_PATH = DESCRIPTORS_PATH+f"/{filename}"
    descriptors_bdr = {}

    if (args.overwrite is False) and (args.update is False):
        descriptors_bdr, _ = utils.get_descriptor_database(MDESCRIPTOR_PATH)

    else:
        if args.overwrite is True:
            images_to_upload = sorted(utils.read_bbdd(BBDD))

        ## In this lines of code we are checking if there is a new image in the BBDD
        elif args.update is True:
            descriptors_bdr, _ = utils.get_descriptor_database(MDESCRIPTOR_PATH)
            images_to_upload = sorted([img for img in images_path if img.name not in descriptors_bdr.keys()])

        if len(images_to_upload) != 0:
            print("STARTING PREPROCESSING THE DATA")
            preprocessed_images = [pipe(img, *PREPROCESSING[args.method]) for img in images_to_upload]
            print("STARTING TO COMPUTE THE DESCRIPTORS OF THE IMAGES")

            if args.method == "multitile":
                tiles = args.tiles
                new_descriptors = METHODS[args.method](preprocessed_images, int(tiles), bins=10)

            else:
                new_descriptors = METHODS[args.method](preprocessed_images)


            for idx, im in enumerate(images_to_upload):
                descriptors_bdr[im.name] = new_descriptors[idx]


        utils.save_descriptor_bbdd(descriptors_bdr, filepath=DESCRIPTORS_PATH, filename=filename)

    ## Applying the process to the queries
    preprocessed_images = [pipe(img, *PREPROCESSING[args.method]) for img in QUERYS]
    if args.method == "multitile":
        tiles = args.tiles
        query_descriptors = METHODS[args.method](preprocessed_images, int(tiles), bins=10)

    else:
        query_descriptors = METHODS[args.method](preprocessed_images)

    response = pipes.generate_K_response(descriptors_bdr=descriptors_bdr, descriptors_queries=query_descriptors, sim_func=SIMILARITY[args.similarity], k=int(args.k))
    print(response)
    if args.queryfile != False:
        print(mapk(querys_gt, response, k=5))
    #utils.write_pickle(response, RESULTS+f"{args.method}_{args.similarity}_"+"result.pkl")
    utils.write_pickle(response, RESULTS+"result.pkl")



    ## Starting the retrieval
main()
exit()















