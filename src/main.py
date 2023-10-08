import os.path

# noinspection PyInterpreter
from utils import utils
from preprocessing import pipelines as pipes
from common.options import Parser
from common.config import *
from metrics.retrieval_distances import *
from metrics.retrieval_metrics import *

from toolz.functoolz import pipe

def main():
    METHODS = {"gray_hist":pipes.generate_grayscale_histogram_descriptors,
               "norm-rg":pipes.generate_normalized_rg_histogram_descriptors,
               "cummulative":pipes.generate_cummulative_histogram_descriptors,
               "multitile":pipes.generate_multi_tile_histogram_descriptors}

    PREPROCESSING = {"gray_hist":[str, utils.read_img , utils.convert2gray],
                    "norm-rg":[str, utils.read_img],
                    "cummulative":[str, utils.read_img , utils.normalize_min_max, utils.convert2lab],
                    "multitile":[str, utils.read_img, utils.convert2lab]}

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
    query_dataset_name = os.path.basename((args.querys))
    filename = f"{query_dataset_name}"+f"{args.method}_{args.similarity}"+".pkl"
    MDESCRIPTOR_PATH = DESCRIPTORS_PATH+"/"+filename
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
            if args.background_removal is True:
                dict_masks, check = utils.get_descriptor_database(filepath=DESCRIPTORS_PATH, filename=f"{query_dataset_name}"+"_masks.pkl")
                if not check:
                    dict_masks = pipes.generate_mask_dict(images_to_upload, )
                    utils.save_descriptor_bbdd(dict_masks, filepath=DESCRIPTORS_PATH, filename=f"{query_dataset_name}"+"_masks.pkl")
                #print(preprocessed_images)
                #print(dict_masks.values())
                for idx, (image, mask) in enumerate(zip(preprocessed_images, dict_masks.values())):

                    #print(mask.shape)
                    #print(image.shape)
                    preprocessed_images[idx] = (image * mask[:, :, None])


            print("STARTING TO COMPUTE THE DESCRIPTORS OF THE IMAGES")
            if args.method == "multitile":
                tiles = args.tiles
                new_descriptors = METHODS[args.method](preprocessed_images, int(tiles), bins=16)

            else:
                new_descriptors = METHODS[args.method](preprocessed_images)


            for idx, im in enumerate(images_to_upload):
                descriptors_bdr[im.name] = new_descriptors[idx]


        utils.save_descriptor_bbdd(descriptors_bdr, filepath=DESCRIPTORS_PATH, filename=filename)

    ## Applying the process to the queries
    preprocessed_images = [pipe(img, *PREPROCESSING[args.method]) for img in QUERYS]
    if args.method == "multitile":
        tiles = args.tiles
        query_descriptors = METHODS[args.method](preprocessed_images, int(tiles), bins=16)

    else:
        query_descriptors = METHODS[args.method](preprocessed_images)



    response = pipes.generate_K_response(descriptors_bdr=descriptors_bdr, descriptors_queries=query_descriptors, sim_func=SIMILARITY[args.similarity], k=int(args.k))
    if args.queryfile != False:
        print(mapk(querys_gt, response, k=1))
    utils.write_pickle(response, RESULTS+"_qst2_"+f"{args.method}_{args.similarity}_"+"result.pkl")
    #utils.write_pickle(response, RESULTS+"result.pkl")



    ## Starting the retrieval
main()
exit()















