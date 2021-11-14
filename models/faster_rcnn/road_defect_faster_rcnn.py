
# import some common libraries
import os, json, cv2, random
import matplotlib.pyplot as plt
from copy import deepcopy

# import some common detectron2 utilities
from detectron2.structures import BoxMode
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators
from detectron2.utils.visualizer import ColorMode

# Import a custom data configuration scripts.
import faster_rcnn.road_defect_dataCfg as dataCfg

class RoadDefectRCNN:
    def __init__(self, dataFolder, model_name, score_threshold=0.5):
        # dataCfg.DETECTRON2_DATASETS = os.getcwd()
        dataCfg.DETECTRON2_DATASETS = dataFolder
        dataCfg.ROADDEFECT_DATASET  = os.path.join(dataCfg.DETECTRON2_DATASETS,"../data")
        dataCfg.DATASET_BASE_PATH = dataCfg.ROADDEFECT_DATASET
        print("Dataset Basepath: ",dataCfg.DATASET_BASE_PATH)

        # Provided train data and test data
        # Original training data set with image annotation
        # print(dataCfg._PREDEFINED_SPLITS_GRC_MD['roaddefect_source'])

        # Data set without image annotation - for generalization testing.
        # print(dataCfg._PREDEFINED_SPLITS_GRC_MD['roaddefect_new'])

        # Create the test train split for them
        # print(dataCfg._PREDEFINED_SPLITS_GRC_MD['roaddefect'])

        # Register the Dataset
        # RDD_DEFECT_CATEGORIES = dataCfg.RDD_DEFECT_CATEGORIES_ALL
        # for item in RDD_DEFECT_CATEGORIES:
        #     print(item)

        DatasetCatalog.clear()
        for dataset_name, splits_per_dataset in dataCfg._PREDEFINED_SPLITS_GRC_MD["roaddefect"].items():
            inst_key = f"{dataset_name}"
            d = dataset_name.split("_")[1]
            print("[",d,"]\t",dataset_name, "\t", splits_per_dataset)
            DatasetCatalog.register(inst_key, lambda path=dataCfg.ROADDEFECT_DATASET, d=deepcopy(splits_per_dataset) : dataCfg.load_images_ann_dicts(path, d, dataCfg.RDD_DEFECT_CATEGORIES_ALL))
            meta = dataCfg.get_rdd_coco_instances_meta(dataCfg.RDD_DEFECT_CATEGORIES_ALL)
            MetadataCatalog.get(inst_key).set(evaluator_type="coco", basepath=dataCfg.ROADDEFECT_DATASET, splits_per_dataset=deepcopy(splits_per_dataset), **meta) 

        # Call the registered function and return its results.
        self.rdd2020_metadata = MetadataCatalog.get("roaddefect_val")

        # Configuration
        # https://detectron2.readthedocs.io/en/latest/modules/config.html?highlight=.DATASETS.TRAIN%20#yaml-config-references
        MODEL_ZOO = model_name
        print("COCO-Detection/faster_rcnn_{}.yaml".format(MODEL_ZOO))

        # Obtain detectron2's default config
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_{}.yaml".format(MODEL_ZOO)))
        self.cfg.OUTPUT_DIR = "./faster_rcnn/output/faster_rcnn_model/"
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(dataCfg.RDD_DEFECT_CATEGORIES_ALL) # The classes of the road defect
            
        # Step 6: Inference using the trained model
        # Inference should use the config with parameters that are used in training
        # cfg now already contains everything we've set previously. We changed it a little bit for inference:
        self.cfg.MODEL.WEIGHTS = os.path.join(self.cfg.OUTPUT_DIR, "models", "model_final_batch_size_256_{}.pth".format(MODEL_ZOO))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_threshold   # set a custom testing threshold for this model
        self.cfg.DATASETS.TEST = ("roaddefect_test",)
        self.predictor = DefaultPredictor(self.cfg)

    def predict(self, im):
        return self.predictor(im)

    def predict1(self, im):

        outputs = self.predictor(im)
        v = Visualizer(im[:, :, ::-1], metadata=self.getMetaData(),scale=1.0 )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        r_image = out.get_image()[:, :, ::-1]

        return r_image, outputs

    def getMetaData(self):
        return self.rdd2020_metadata

    #from google.colab.patches import cv2_imshow
    def cv2_imshow(self, im):
        plt.figure(figsize=(8,8))
        plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))

    # Visualize Training Dataset
    def visualise_dataset(self, dataset: str = "val"):
        splits_per_dataset = ( "lval/Czech", "lval/India", "lval/Japan")
        dataset_dicts = dataCfg.load_images_ann_dicts(dataCfg.ROADDEFECT_DATASET, splits_per_dataset, dataCfg.RDD_DEFECT_CATEGORIES_ALL)
        for d in random.sample(dataset_dicts, 3):
            print(d["file_name"])
            img = cv2.imread(d["file_name"])
            visualizer = Visualizer(img[:, :, ::-1], metadata=self.rdd2020_metadata, scale=1.0)
            out = visualizer.draw_dataset_dict(d)
            self.cv2_imshow(out.get_image()[:, :, ::-1])












