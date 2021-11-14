from PIL import Image
import os, random, cv2
from faster_rcnn.road_defect_faster_rcnn import RoadDefectRCNN
import faster_rcnn.road_defect_dataCfg as dataCfg
from detectron2.utils.visualizer import Visualizer

if __name__ == "__main__":
    
    faster_rcnn = RoadDefectRCNN(os.getcwd(), 'R_50_FPN_3x', 0.6)

    # faster_rcnn.visualise_dataset()

    # Then, we randomly select several samples to visualize the prediction results.
    splits_per_dataset = ( "ltest/India", "ltest/Japan", "ltest/Czech")
    dataset_dicts = dataCfg.load_images_ann_dicts(dataCfg.ROADDEFECT_DATASET, splits_per_dataset, dataCfg.RDD_DEFECT_CATEGORIES_ALL)

    for idx, d in enumerate(random.sample(dataset_dicts, 3)):

        im = cv2.imread(d["file_name"])
        # im = cv2.imread('./01.jpg')
        outputs = faster_rcnn.predict(im)
        
        print(idx, ".) ", "-", d["file_name"], outputs["instances"].pred_classes)
        print("     ", outputs["instances"].scores)
        v = Visualizer(im[:, :, ::-1],
                       metadata=faster_rcnn.getMetaData(), 
                       scale=1.0
        )

        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        # faster_rcnn.cv2_imshow(out.get_image()[:, :, ::-1])
        image = Image.fromarray(out.get_image()[:, :, ::-1])
        image.show()
        print("=======>", outputs)

    

        














