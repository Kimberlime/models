import numpy as np
from six import BytesIO
from PIL import Image

import tensorflow as tf
import json
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import os
import pathlib


def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    img_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(BytesIO(img_data))
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

def get_keypoint_tuples(eval_config):
    """Return a tuple list of keypoint edges from the eval config.

    Args:
      eval_config: an eval config containing the keypoint edges

    Returns:
      a list of edge tuples, each in the format (start, end)
    """
    tuple_list = []
    kp_list = eval_config.keypoint_edge
    for edge in kp_list:
        tuple_list.append((edge.start, edge.end))
    return tuple_list


def get_model_detection_function(model):
    """Get a tf.function for detection."""

    @tf.function
    def detect_fn(image):
        """Detect objects in image."""

        image, shapes = model.preprocess(image)
        prediction_dict = model.predict(image, shapes)
        detections = model.postprocess(prediction_dict, shapes)

        return detections, prediction_dict, tf.reshape(shapes, [-1])

    return detect_fn


def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis,...]

    # Run inference
    model_fn = model.signatures['serving_default']
    output_dict = model_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key:value[0, :num_detections].numpy()
                   for key,value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    # Handle models with masks:
    if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            output_dict['detection_masks'], output_dict['detection_boxes'],
            image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                           tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

    return output_dict


def show_inference(model, image_path, category_index, output_path):
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = np.array(Image.open(image_path))
    # Actual detection.
    file_name = os.path.basename(image_path)
    output_dict = run_inference_for_single_image(model, image_np)
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks_reframed', None),
        use_normalized_coordinates=True,
        line_thickness=4,
        max_boxes_to_draw=1000,)

    image = Image.fromarray(image_np)
    image.save(os.path.join(output_path, file_name))


def show_gt(image_path, category_index, coors, tags, output_path):
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = np.array(Image.open(image_path))

    # Actual detection.
    file_name = os.path.basename(image_path)
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        coors,
        tags,
        np.ones_like(tags),
        category_index,
        instance_masks=None,
        use_normalized_coordinates=True,
        line_thickness=4,
        max_boxes_to_draw=1000,)

    image = Image.fromarray(image_np)
    image.save(os.path.join(output_path, file_name))


def main():
    assets_dict = {}
    with open('/home/kimberly/projects/ui_data_v13/ui/test_vott/overlapped.vott', "r") as fp:
        data = json.load(fp)
        assets = data['assets']
        for a in assets:
            assets_dict[assets[a]['name']] = a
    base_dir = '/home/kimberly/projects/ui_data_v13/ui/'
    pbtxt = os.path.join(base_dir, 'ui.pbtxt')

    model_dir = '/home/kimberly/projects/ui_data_v13/ui/model'

    detection_model = tf.saved_model.load(model_dir)
    PATH_TO_LABELS = pbtxt
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
    category_dict = {}
    for cat in category_index.values():
        category_dict[cat['name']] = cat['id']
    PATH_TO_TEST_IMAGES_DIR = pathlib.Path(base_dir+'/test')
    TEST_IMAGE_PATHS = sorted(list(PATH_TO_TEST_IMAGES_DIR.glob("*")))
    for image_path in TEST_IMAGE_PATHS:
        # show_inference(detection_model, image_path, category_index)
        image_name = os.path.basename(image_path)
        asset_key = assets_dict[image_name]
        with open('/home/kimberly/projects/ui_data_v13/ui/test_vott/'+asset_key+ '-asset.json', "r") as fp:
            asset_data = json.load(fp)
            regions = asset_data['regions']
            im_width = asset_data['asset']['size']['width']
            im_height = asset_data['asset']['size']['height']
            coors = []
            tags = []
            for r in regions:
                height = r['boundingBox']['height']
                width = r['boundingBox']['width']
                left = r['boundingBox']['left']
                top = r['boundingBox']['top']
                # coor = [left / width, top / height, (left + width) / width, (top + height) / height]
                coor = [top / im_height, left / im_width, (top + height) / im_height, (left + width) / im_width]

                coors.append(coor)
                tag = r['tags'][0]
                tag = tag.lower()
                tag_id = category_dict[tag]
                tags.append(tag_id)
            coors = np.asarray(coors)
            tags = np.asarray(tags)
        show_inference(detection_model, image_path, category_index, '/home/kimberly/projects/ui_data_v13/ui/test_detected')
        show_gt(image_path, category_index, coors, tags, '/home/kimberly/projects/ui_data_v13/ui/gt')

if __name__ == '__main__':
    main()
