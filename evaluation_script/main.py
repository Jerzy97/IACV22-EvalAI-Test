"""
Evaluation script for eval.ai project server

Author: Georg Leonhard Brunner
Email : georg.brunner@vision.ee.ethz.ch
Date  : 25. August 2022
"""


from PIL import Image
from pathlib import Path
from zipfile import ZipFile

from skimage import metrics


def load_img(path):
    """ Load image using PIL then convert
        H x W x C in [0, 255] --> C x H x W in [0, 1]
    """
    img = Image.open(path)
    arr = np.array(img)
    arr = arr[None, :, :] if arr.ndim == 2 else arr.transpose(2, 0, 1)
    return arr.astype(np.float32) / 255.


def evaluate(
    test_annotation_file, user_submission_file, phase_codename, **kwargs
):
    """
    Evaluates submission for a particular challenge phase and returns score
    
    Args:
        test_annotations_file   ... Path to ground truth on server
                                    This should be a directory
        user_submission_file    ... Path to file submitted by the user
        phase_codename:         ... Phase to which submission is made

        `**kwargs`: keyword arguments that contains additional submission
        metadata that challenge hosts can use to send slack notification.
        Access submission metadata with kwargs['submission_metadata']
    """
    output = {}
    
    if phase_codename == 'ex1':
        # Load ground truth images
        gt_path = Path(test_annotation_file).parent
        gt_public = load_img(gt_path / 'public.png')
        gt_private = load_img(gt_path / 'private.png')

        # Load submitted images
        with ZipFile(user_submission_file, 'r') as zf:
            for entry in zf.infolist():
                with zf.open(entry) as f:
                    fname = Path(f.name).name
                    if fname == 'public.png':
                        sub_public = load_img(f)
                    elif fname == 'private.png':
                        sub_private = load_img(f)

        psnr_public = metrics.peak_signal_noise_ratio(gt_public, sub_public)
        ssim_public = metrics.structural_similarity(
            gt_public, sub_public, channel_axis=0
        )
        
        psnr_private = metrics.peak_signal_noise_ratio(gt_private, sub_private)
        ssim_private = metrics.structural_similarity(
            gt_private, sub_private, channel_axis=0
        )
        
        output['result'] = [
            {
                'ex1_public': {
                    'PSNR': psnr_public,
                    'SSIM': ssim_public,
                    'Total': ssim_public * psnr_public
                }
            },
            {
                'ex1_private': {
                    'PSNR': psnr_private,
                    'SSIM': ssim_private,
                    'Total': ssim_private * psnr_private

                }
            }
        ]
        # To display the results in the result file
        output['submission_result'] = output['result'][0]['ex1_public']
    
    elif phase_codename == 'ex2':
        output['result'] = [
            {'ex2_public': {'Accuracy': 0.5}},
            {'ex2_private': {'Accuracy': 0.5}}
        ]
        # To display the results in the result file
        output['submission_result'] = output['result'][0]['ex2_public']

    return output
