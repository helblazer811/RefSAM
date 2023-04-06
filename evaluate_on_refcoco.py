import argparse
import os
import warnings
from tqdm import tqdm
import numpy as np

import cv2
import torch
import torch.nn.parallel
import torch.utils.data

from tools.dataset import RefDataset

from model import RefSAMModel
from PIL import Image, ImageDraw

warnings.filterwarnings("ignore")
cv2.setNumThreads(0)

def run_evaluation(model, dataloader, args):
    """
        Evaluate the SAM based approach on RefCOCO
    """
    print("Starting evaluation on RefCOCO")
    iou_list = []
    tbar = tqdm(dataloader, desc='Inference:', ncols=100)
    model.eval()
    for img, param in tbar:
        # data
        img = img.cuda(non_blocking=True)
        mask = cv2.imread(param['mask_dir'][0], flags=cv2.IMREAD_GRAYSCALE)
        # dump image & mask
        if args.visualize:
            seg_id = param['seg_id'][0].cpu().numpy()
            img_name = '{}-img.jpg'.format(seg_id)
            mask_name = '{}-mask.png'.format(seg_id)
            cv2.imwrite(filename=os.path.join(args.vis_dir, img_name),
                        img=param['ori_img'][0].cpu().numpy())
            cv2.imwrite(filename=os.path.join(args.vis_dir, mask_name),
                        img=mask)
        # multiple sentences
        for sent in param['sents']:
            mask = mask / 255.
            # text = tokenize(sent, args.word_len, True)
            # text = text.cuda(non_blocking=True)
            # inference
            text = sent
            image = img.squeeze()
            # convert tensor image to cv2 imread format
            image = image.permute(1, 2, 0)
            image = image.cpu().numpy()
            # img = torch.permute(img, (1, 2, 0))
            pred, mask_image = model(image, text)
            # pred = torch.sigmoid(pred)
            """
            if pred.shape[-2:] != img.shape[-2:]:
                pred = F.interpolate(pred,
                                     size=img.shape[-2:],
                                     mode='bicubic',
                                     align_corners=True).squeeze()
            """
            # process one sentence
            h, w = param['ori_size'].numpy()[0]
            mat = param['inverse'].numpy()[0]
            # convert pred to float32 
            pred_orig = pred
            pred = pred.astype(np.float32)
            pred = cv2.warpAffine(
                pred,
                mat, 
                (w, h),
                flags=cv2.INTER_CUBIC,
                borderValue=0.
            )
            # pred = np.array(pred > 0.35)
            # iou
            inter = np.logical_and(pred, mask)
            union = np.logical_or(pred, mask)
            iou = np.sum(inter) / (np.sum(union) + 1e-6)
            iou_list.append(iou)
            # dump prediction
            if args.visualize:
                """
                ori_img = cv2.warpAffine(
                    img.squeeze().detach().cpu(),
                    mat, 
                    (w, h),
                    flags=cv2.INTER_CUBIC,
                    borderValue=0.
                )
                """
                # ori_img = img.squeeze().cpu().numpy()
                # ori_img = np.transpose(ori_img, (1, 2, 0))
                ori_img = param['ori_img'].detach().cpu().numpy().squeeze()
                ori_img = cv2.cvtColor(ori_img, cv2.COLOR_RGB2BGR)
                original_image = Image.fromarray(ori_img)

                overlay_image = Image.new('RGBA', (w, h), (0, 0, 0, 0))
                overlay_color = (255, 0, 0, 150)
                draw = ImageDraw.Draw(overlay_image)
                pred = Image.fromarray(pred.astype(np.uint8) * 255)
                draw.bitmap((0, 0), pred, fill=overlay_color)

                print(original_image.size, overlay_image.size)
                result_image = Image.alpha_composite(
                    original_image.convert('RGBA'),
                    overlay_image
                )

                sent = "_".join(sent[0].split(" "))
                pred_name = '{}-iou={:.2f}-{}.png'.format(seg_id, iou*100, sent)

                result_image.save(
                    os.path.join(args.vis_dir, pred_name)
                )
                """
                pred = np.array(
                    pred*255, 
                    dtype=np.uint8
                )
                sent = "_".join(sent[0].split(" "))
                pred_name = '{}-iou={:.2f}-{}.png'.format(seg_id, iou*100, sent)
                cv2.imwrite(
                    filename=os.path.join(args.vis_dir, pred_name),
                    img=pred
                )
                """
    print('=> Metric Calculation <=')
    iou_list = np.stack(iou_list)
    iou_list = torch.from_numpy(iou_list).to(img.device)
    prec_list = []
    for thres in torch.arange(0.5, 1.0, 0.1):
        tmp = (iou_list > thres).float().mean()
        prec_list.append(tmp)
    iou = iou_list.mean()
    prec = {}
    for i, thres in enumerate(range(5, 10)):
        key = 'Pr@{}'.format(thres*10)
        value = prec_list[i].item()
        prec[key] = value
    print('IoU={:.2f}'.format(100.*iou.item()))
    for k, v in prec.items():
        print('{}: {:.2f}.'.format(k, 100.*v))

    return iou.item(), prec

def main(model, args):
    # Build dataset & dataloader
    print("Loading the RefDataset")
    test_data = RefDataset(
        lmdb_dir=args.test_lmdb,
        mask_dir=args.mask_root,
        dataset=args.dataset,
        split=args.test_split,
        mode='test',
        input_size=args.input_size,
        word_length=args.word_len
    )

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )
    # Build model
    model = torch.nn.DataParallel(model).cuda()
    # Run the evaluation
    run_evaluation(model, test_loader, args)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Pytorch Referring Expression Segmentation')

    parser.add_argument(
        "--dataset",
        default="refcoco",
    )
    parser.add_argument(
        "--test_split",
        default="val",
    )
    parser.add_argument(
        "--test_lmdb",
        default="datasets_link/lmdb/refcoco/val.lmdb",
    )
    parser.add_argument(
        "--mask_root",
        default="datasets_link/masks/refcoco",
    )
    parser.add_argument(
        "--sam_model_path",
        default="pretrained/sam_vit_h_4b8939.pth",
    )
    parser.add_argument(
        "--vis_dir",
        default="vis",
    )
    parser.add_argument(
        "--visualize",
        default=True
    )
    parser.add_argument(
        "--word_len",
        default=17,
    )
    parser.add_argument(
        "--input_size",
        default=416,
    )
    args = parser.parse_args()
    # Load the model
    print("Loading the RefSAMModel")
    model = RefSAMModel(
        sam_model_path=args.sam_model_path,
    )
    print("Running the evaluation")
    main(model, args)
