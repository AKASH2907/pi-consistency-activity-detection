import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch


def get_accuracy(predicted_actor, actor):
    maxm, prediction = torch.max(predicted_actor, 1)
    prediction = prediction.view(-1, 1)
    actor = actor.view(-1, 1)
    correct = torch.sum(actor == prediction.float()).item()
    accuracy = correct / float(prediction.shape[0])
    return accuracy


def get_accuracy2(predicted_actor, actor):
    # This gets the f-measure of our network
    predictions = predicted_actor > 0.5

    tp = ((predictions + actor) > 1).sum()
    tn = ((predictions + actor) < 1).sum()
    fp = (predictions > actor).sum()
    fn = (predictions < actor).sum()

    return (tp + tn) / (tp + tn + fp + fn)

def normalize_image(pic):
    # print('type is ',type(pic))
    if pic.min() == 0 and pic.max()==0:
        return(pic)
    else:
        npic = (pic - pic.min()) / (pic.max() - pic.min())
        return npic

def show(image, title='.'):
    # display an image along with title
    # handles PIL format,and numpy arrays
    if isinstance(image, torch.Tensor) and len(image.size()) == 3:
        # image = image.numpy()
        print(image.shape)
        image = image.permute(1, 2, 0)
    # image = normalize_image(image)
    f, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)
    ax.set_title(title, fontsize=30)
    plt.show()

def overlay(img,mask,orig,title='.'):
    masked = np.ma.masked_where(mask == 0, mask)
    img_masked = np.ma.masked_where(img == 0, img)

    img = img + 1
    img[img > 1] = .9


    f, ax = plt.subplots(figsize=(10, 10))
    # pic = orig
    pic = orig[:,1,:,:]
    pic = np.transpose(pic,(1,2,0))
    pic = normalize_image(pic)
    ax.imshow(pic)
    ax.imshow(img_masked,'autumn',interpolation='none', alpha=0.5)
    ax.imshow(masked, 'jet', interpolation='none', alpha=0.5)
    ax.set_title(title, fontsize=30)
    plt.show()

def side(img,mask,orig,title='.'):
    masked = np.ma.masked_where(mask == 0, mask)
    img_masked = np.ma.masked_where(img == 0, img)

    img = img + 1
    img[img > 1] = .9


    f, ax = plt.subplots(figsize=(10, 10))
    # pic = orig
    pic = orig[:,0,:,:]
    pic = np.transpose(pic,(1,2,0))
    pic = normalize_image(pic)
    ax.imshow(pic)
    ax.imshow(img_masked,'autumn',interpolation='none', alpha=0.5)
    ax.set_title(title, fontsize=30)
    plt.show()

def byside(img,mask,orig,title='.'):
    masked = np.ma.masked_where(mask == 0, mask)
    img_masked = np.ma.masked_where(img == 0, img)

    img = img + 1
    img[img > 1] = .9


    f, ax = plt.subplots(figsize=(10, 10))
    # pic = orig
    pic = orig[:,0,:,:]
    pic = np.transpose(pic,(1,2,0))
    pic = normalize_image(pic)
    ax.imshow(pic)
    # ax.imshow(img_masked,'autumn',interpolation='none', alpha=0.5)
    ax.imshow(masked, 'jet', interpolation='none', alpha=0.5)
    ax.set_title(title, fontsize=30)
    plt.show()


def overlay2(mask,orig,title='.'):
    masked = np.ma.masked_where(mask == 0, mask)
    # histogram(masked)
    # img_masked = np.ma.masked_where(img == 0, img)

    # img = img + 1
    # img[img > 1] = .9


    f, ax = plt.subplots(figsize=(10, 10))
    pic = orig
    # pic = orig[:,1,:,:]
    # pic = np.transpose(pic,(1,2,0))
    # pic = normalize_image(pic)
    ax.imshow(pic)
    # ax.imshow(img_masked,'autumn',interpolation='none', alpha=0.5)
    ax.imshow(masked, 'autumn', interpolation='none', alpha=0.5)
    ax.set_title(title, fontsize=30)
    plt.show()

def oldIOU(gt,img,orig):
    #takes ground truth, gt, and and ouput image ,img, and calculates IOU
    # make sure they are in numpy
    #test to see if they are binary - reject or fix if not

    for i in range(0,10):
        intersection = gt[i] + img[i]

        intersection[intersection < 2] = 0
        intersection[intersection > 0] = 1
        intersection_sum = intersection.sum()

        union = gt[i] + img[i]
        union[union > 1] = 1
        union_sum = union.sum()


        IOU = intersection_sum/union_sum
        # overlay(gt[i],img[i],orig[i],IOU)

    return IOU

def IOU(gt,img):
    #takes ground truth, gt, and and ouput image ,img, and calculates IOU
    # make sure they are in numpy
    #test to see if they are binary - reject or fix if not


    intersection = gt + img

    intersection[intersection < 2] = 0
    intersection[intersection > 0] = 1
    intersection_sum = intersection.sum()

    union = gt + img
    union[union > 1] = 1
    union_sum = union.sum()

    if union_sum > 0:
        IOU = intersection_sum/union_sum
    else:
        # print('union sum not positive ',union_sum)
        IOU = torch.Tensor([0])

    return IOU

def IOU2(gt,img):
    #takes ground truth, gt, and and ouput image ,img, and calculates IOU
    # make sure they are in numpy
    #test to see if they are binary - reject or fix if not


    intersection = gt + img

    intersection[intersection < 2] = 0
    intersection[intersection > 0] = 1
    intersection_sum = intersection.sum()

    union = gt + img
    union[union > 1] = 1
    union_sum = union.sum()

    if gt.sum() > 0:
        IOU = intersection_sum/union_sum
    else:
        # print('union sum not positive ',union_sum)
        IOU = float('NaN')

    return IOU

def basic_overlay(img,mask,title='.'):
    masked = np.ma.masked_where(mask == 0, mask)
    img_masked = np.ma.masked_where(img == 0, img)

    img = img + 1
    img[img > 1] = .9


    f, ax = plt.subplots(figsize=(10, 10))

    ax.imshow(img_masked,'autumn',interpolation='none', alpha=0.5)
    ax.imshow(masked, 'jet', interpolation='none', alpha=0.5)
    ax.set_title(title, fontsize=30)
    plt.show()
def testIOU():
    a = np.zeros((10,10))
    a[3:6,3:6]= 1
    b = np.zeros((10, 10))
    b[3:6,3:6]= 1
    iou = IOU(a,b)
    basic_overlay(a,b,iou)

    a = np.zeros((10,10))
    a[3:6,3:6]= 1
    b = np.zeros((10, 10))
    b[7:9,7:9]= 1
    iou = IOU(a,b)
    basic_overlay(a,b,iou)

    a = np.zeros((10,10))
    a[3:6,3:6]= 1
    b = np.zeros((10, 10))
    b[5:8,3:6]= 1
    iou = IOU(a,b)
    basic_overlay(a,b,iou)

    a = np.zeros((10,10))
    a[3:7,3:7]= 1
    b = np.zeros((10, 10))
    b[4:6,4:6]= 1
    iou = IOU(a,b)
    basic_overlay(a,b,iou)
    
def histogram(arr):
    if isinstance(arr, torch.Tensor):
        arr = arr.cpu()
        arr = arr.data.numpy()
    num_bins = 200
    arr = arr.ravel()
    n, bins, patches = plt.hist(arr, num_bins, facecolor='blue', alpha=0.5)
    plt.show()

# def mAP(gt,img):
#     #takes ground truth, gt, and and ouput image ,img, and calculates IOU
#     # make sure they are in numpy
#     #test to see if they are binary - reject or fix if not
#
#
#     intersection = gt + img
#
#     intersection[intersection < 2] = 0
#     intersection[intersection > 0] = 1
#     intersection_sum = intersection.sum()
#
#     gt_sum = gt.sum()
#
#     if gt_sum > 0:
#         overlap = intersection_sum/gt_sum
#     else:
#         # print('union sum not positive ',union_sum)
#         overlap = float('NaN')
#
#     return overlap


if __name__ == "__main__":

    import torchvision.models as models
    from tensorboardX import SummaryWriter
    from torch.autograd import Variable
    import torch
    resnet18 = models.resnet18(False)
    writer = SummaryWriter()
    for name, param in resnet18.named_parameters():
                writer.add_histogram(name, param.clone().cpu().data.numpy(), 0)
    writer.export_scalars_to_json("./all_scalars.json")
    dummy_img = Variable(torch.rand(32, 3, 64, 64))
    res = resnet18(dummy_img)
    writer.add_graph(resnet18, res)
    writer.close()
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--awesome', default='yes')
    # args = parser.parse_args()
    # print('bloop')
    # print(args.awesome)

    # def write_csv(self):
    #     with open('master.csv','w') as csv_file:
    #         writer = csv.writer(csv_file)
    #         for a in self.master:
    #             writer.writerow([a.get('path'),a.get('actor'),a.get('view'),a.get('label'),a.get('frame')])
    #
    # def read_csv(self):
    #     self.master = []
    #     with open('master.csv', 'r') as csv_file:
    #         reader = csv.reader(csv_file)
    #         for row in reader:
    #             self.master.append({'path': row[0], 'actor': row[1], 'view': row[2], 'label': row[3], 'frame': row[4]})


