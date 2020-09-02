
from .image_folder import make_dataset
import torch.utils.data as data
from PIL import Image
from torchvision import transforms


class LoadDataset(data.Dataset):

    def __init__(self, opt, label_transform, img_transform, label_dir, image_dir):

        self.opt = opt
        self.label_transform = label_transform
        self.img_transform = img_transform
        self.label = label_dir
        self.image = image_dir
        self.label_paths, self.image_paths = self.get_paths(label_dir, image_dir)

    def __len__(self):
        return len(self.label_paths)

    def __getitem__(self, index):

        label = Image.open(self.label_paths[index])
        image = Image.open(self.image_paths[index])

        label_tensor = self.label_transform(label).squeeze(0)
        label_tensor = label_tensor * 255.0
        image_tensor = self.img_transform(image)
                 
        '''       
        nc, h, w = label_tensor.size()
        input_label = self.FloatTensor(nc, h, w).zero_()
        semantic_label = input_label.scatter_(1, label_tensor, 1.0)
        '''
        
        data = {'label':label_tensor, 'image':image_tensor}

        return image_tensor, label_tensor



    def get_paths(self, label_dir, image_dir):

        label_paths = make_dataset(label_dir, recursive=False, read_cache=True)
        image_paths = make_dataset(image_dir, recursive=False, read_cache=True)

        assert len(label_paths) == len(image_paths), "The number of images in label_dir and image_dir dont match"

        return label_paths, image_paths


def img_transforms(img_size, normalize=False):

    trans = []
    trans.append(transforms.Resize(size=(img_size, img_size)))
    trans.append(transforms.ToTensor())
    if normalize:
        trans.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.299, 0.224, 0.225]))

    return transforms.Compose(trans)


def get_custom_dataset(opt):
    
    label_transform = img_transforms(opt.img_size, normalize=False)
    img_transform = img_transforms(opt.img_size, normalize=True)
    train_img_data = LoadDataset(opt, label_transform, img_transform, opt.train_label, opt.train_image)
    train_data_loader = data.DataLoader(train_img_data, batch_size = opt.batch_size, shuffle=True, num_workers=4)
    val_img_data = LoadDataset(opt, label_transform, img_transform, opt.val_label, opt.val_image)
    val_data_loader = data.DataLoader(val_img_data, batch_size = opt.batch_size, shuffle=False, num_workers=4)
    return train_img_data, val_img_data
