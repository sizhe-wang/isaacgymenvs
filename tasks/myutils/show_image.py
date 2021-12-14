from PIL import Image
import numpy as np
import torch


if __name__ == '__main__':
    image_tensor = torch.load("/home/v-wewei/code/IsaacGym_Preview3/IsaacGymEnvs/isaacgymenvs/image_tensors_obs_nan.pt")
    image_tensor = image_tensor.view(64, 64)
    print(image_tensor)
    index = torch.isinf(image_tensor).int().nonzero(as_tuple=False)
    print(index[0])
    print(image_tensor[57, 57])
    mask = torch.full_like(image_tensor, 0.).cuda()
    mask1 = torch.full_like(image_tensor, 0, dtype=torch.uint8).cuda()
    mask2 = torch.full_like(image_tensor, 255, dtype=torch.uint8).cuda()
    new_tensor = torch.where(torch.isinf(image_tensor), mask, image_tensor)
    infs = torch.where(torch.isinf(image_tensor), mask2, mask1)

    image_array = new_tensor.cpu().numpy()
    infs_array = infs.cpu().numpy()
    # image_array = image_tensor.cpu().numpy()
    # print(image_array.shape)
    # print(np.min(image_array), np.max(image_array))
    # exit()
    image_array = (image_array * 1.0 - np.min(image_array)) / (np.max(image_array) - np.min(image_array)) * 255
    print(image_array)
    img = Image.fromarray(image_array.astype(np.uint8), mode="L")
    img.show()

    inf_img = Image.fromarray(infs_array.astype(np.uint8), mode='L')
    inf_img.show()
