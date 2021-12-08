import torch


if __name__ == "__main__":
    image_tensors = torch.load("image_tensors/image_tensors.pt")
    # normalization
    # image_tensors = (image_tensors - torch.max(image_tensors)) / (torch.min(image_tensors) - torch.max(image_tensors))
    # image_tensors = image_tensors.view(-1, 1, 64, 64)
    # image_tensors = image_tensors * 2.0 - 1.0
    print(image_tensors.shape)
    print(torch.max(image_tensors))
    print(torch.min(image_tensors))

    # torch.save(image_tensors, "image_tensors/image_tensors.pt")