from mmxai.interpretability.classification.torchray.extremal_perturbation.multimodal_extremal_perturbation import multi_extremal_perturbation, image2tensor
import torch
from PIL import Image
from mmf.models.first_model import First_Model

def main():
    image_path = input("entßer your image path : ")
    text = input("enter your text : ")

    model = First_Model.from_pretrained("first_model")
    model.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    image_tensor = image2tensor(image_path)
    mask_, hist_, output_tensor, txt_summary, text_explaination = multi_extremal_perturbation(
        model,
        image_tensor,
        image_path,
        text,
        0,  # 0 non hateful 1 hateful
        max_iter=50,
        areas=[0.12],
    )
    return output_tensor, txt_summary, text_explaination
if __name__ == "__main__":
    main()
