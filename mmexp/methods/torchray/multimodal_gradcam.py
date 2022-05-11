from torchray.attribution.common import Probe, get_module, resize_saliency
from torchray.attribution.grad_cam import gradient_to_grad_cam_saliency



def multimodal_gradcam(model, image_object,
                       question, category_id, 
                       context_builder=None, 
                       resize=False, resize_mode='bilinear',
                       **kwargs):

    
    layer_names = list(zip(*list(model.named_modules())))[0]
    
    # Grad-CAM backprop.
    saliency_layer = get_module(model, 'model.vision_module')
    
    probe = Probe(saliency_layer, target='output')
    
    # Gradient method.
    y = model.classify(image_object, question, explain=True)
    z = y[0, category_id]
    z.backward()
    
    image_tensor = model.image_tensor

    saliency = gradient_to_grad_cam_saliency(probe.data[0])
    
    saliency = resize_saliency(image_tensor, 
                               saliency, 
                               True, 
                               mode=resize_mode,
                               )
    
    return saliency