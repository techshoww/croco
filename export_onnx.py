import onnx
from onnx.shape_inference import infer_shapes
import onnxsim
import torch
from models.croco import CroCoNet
def export_onnx(onnx_path):
    device=torch.device("cpu")

    ckpt = torch.load('CroCo.pth', 'cpu')
    print("ckpt.keys()",ckpt.keys())
    model = CroCoNet( **ckpt.get('croco_kwargs',{})).to(device)
    model.eval()
    msg = model.load_state_dict(ckpt['model'], strict=True)
    
    
    img1 = torch.ones([1,3,224,224],dtype=torch.float32).to(device)
    img2 = torch.ones([1,3,224,224],dtype=torch.float32).to(device)
    nonmask,_ = model.mask_generator(img1)
    inputs = (img1, img2, nonmask)
    torch.onnx.export(model, inputs, onnx_path, input_names=['img1', 'img2', "nonmask" ], output_names=["out", "target"], opset_version=12)
    

    onnx_model = onnx.load(onnx_path)
    onnx_model = infer_shapes(onnx_model)
    # convert model
    model_simp, check = onnxsim.simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, onnx_path)
    print("onnx simpilfy successed, and model saved in {}".format(onnx_path))


if __name__=="__main__":
    export_onnx("croco.onnx")