import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn as nn


def _compute_cam(activations, gradients):
    # GAP over spatial dims on grads -> per-channel weights
    # activations: [B,C,H,W], gradients: [B,C,H,W] (we use B=1)
    if activations is None or gradients is None:
        raise RuntimeError("Activations/gradients not captured. Check target_layer.")
    if activations.dim() != 4 or gradients.dim() != 4:
        raise RuntimeError("Unexpected tensor shapes for CAM computation.")
    if activations.size(0) != 1 or gradients.size(0) != 1:
        # enforce single-sample Grad-CAM
        activations = activations[:1]
        gradients = gradients[:1]

    # per-channel weights via global-average pooling on gradients
    weights = torch.mean(gradients, dim=(0, 2, 3))  # [C]
    cam = torch.sum(weights[None, :, None, None] * activations, dim=1)  # [1,H,W]
    cam = torch.relu(cam)  # Grad-CAM uses ReLU
    cam = cam[0].detach().cpu().numpy().astype(np.float32)
    return cam  # (leave raw; we'll normalize robustly later)


def _normalize(cam):
    
    if cam.size == 0:
        return cam
    # handle constant arrays safely
    if np.allclose(cam.max(), cam.min()):
        return np.zeros_like(cam, dtype=np.float32)
    p_low, p_high = np.percentile(cam, [1, 99])
    if p_high <= p_low:
        cam = cam - cam.min()
        denom = cam.max() + 1e-8
        return (cam / denom).astype(np.float32)
    cam = np.clip(cam, p_low, p_high)
    cam = cam - cam.min()
    denom = cam.max() + 1e-8
    return (cam / denom).astype(np.float32)


def _circular_mask(h, w, margin=4, feather_sigma=6.0):
    # Soft mask to suppress the black corners of the otoscope frame
    cY, cX = h // 2, w // 2
    r = min(cY, cX) - margin
    Y, X = np.ogrid[:h, :w]
    hard = ((X - cX) ** 2 + (Y - cY) ** 2 <= r**2).astype(np.float32)
    # Feather the rim so we don't get a hard ring
    if feather_sigma > 0:
        hard = cv2.GaussianBlur(hard, (0, 0), feather_sigma)
        hard = np.clip(hard, 0.0, 1.0)
    return hard


def _overlay(heatmap, original_bgr, alpha=0.40, colormap=cv2.COLORMAP_JET):
    h, w = original_bgr.shape[:2]

    # Normalize first (robust) then resize
    heatmap = _normalize(heatmap)
    heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)

    # Apply soft circular mask so the rim doesn’t glow
    mask = _circular_mask(h, w, margin=6, feather_sigma=6.0)
    heatmap = heatmap * mask

    # Light smoothing for a cleaner look
    heatmap = cv2.GaussianBlur(heatmap, (9, 9), 0)

    heatmap_uint8 = np.uint8(255 * np.clip(heatmap, 0, 1))
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
    overlay = cv2.addWeighted(original_bgr, 1 - alpha, heatmap_color, alpha, 0)
    return overlay


def _find_last_conv_layer(model, sample_tensor):
    """
    Auto-discover the last Conv2d that actually participates in the forward pass.

    """
    fired = []
    hooks = []

    def _recorder(m, inp, out):
        fired.append(m)

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(_recorder))

    try:
        model.eval()
        with torch.no_grad():
            _ = model(sample_tensor)
    finally:
        for h in hooks:
            h.remove()

    if not fired:
        raise RuntimeError("No Conv2d layers fired during forward pass.")
    return fired[-1]  # last conv that participated


def _find_last_convbnrelu_or_features(model):
    # Fallback candidate modules commonly present in MobileNetV2
    try:
        feats = getattr(model, "features", None)
        if feats is None:
            return None
        # Try the last block (often ConvBNReLU)
        last_block = feats[-1]
        return last_block
    except Exception:
        return None


def get_gradcam(model, input_tensor, target_layer, class_idx=None, smooth=True):
    """
    Returns a Grad-CAM heatmap for `input_tensor` (shape: [1,3,H,W]).
    If `smooth=True`, averages CAMs from the original and a horizontal flip.
    """
    model.eval()
    # make sure grads can flow (some checkpoints get saved with requires_grad=False)
    for p in model.parameters():
        if not p.requires_grad:
            p.requires_grad_(True)

    acts = None
    grads = None

    # Robust hooks: capture activations AND attach a tensor-level grad hook
    def fwd_hook(_, __, output):  # activations from target layer
        nonlocal acts, grads
        acts = output
        grads = None  # reset for each forward

        # attach hook to the tensor to capture its gradient during backward
        try:

            def _save_grad(g):
                nonlocal grads
                grads = g

            output.register_hook(_save_grad)
        except Exception:
            pass

    def bwd_hook(_, grad_input, grad_output):  # grads wrt activations (module-level)
        nonlocal grads
        if grads is None and len(grad_output) > 0:
            grads = grad_output[0]

    h1 = target_layer.register_forward_hook(fwd_hook)
    try:
        # Prefer full backward hook when available
        h2 = target_layer.register_full_backward_hook(bwd_hook)
    except Exception:
        # Fallback for Torch versions where full_backward_hook is flaky
        h2 = target_layer.register_backward_hook(
            lambda module, grad_input, grad_output: bwd_hook(
                module, grad_input, grad_output
            )
        )

    def cam_for_tensor(x):
        nonlocal acts, grads
        acts = grads = None
        # ensure forward+backward run with grad enabled even if outer code disabled it
        with torch.enable_grad():
            out = model(x)
            idx = out.argmax(dim=1).item() if class_idx is None else int(class_idx)
            score = out[0, idx]
            model.zero_grad(set_to_none=True)
            score.backward(retain_graph=True)
        if acts is None or grads is None:
            raise RuntimeError(
                "Failed to capture activations/gradients. Check target_layer."
            )
        return _compute_cam(acts, grads)

    try:
        cam = cam_for_tensor(input_tensor)
        if smooth:
            # Horizontal flip TTA (align back before average)
            x_flip = torch.flip(input_tensor, dims=[3])
            cam_flip = cam_for_tensor(x_flip)
            cam_flip = np.fliplr(cam_flip)  # bring back to original orientation
            cam = (cam + cam_flip) / 2.0
    finally:
        h1.remove()
        h2.remove()

    return cam


def visualize_gradcam(
    model, image_path, transform, target_layer, device, class_idx=None, save_path=None
):
    # Read original with OpenCV (BGR)
    original = cv2.imread(image_path)
    if original is None:
        raise FileNotFoundError(f"Unable to load image at {image_path}")

    # Prepare network input:
    # Use the provided transform directly (it should include Resize to the correct model-specific size)
    rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    inp = transform(pil_img).unsqueeze(0).to(device)

    # Try provided layer → last firing Conv2d → last ConvBNReLU/features[-1]
    try:
        heatmap = get_gradcam(
            model, inp, target_layer, class_idx=class_idx, smooth=False
        )
    except RuntimeError as e:
        if "Failed to capture activations/gradients" in str(e):
            try:
                auto_layer = _find_last_conv_layer(model, inp)
                print(f"ℹ️ Grad-CAM: auto-selected target layer: {auto_layer}")
                heatmap = get_gradcam(
                    model, inp, auto_layer, class_idx=class_idx, smooth=False
                )
            except Exception:
                alt = _find_last_convbnrelu_or_features(model)
                if alt is None:
                    raise
                print(f"ℹ️ Grad-CAM: fallback to last features block: {alt}")
                heatmap = get_gradcam(
                    model, inp, alt, class_idx=class_idx, smooth=False
                )
        else:
            raise

    # Overlay
    overlay = _overlay(heatmap, original, alpha=0.40, colormap=cv2.COLORMAP_JET)

    # Plot
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title("Grad-CAM Overlay")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    if save_path:
        cv2.imwrite(save_path, overlay)
        print(f"📸 Grad-CAM saved to: {save_path}")
