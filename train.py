import sys
import os
import torch
from utils.feature_extractor import get_Feature_from_DinoV2
from random import randint
from utils.loss_utils import l1_loss, ssim, cosine_similarity_loss
from gaussian_renderer import render
from torchmetrics.functional.regression import pearson_corrcoef
import uuid
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import numpy as np ###
import matplotlib.pyplot as plt ###
from scene import Scene, GaussianModel
### midas ###
from utils.depth_utils import estimate_depth

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    perturbation_viewpoint_stack = None
    ema_loss_for_log = 0.0
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):   
        iter_start.record()
        gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
        
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        rendered_depth = render_pkg["depth"] 
        gt_depth = torch.tensor(viewpoint_cam.depth_image).cuda() 
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)

        depth_weight = 0.05
        loss_depth = depth_weight * (1 - pearson_corrcoef(rendered_depth.reshape(-1, 1)[:, 0], -gt_depth.reshape(-1, 1)[:, 0]))
        
        loss =  (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + depth_weight * loss_depth
        loss_feature = torch.tensor(0).cuda() 
        
        loss_perturbation_depth = torch.tensor(0).cuda() 

        if iteration > 5400:
            if iteration > 5400 and iteration <= 6600:
                if not perturbation_viewpoint_stack:
                    perturbation_viewpoint_stack = scene.getPerturbationCameras(stage=1).copy()
                perturbation_viewpoint_cam = perturbation_viewpoint_stack.pop(randint(0, len(perturbation_viewpoint_stack)-1))
            elif iteration > 6600 and iteration <= 7800:
                if not perturbation_viewpoint_stack:
                    perturbation_viewpoint_stack = scene.getPerturbationCameras(stage=2).copy()
                perturbation_viewpoint_cam = perturbation_viewpoint_stack.pop(randint(0, len(perturbation_viewpoint_stack)-1))
            elif iteration <= 9000:
                if not perturbation_viewpoint_stack:
                    perturbation_viewpoint_stack = scene.getPerturbationCameras(stage=3).copy()
                perturbation_viewpoint_cam = perturbation_viewpoint_stack.pop(randint(0, len(perturbation_viewpoint_stack)-1))

            perturbation_render_pkg = render(perturbation_viewpoint_cam, gaussians, pipe, bg)
            perturbation_image, perturbation_rendered_depth= perturbation_render_pkg["render"], perturbation_render_pkg["depth"]
            ### perturbation depth loss
            pred_depth = estimate_depth(perturbation_image)
            loss_perturbation_depth = (1 - pearson_corrcoef(rendered_depth.reshape(-1, 1)[:, 0], -gt_depth.reshape(-1, 1)[:, 0]))

            if torch.isnan(loss_perturbation_depth).sum() == 0:
                loss += depth_weight * loss_perturbation_depth
            
            ### feature loss
            pred_feature = get_Feature_from_DinoV2(perturbation_image)
            ref_image = perturbation_viewpoint_cam.original_image.cuda()
            ref_feature = get_Feature_from_DinoV2(ref_image)
            loss_feature = cosine_similarity_loss(pred_feature, ref_feature)
            
            feature_loss_weight = 0.05
            loss += feature_loss_weight * loss_feature 

        loss.backward()
        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                print(f"Iteration {iteration}/{opt.iterations} - Loss: {ema_loss_for_log:.7f}")

            if iteration == opt.iterations:
                print("Training complete")

            if (iteration in saving_iterations):
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene.save(iteration)

            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if (iteration in checkpoint_iterations):
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[5_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[2_000])
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
