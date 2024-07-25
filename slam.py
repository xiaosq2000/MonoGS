import os
import sys
import time
from argparse import ArgumentParser
from datetime import datetime

import torch
import torch.multiprocessing as mp
import yaml
from munch import munchify

import wandb
from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.utils.system_utils import mkdir_p
from gui import gui_utils, slam_gui
from utils.config_utils import load_config
from utils.dataset import load_dataset
from utils.eval_utils import eval_ate, eval_rendering, save_gaussians
from utils.logging_utils import Log
from utils.multiprocessing_utils import FakeQueue
from utils.slam_backend import BackEnd
from utils.slam_frontend import FrontEnd


class SLAM:
    def __init__(self, config, save_dir=None, wandb_group_id=None):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()

        self.config = config
        self.save_dir = save_dir
        model_params = munchify(config["model_params"])
        opt_params = munchify(config["opt_params"])
        pipeline_params = munchify(config["pipeline_params"])
        self.model_params, self.opt_params, self.pipeline_params = (
            model_params,
            opt_params,
            pipeline_params,
        )

        if self.config["Dataset"]["type"] == "realsense":
            self.live_mode = True
        elif self.config["Training"]["live_mode"] is True:
            self.live_mode = True
        else:
            self.live_mode = False

        self.use_gui = self.config["Results"]["use_gui"]
        if self.live_mode:
            self.use_gui = True

        self.monocular = self.config["Dataset"]["sensor_type"] == "monocular"
        if self.monocular:
            Log("Depth is unavaiable.")
        else:
            Log("Depth is avaiable.")

        self.semantic = self.config["Dataset"].get("semantic")
        if self.semantic is None:
            self.semantic = False

        if self.semantic:
            self.semantic_embedding_dim = self.config["Training"][
                "semantic_embedding_dim"
            ]
            Log(
                f"Semantics is avaiable and embedding dimension is {self.semantic_embedding_dim}."
            )
        else:
            Log("Semantics is unavaiable.")

        self.use_spherical_harmonics = self.config["Training"]["spherical_harmonics"]
        self.eval_rendering = self.config["Results"]["eval_rendering"]

        model_params.sh_degree = 3 if self.use_spherical_harmonics else 0

        self.gaussians = GaussianModel(model_params.sh_degree, config=self.config)
        # TODO: A magic initial learning rate here?
        self.gaussians.init_lr(6.0)

        self.dataset = load_dataset(
            model_params, model_params.source_path, config=config
        )
        Log(f"Load {type(self.dataset)} with {self.dataset.num_imgs} frames.")

        self.gaussians.training_setup(opt_params)
        bg_color = [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if self.semantic:
            self.background_semantics = torch.zeros(
                self.semantic_embedding_dim,
                dtype=torch.float32,
                device="cuda",
            )
        else:
            self.background_semantics = None

        frontend_queue = mp.Queue()
        backend_queue = mp.Queue()

        q_main2vis = mp.Queue() if self.use_gui else FakeQueue()
        q_vis2main = mp.Queue() if self.use_gui else FakeQueue()

        self.config["Results"]["save_dir"] = save_dir
        self.config["Training"]["monocular"] = self.monocular
        self.config["Training"]["semantic"] = self.semantic

        self.frontend = FrontEnd(self.config)
        self.backend = BackEnd(self.config)

        self.frontend.dataset = self.dataset
        self.frontend.background = self.background
        if self.semantic:
            self.frontend.background_semantics = self.background_semantics
        else:
            self.frontend.background_semantics = None

        self.frontend.pipeline_params = self.pipeline_params
        self.frontend.frontend_queue = frontend_queue
        self.frontend.backend_queue = backend_queue
        self.frontend.q_main2vis = q_main2vis
        self.frontend.q_vis2main = q_vis2main
        self.frontend.set_hyperparams()

        self.backend.gaussians = self.gaussians
        self.backend.background = self.background
        if self.semantic:
            self.backend.background_semantics = self.background_semantics
        else:
            self.backend.background_semantics = None
        self.backend.cameras_extent = 6.0
        self.backend.pipeline_params = self.pipeline_params
        self.backend.opt_params = self.opt_params
        self.backend.frontend_queue = frontend_queue
        self.backend.backend_queue = backend_queue
        self.backend.live_mode = self.live_mode

        self.backend.set_hyperparams()

        self.params_gui = gui_utils.ParamsGUI(
            pipe=self.pipeline_params,
            background=self.background,
            background_semantics=self.background_semantics,
            gaussians=self.gaussians,
            q_main2vis=q_main2vis,
            q_vis2main=q_vis2main,
        )
        if self.use_gui:
            Log("Launch GUI process.", tag="Semantic-3DGS-SLAM")
            gui_process = mp.Process(
                target=slam_gui.run,
                args=(self.params_gui, self.gaussians.semantic_decoder),
            )
            gui_process.start()
            Log("Sleep 10 seconds before GUI available.", tag="Semantic-3DGS-SLAM")
            time.sleep(10)

        Log("Launch backend process.", tag="Semantic-3DGS-SLAM")
        self.backend.wandb_group_id = wandb_group_id
        backend_process = mp.Process(target=self.backend.run)
        backend_process.start()

        Log("Launch frontend process (main process).", tag="Semantic-3DGS-SLAM")
        self.frontend.run()

        Log("Frontend process is stopped.", tag="Semantic-3DGS-SLAM")
        backend_queue.put(["pause"])
        Log("Backend process is notified to pause.", tag="Semantic-3DGS-SLAM")

        end.record()
        torch.cuda.synchronize()
        # empty the frontend queue
        N_frames = len(self.frontend.cameras)
        FPS = N_frames / (start.elapsed_time(end) * 0.001)
        Log("Total time", start.elapsed_time(end) * 0.001, tag="Eval")
        Log("Total FPS", N_frames / (start.elapsed_time(end) * 0.001), tag="Eval")

        if self.eval_rendering:
            self.gaussians = self.frontend.gaussians
            kf_indices = self.frontend.kf_indices
            ATE = eval_ate(
                self.frontend.cameras,
                self.frontend.kf_indices,
                self.save_dir,
                0,
                final=True,
                monocular=self.monocular,
            )

            rendering_result = eval_rendering(
                self.frontend.cameras,
                self.gaussians,
                self.dataset,
                self.save_dir,
                self.pipeline_params,
                self.background,
                self.background_semantics,
                kf_indices=kf_indices,
                iteration="before_opt",
            )
            columns = ["tag", "psnr", "ssim", "lpips", "RMSE ATE", "FPS"]
            metrics_table = wandb.Table(columns=columns)
            metrics_table.add_data(
                "Before",
                rendering_result["mean_psnr"],
                rendering_result["mean_ssim"],
                rendering_result["mean_lpips"],
                ATE,
                FPS,
            )

            # re-used the frontend queue to retrive the gaussians from the backend.
            while not frontend_queue.empty():
                frontend_queue.get()
            Log(
                "Backend process is notified to do color refinement.",
                tag="Semantic-3DGS-SLAM",
            )
            backend_queue.put(["color_refinement"])
            while True:
                if frontend_queue.empty():
                    time.sleep(0.01)
                    continue
                data = frontend_queue.get()
                if data[0] == "sync_backend" and frontend_queue.empty():
                    gaussians = data[1]
                    self.gaussians = gaussians
                    break

            rendering_result = eval_rendering(
                self.frontend.cameras,
                self.gaussians,
                self.dataset,
                self.save_dir,
                self.pipeline_params,
                self.background,
                self.background_semantics,
                kf_indices=kf_indices,
                iteration="after_opt",
            )
            metrics_table.add_data(
                "After",
                rendering_result["mean_psnr"],
                rendering_result["mean_ssim"],
                rendering_result["mean_lpips"],
                ATE,
                FPS,
            )
            wandb.log({"Metrics": metrics_table})
            save_gaussians(self.gaussians, self.save_dir, "final_after_opt", final=True)

        backend_queue.put(["stop"])
        backend_process.join()
        Log("Backend stopped and joined the main thread")
        if self.use_gui:
            q_main2vis.put(gui_utils.GaussianPacket(finish=True))
            gui_process.join()
            Log("GUI Stopped and joined the main thread")

    def run(self):
        pass


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--config", type=str)
    parser.add_argument("--eval", action="store_true")

    args = parser.parse_args(sys.argv[1:])

    mp.set_start_method("spawn")

    with open(args.config, "r") as yml:
        config = yaml.safe_load(yml)

    config = load_config(args.config)
    save_dir = None

    if args.eval:
        Log("Running Semantic-3DGS-SLAM in Evaluation Mode")
        Log("Following config will be overriden")
        Log("\tsave_results=True")
        config["Results"]["save_results"] = True
        Log("\tuse_gui=False")
        config["Results"]["use_gui"] = False
        Log("\teval_rendering=True")
        config["Results"]["eval_rendering"] = True
        Log("\tuse_wandb=True")
        config["Results"]["use_wandb"] = True

    if config["Results"]["save_results"]:
        mkdir_p(config["Results"]["save_dir"])
        current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        path = config["Dataset"]["dataset_path"].split("/")
        save_dir = os.path.join(
            config["Results"]["save_dir"], path[-3] + "_" + path[-2], current_datetime
        )
        tmp = args.config
        tmp = tmp.split(".")[0]
        config["Results"]["save_dir"] = save_dir
        mkdir_p(save_dir)
        with open(os.path.join(save_dir, "config.yml"), "w") as file:
            documents = yaml.dump(config, file)
        Log("Results will be saved in " + save_dir)

    current_datetime = datetime.now().strftime("%Y%m%d")
    wandb_group_id = current_datetime + "-" + wandb.util.generate_id()
    Log(
        f"ID, {wandb_group_id}",
        tag="Semantic-3DGS-SLAM",
    )
    if config["Results"]["use_wandb"]:
        Log("Initialize wandb for frontend process.")
    wandb.init(
        project="Semantic-3DGS-SLAM",
        name="frontend",
        group=wandb_group_id,
        config=config,
        mode=None if config["Results"]["use_wandb"] else "disabled",
    )
    wandb.define_metric("Frame Index")
    wandb.define_metric("Absolute Trajectory Error", step_metric="Frame Index")
    slam = SLAM(
        config,
        save_dir=save_dir,
        wandb_group_id=wandb_group_id if config["Results"]["use_wandb"] else None,
    )

    slam.run()
    wandb.finish()

    # All done
    Log("Done.")
