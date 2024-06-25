import os
import shutil

if __name__ == "__main__":
    src_dir = "/mnt/dev-ssd-8T/shuqixiao/dev/projects/MonoGS/datasets/replica_semantic/room0_gt/"
    sequence_index = "00"

    os.chdir(src_dir)
    os.makedirs("results", exist_ok=True)
    os.makedirs("results_segmentation_labels", exist_ok=True)
    os.makedirs("results_segmentation_maps/Annotations", exist_ok=True)

    sequence_dir = os.path.join(src_dir, "imap", sequence_index)

    os.chdir(sequence_dir)
    os.chdir("rgb")
    rgb_files = os.listdir()
    data_size = len(rgb_files)
    for i in range(data_size):
        shutil.copy(
            "rgb_" + str(i) + ".png",
            os.path.join(src_dir, "results", "frame" + str(i).zfill(6) + ".png"),
        )

    os.chdir(sequence_dir)
    os.chdir("depth")
    depth_files = os.listdir()
    for i in range(data_size):
        shutil.copy(
            "depth_" + str(i) + ".png",
            os.path.join(src_dir, "results", "depth" + str(i).zfill(6) + ".png"),
        )

    os.chdir(sequence_dir)
    os.chdir("semantic_instance")
    segmentation_files = os.listdir()
    for i in range(data_size):
        shutil.copy(
            "vis_sem_instance_" + str(i) + ".png",
            os.path.join(
                src_dir, "results_segmentation_maps/Annotations", "frame" + str(i).zfill(6) + ".png"
            ),
        )

    os.chdir(sequence_dir)
    shutil.copy("traj_w_c.txt", os.path.join(src_dir, "traj.txt"))
