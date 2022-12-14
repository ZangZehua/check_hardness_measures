import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import shutil
import datetime

CONFIG_TIME = str(datetime.datetime.now().replace(microsecond=0).strftime("%Y-%m-%d-%H-%M-%S"))
SAVE_PATH_BASE = "saved/" + CONFIG_TIME + "/"
if not os.path.isdir(SAVE_PATH_BASE):
    os.makedirs(SAVE_PATH_BASE)
shutil.copyfile("config.py", SAVE_PATH_BASE + "config.py")


class CudaConfig:
    device_ids = [0]


class TrainCMCConfig:
    # base config
    batch_size = 256
    num_workers = 8
    # batch_size = 32
    # num_workers = 0

    epochs = 240

    # paths
    dataset = 'STL-10'
    # data_folder = "d:/data/STL-10"
    # data_folder = "~/zdata/data/STL-10"
    data_folder = "/data/zzh/data/STL-10"

    # resume path
    # resume = "d:/data/cmc_models/ckpt_epoch_220.pth"
    # resume = "~/zdata/data/cmc_models/ckpt_epoch_230.pth"
    resume = "/data/zzh/data/cmc_models/ckpt_epoch_220.pth"

    print_freq = 1
    tb_freq = 500
    save_freq = 1

    # optimization
    learning_rate = 0.03
    lr_decay_epochs = [120, 160, 200]
    lr_decay_rate = 0.1
    beta1 = 0.5
    beta2 = 0.999
    weight_decay = 1e-4
    momentum = 0.9

    # model definition
    model = 'alexnet'
    softmax = False
    nce_k = 16384
    nce_t = 0.07
    nce_m = 0.5
    feat_dim = 128

    # add new views
    view = 'Lab'

    # mixed precision setting
    amp = False
    opt_level = '02'

    # data crop threshold
    candidate_number = 20
    augment_protection = 0.2
    crop_low = 0.2

    # save path
    model_folder = os.path.join(SAVE_PATH_BASE, "models_pt")
    if not os.path.isdir(model_folder):
        os.makedirs(model_folder)

    tb_folder = os.path.join(SAVE_PATH_BASE, "runs_pt")
    if not os.path.isdir(tb_folder):
        os.makedirs(tb_folder)

    # save config
    reward_writer_path = os.path.join(tb_folder, "train_reward")
    rl_loss_writer_path = os.path.join(tb_folder, "rl-loss")
    cl_loss_writer_path = os.path.join(tb_folder, "cl-loss")

    log_path = os.path.join(SAVE_PATH_BASE, "logs_pt.txt")


class LinearProbingConfig:
    # base config
    batch_size = 256
    num_workers = 6
    epochs = 60

    gpu = 0

    # data_folder = "d:/projects/data/STL-10"
    data_folder = "/data/data/STL-10"

    # models to load
    model_paths = ["saved/2022-08-10-17-04-43/models_pt/ckpt_epoch_10.pth",
                   "saved/2022-08-10-17-04-43/models_pt/ckpt_epoch_20.pth",
                   "saved/2022-08-10-17-04-43/models_pt/ckpt_epoch_30.pth",
                   "saved/2022-08-10-17-04-43/models_pt/ckpt_epoch_40.pth",
                   "saved/2022-08-10-17-04-43/models_pt/ckpt_epoch_50.pth",
                   "saved/2022-08-10-17-04-43/models_pt/ckpt_epoch_60.pth",
                   "saved/2022-08-10-17-04-43/models_pt/ckpt_epoch_70.pth",
                   "saved/2022-08-10-17-04-43/models_pt/ckpt_epoch_80.pth",
                   "saved/2022-08-10-17-04-43/models_pt/ckpt_epoch_90.pth",
                   "saved/2022-08-10-17-04-43/models_pt/ckpt_epoch_100.pth",
                   "saved/2022-08-10-17-04-43/models_pt/ckpt_epoch_110.pth",
                   "saved/2022-08-10-17-04-43/models_pt/ckpt_epoch_120.pth",
                   "saved/2022-08-10-17-04-43/models_pt/ckpt_epoch_130.pth",
                   "saved/2022-08-10-17-04-43/models_pt/ckpt_epoch_140.pth",
                   "saved/2022-08-10-17-04-43/models_pt/ckpt_epoch_150.pth",
                   "saved/2022-08-10-17-04-43/models_pt/ckpt_epoch_160.pth",
                   "saved/2022-08-10-17-04-43/models_pt/ckpt_epoch_170.pth",
                   "saved/2022-08-10-17-04-43/models_pt/ckpt_epoch_180.pth",
                   "saved/2022-08-10-17-04-43/models_pt/ckpt_epoch_190.pth",
                   "saved/2022-08-10-17-04-43/models_pt/ckpt_epoch_200.pth",
                   "saved/2022-08-10-17-04-43/models_pt/ckpt_epoch_210.pth",
                   "saved/2022-08-10-17-04-43/models_pt/ckpt_epoch_220.pth",
                   "saved/2022-08-10-17-04-43/models_pt/ckpt_epoch_230.pth",
                   "saved/2022-08-10-17-04-43/models_pt/ckpt_epoch_240.pth"]
    model_path = "saved/2022-08-10-17-04-43/models_pt/ckpt_epoch_10.pth"

    # resume path
    resume = ''

    print_freq = 10
    tb_freq = 500
    save_freq = 5

    # optimization
    learning_rate = 0.1
    lr_decay_epochs = [30, 40, 50]
    lr_decay_rate = 0.2
    momentum = 0.9
    weight_decay = 0
    beta1 = 0.5
    beta2 = 0.999

    # model definition
    model = 'alexnet'

    layer = 5
    dataset = 'STL-10'
    n_label = 10
    view = 'Lab'

    crop_low = 0.2

    # save path
    save_folder = os.path.join(SAVE_PATH_BASE, "models_lp")
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)

    tb_folder = os.path.join(SAVE_PATH_BASE, "runs_lp")
    if not os.path.isdir(tb_folder):
        os.makedirs(tb_folder)

    log_path = os.path.join(SAVE_PATH_BASE, "logs_lp.txt")


class RunnerConfig:
    # common config
    random_seed = 27  # set random seed if required (0 = no random seed)
    torch.manual_seed(random_seed)

    # cuda config
    device_ids = [0]

    # train config
    max_step = 10000000
    memory_size = 10000

    update_target_interval = 100
    print_interval = update_target_interval
    log_interval = update_target_interval


class PPOConfig:
    actor_lr = 0.0003
    critic_lr = 0.0003
    weight_decay = 0.001
    observation_dim = 128 * 2
    cmc_batch_size = 32
    k_epoch = 10

    # resized crop config
    rc_action_dim = 4

    # horizontal flip config
    hf_action_dim = 2
    batch_size = 64
    gamma = 0.99
    lamda = 0.98
    clip_param = 0.2
    update_interval = 2000
    update_episode = 50


class PPONetworkConfig:  # for ppo_continuous and ppo_discrete
    hidden_dim_1 = 128
    hidden_dim_2 = 64



