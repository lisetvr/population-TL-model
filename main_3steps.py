#============================================================
#
#  Main script for training and testing. 
#  If you use this motion model, please our work: 
#  "Predictive online 3D target tracking with population-based 
#  generative networks for image-guided radiotherapy" 
#  Accepted at IPCAI 2021
#
#  github id: lisetvr
#  MedICAL Lab
#============================================================


import warnings
warnings.filterwarnings("ignore")
import argparse
from tensorboardX import SummaryWriter
import datetime
import torchvision
import PIL.Image
from torchvision.transforms import ToTensor
from torch.utils.data.dataloader import DataLoader
from skimage.metrics import structural_similarity as ss

from models import *
from data_loader import *
from utiles import *
from losses import *
from convlstm import *
from convgru import *

parser = argparse.ArgumentParser()
# Path configuration
parser.add_argument('--train_test', type=str, required=True, help='Whether to run training or testing. Options are \"train\" or \"test\"')
parser.add_argument('--data_dir', required=True, help='Path to directory that holds the data')
parser.add_argument('--logging_dir', required=True, help='Path to logging directory')
parser.add_argument('--experiment_name', type=str, default='', help='(optional) Name for experiment')
parser.add_argument('--checkpoint', default='', help='Path to a checkpoint to load model weights from')
parser.add_argument('--VM_checkpoint', default='./VM.pth', help='Path to a Voxelmorph checkpoint to load weights from')

# Optimization parameters
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--lr_policy', type=str, default='plateau', help='Learning rate scheduler policy. Options are \"linear\", \"step\", \"plateau\" or \"cosine\"')
parser.add_argument('--batch_size', type=int, default=10, help='Batch size')

# Others
parser.add_argument('--gpu_idx', type=str, default="0", help='Index of gpu to use if running single GPU')
parser.add_argument('--multi_gpu', type=bool, default=False, help='Whether to use multiple GPUs (DataParallel) to run the code')
parser.add_argument('--input_type', type=int, default=3, help='Input type, whether motion vectors (3) or voxel intensities (1)')
parser.add_argument('--condi_type', type=str, default="2", help='Type of conditioning. Options are \"1\" for sagittal (MRI), \"2\" for coronal (MRI)')

opt = parser.parse_args()
print('\n'.join(["%s: %s" % (key, value) for key, value in vars(opt).items()]))
# --------------------------------------------------------------------------------
device = torch.device("cuda:" + opt.gpu_idx)
stn = SpatialTransformer([32, 64, 64]).to(device)
np.random.seed(seed=123)
criterion = ncc_loss


def train_step1(folds, vm, model, dir_name, max_epoch):
    # -------------------------------------------------------------------------------------------------
    iter = 0
    best_val_loss = np.inf
    optimizerG = torch.optim.Adam(model.parameters(), lr=opt.lr)
    schedulerG = get_scheduler(optimizerG, opt)
    earlyStopper_recon = EarlyStopping(patience=6, verbose=True, delta=0.01)
    # -------------------------------------------------------------------------------------------------
    run_dir = os.path.join(opt.logging_dir, dir_name)
    val_vol_dir = os.path.join(run_dir, 'validation_vols')
    cond_mkdir(run_dir)
    cond_mkdir(val_vol_dir)
    # Save all command line arguments into a txt file in the logging directory for later reference.
    with open(os.path.join(run_dir, "params.txt"), "w") as out_file:
        out_file.write('\n'.join(["%s: %s" % (key, value) for key, value in vars(opt).items()]))
    writer = SummaryWriter(run_dir)
    # -------------------------------------------------------------------------------------------------
    train_set = NAVIGATOR_4D_Dataset_img_seq(opt.data_dir, sequence_list=folds[0])
    valid_set = NAVIGATOR_4D_Dataset_img_seq(opt.data_dir, sequence_list=folds[1], valid=True)
    train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False, num_workers=4)
    # -------------------------------------------------------------------------------------------------
    print('Begin training... STEP 1')
    for epoch in range(0, max_epoch):
        model.train()
        print('Epoch: {}'.format(epoch))
        for ref_volume, _, current_volume, _ in Bar(train_loader):

            optimizerG.zero_grad()
            ref_volume = ref_volume.unsqueeze(1).to(device)
            current_volume = current_volume.unsqueeze(1).to(device)
            # -------------- MODEL ----------------
            dvf = vm(ref_volume, current_volume)
            vmorph_current_volume, generated_dvf, generated_current_volume, dec_in = model(ref_volume, current_volume, dvf)
            # -------------------------------------
            recon_loss = criterion(generated_current_volume, vmorph_current_volume)
            recon_loss.backward()
            optimizerG.step()
            vmorph_recon_loss = criterion(vmorph_current_volume, current_volume)

            writer.add_scalar("vmorph_recon_loss", vmorph_recon_loss, iter)
            writer.add_scalar("recon_loss", recon_loss, iter)

            if iter%2000 == 0:
                writer.add_embedding(dec_in, label_img=current_volume[:, :, 16, :, :], global_step=(epoch+1)*iter)
            iter += 1

        # -------- VALIDATION -------------
        optimizerG.zero_grad()
        with torch.no_grad():
            model.eval()
            val_loss = 0
            for idx, [ref_volume, _, current_volume, _] in enumerate(Bar(valid_loader)):

                ref_volume = ref_volume.unsqueeze(1).to(device)
                current_volume = current_volume.unsqueeze(1).to(device)
                # -------------- MODEL ----------------
                dvf = vm(ref_volume, current_volume)
                vmorph_current_volume, generated_dvf, generated_current_volume, dec_in = model(ref_volume, current_volume, dvf)
                # -------------------------------------

                val_loss += criterion(generated_current_volume, current_volume).item()
            val_loss /= (len(valid_set))

            if val_loss < best_val_loss:
                print("val_loss improved from %0.4f to %0.4f \n" % (best_val_loss, val_loss))
                best_val_loss = val_loss
                custom_save(model, os.path.join(run_dir, 'model_best.pth'))
            else:
                print("val_loss did not improve from %0.4f \n" % (best_val_loss))
                custom_save(model, os.path.join(run_dir, 'model_latest.pth'))

            writer.add_scalar("val_loss", val_loss, iter)
            schedulerG.step(val_loss)
            earlyStopper_recon(val_loss)

        if earlyStopper_recon.early_stop:
            print("Early stopping")
            break


def train_steps_2_3(folds, step, vm, model, dir_name, max_epoch):
    # -------------------------------------------------------------------------------------------------
    iter = 0
    best_val_loss = np.inf
    optimizerG = torch.optim.Adam(model.parameters(), lr=opt.lr)
    schedulerG = get_scheduler(optimizerG, opt)
    if step==2:
        earlyStopper_recon = EarlyStopping(patience=6, verbose=True, delta=25)
    else:
        earlyStopper_recon = EarlyStopping(patience=3, verbose=True, delta=0.01)
    # -------------------------------------------------------------------------------------------------
    run_dir = os.path.join(opt.logging_dir, dir_name)
    val_vol_dir = os.path.join(run_dir, 'validation_vols')
    cond_mkdir(run_dir)
    cond_mkdir(val_vol_dir)
    # Save all command line arguments into a txt file in the logging directory for later reference.
    with open(os.path.join(run_dir, "params.txt"), "w") as out_file:
        out_file.write('\n'.join(["%s: %s" % (key, value) for key, value in vars(opt).items()]))
    writer = SummaryWriter(run_dir)
    # -------------------------------------------------------------------------------------------------
    train_set = NAVIGATOR_4D_Dataset_img_seq(opt.data_dir, sequence_list=folds[0])
    valid_set = NAVIGATOR_4D_Dataset_img_seq(opt.data_dir, sequence_list=folds[1], valid=True)
    train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False, num_workers=4)
    # -------------------------------------------------------------------------------------------------
    print('Begin training... STEP ' + str(step))
    for epoch in range(0, max_epoch):
        model.train()
        print('Epoch: {}'.format(epoch))

        for ref_volume, input_volume_list, current_volume, _ in Bar(train_loader):

            optimizerG.zero_grad()
            ref_volume = ref_volume.unsqueeze(1).to(device)
            current_volume = current_volume.unsqueeze(1).to(device)
            dvf = vm(ref_volume, current_volume)

            if opt.condi_type == "1":
                c1 = list()
                c1_ref = ref_volume[:, :, 16, :, :]
                for q in range(3):
                    c1temp = input_volume_list[q].unsqueeze(1)[:, :, 16, :, :]
                    c1.append(torch.cat([c1temp.to(device), c1_ref.to(device)], dim=1))
                c1 = torch.stack(c1, dim=2).to(device)
                c2 = None
            else:
                c2 = list()
                c2_ref = ref_volume[:, :, :, 32, :]
                for q in range(3):
                    c2temp = input_volume_list[q].unsqueeze(1)[:, :, :, 32, :]
                    c2.append(torch.cat([c2temp.to(device), c2_ref.to(device)], dim=1))
                c1 = None
                c2 = torch.stack(c2, dim=2).to(device)

            generated_dvf, generated_current_volume, l2_loss = model(current_volume, ref_volume, c1, c2, dvf)

            recon_loss = criterion(generated_current_volume, current_volume)
            if step == 2:
                loss = l2_loss  # step 2
            else:
                loss = 0.001*l2_loss + recon_loss  # step 3

            # -------------------------------------
            loss.backward()
            optimizerG.step()
            writer.add_scalar("recon_loss", recon_loss, iter)
            writer.add_scalar("l2_loss", l2_loss, iter)
            iter += 1

        # -------- VALIDATION -------------
        optimizerG.zero_grad()
        with torch.no_grad():
            model.eval()
            val_loss = 0

            for idx, [ref_volume, input_volume_list, current_volume, _] in enumerate(Bar(valid_loader)):

                ref_volume = ref_volume.unsqueeze(1).to(device)
                current_volume = current_volume.unsqueeze(1).to(device)
                dvf = vm(ref_volume, current_volume)

                if opt.condi_type == "1":
                    c1 = list()
                    c1_ref = ref_volume[:, :, 16, :, :]
                    for q in range(3):
                        c1temp = input_volume_list[q].unsqueeze(1)[:, :, 16, :, :]
                        c1.append(torch.cat([c1temp.to(device), c1_ref.to(device)], dim=1))
                    c1 = torch.stack(c1, dim=2).to(device)
                    c2 = None
                else:
                    c2 = list()
                    c2_ref = ref_volume[:, :, :, 32, :]
                    for q in range(3):
                        c2temp = input_volume_list[q].unsqueeze(1)[:, :, :, 32, :]
                        c2.append(torch.cat([c2temp.to(device), c2_ref.to(device)], dim=1))
                    c1 = None
                    c2 = torch.stack(c2, dim=2).to(device)

                generated_dvf, generated_current_volume = model(None, ref_volume, c1, c2, dvf)
                recon_loss = criterion(generated_current_volume, current_volume)
                val_loss += recon_loss
            val_loss /= (len(valid_set))

            if val_loss < best_val_loss:
                print("val_loss improved from %0.4f to %0.4f \n" % (best_val_loss, val_loss))
                best_val_loss = val_loss
                custom_save(model, os.path.join(run_dir, 'model_best.pth'))
            else:
                print("val_loss did not improve from %0.4f \n" % (best_val_loss))
                custom_save(model, os.path.join(run_dir, 'model_latest.pth'))

            writer.add_scalar("val_loss", val_loss, iter)
            schedulerG.step(val_loss)
            earlyStopper_recon(val_loss)

        if earlyStopper_recon.early_stop:
            print("Early stopping")
            break


def test(fold, vm, model, dir_name):

        with torch.no_grad():
            model.eval()
            test_set = NAVIGATOR_4D_Dataset_img_seq(opt.data_dir, sequence_list=fold, test=True)
            test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4)

            MSE_loss, NCC_loss, SSIM_loss = [], [], []
            mse = nn.MSELoss(reduction='mean').to(device)

            for idx, [ref_volume, input_volume_list, current_volume, _] in enumerate(Bar(test_loader)):

                ref_volume = ref_volume.unsqueeze(1).to(device)
                current_volume = current_volume.unsqueeze(1).to(device)
                dvf = vm(ref_volume, current_volume)
                vmorph_volume = stn(ref_volume, dvf)

                if opt.condi_type == "1":
                    c1 = list()
                    c1_ref = ref_volume[:, :, 16, :, :]
                    for q in range(3):
                        c1temp = input_volume_list[q].unsqueeze(1)[:, :, 16, :, :]
                        c1.append(torch.cat([c1temp.to(device), c1_ref.to(device)], dim=1))
                    c1 = torch.stack(c1, dim=2).to(device)
                    c2 = None
                else:
                    c2 = list()
                    c2_ref = ref_volume[:, :, :, 32, :]
                    for q in range(3):
                        c2temp = input_volume_list[q].unsqueeze(1)[:, :, :, 32, :]
                        c2.append(torch.cat([c2temp.to(device), c2_ref.to(device)], dim=1))
                    c1 = None
                    c2 = torch.stack(c2, dim=2).to(device)

                generated_dvf, generated_current_volume = model(None, ref_volume, c1, c2, dvf)

                NCC_loss.append(ncc_loss(generated_current_volume, vmorph_volume, device=device).item())
                MSE_loss.append(mse(generated_current_volume, vmorph_volume).item())
                SSIM_loss.append(ss(generated_current_volume[0, 0, :, :, :].detach().cpu().numpy(),
                                    vmorph_volume[0, 0, :, :, :].detach().cpu().numpy()))

            np.save(os.path.join(opt.logging_dir, dir_name, "NCC_loss.npy"), np.asarray(NCC_loss))
            np.save(os.path.join(opt.logging_dir, dir_name, "MSE_loss.npy"), np.asarray(MSE_loss))
            np.save(os.path.join(opt.logging_dir, dir_name, "SSIM_loss.npy"), np.asarray(SSIM_loss))
            print("\nTest set average loss NCC: %0.4f, MSE: %0.4f, SSIM: %0.4f" % (np.mean(np.asarray(NCC_loss)),
                                                                                   np.mean(np.asarray(MSE_loss)),
                                                                                   np.mean(np.asarray(SSIM_loss))))


def main():
    if opt.train_test == "train":
        train_folds_all, valid_folds_all, test_folds_all = make_folds_originalversion()

        for fold in range(25):  # 25 folds (leave-one-out)

            train_folds = train_folds_all[fold]
            valid_folds = valid_folds_all[fold]
            vm = Voxelmorph([32, 64, 64], [16, 32, 32, 32], [32, 32, 32, 32, 32, 16, 16], full_size=True).to(device)
            custom_load(vm, opt.VM_checkpoint, device)

            for step in range(1, 4):

                dir_name = 'f' + str(fold) + "/s" + str(step)
                # ---- Select model according to the step ----
                if step==1:
                    model = AE(in_channels=opt.input_type).to(device)
                    max_epoch = 5
                elif step==2:
                    model = TLNet_dvf_3S(in_channels=opt.input_type, train_step=step).to(device)
                    custom_load(model.AE, opt.logging_dir + "f" + str(fold) + "/s1/model_best.pth", device)
                    max_epoch = 10
                else:
                    model = TLNet_dvf_3S(in_channels=opt.input_type, train_step=step).to(device)
                    custom_load(model, opt.logging_dir + "f" + str(fold) + "/s2/model_best.pth", device)
                    max_epoch = 10

                # --- Select training function according to the step ----
                if step==1:
                    train_step1((train_folds, valid_folds), vm=vm, model=model, dir_name=dir_name, max_epoch=max_epoch)
                else:
                    train_steps_2_3((train_folds, valid_folds), step=step, vm=vm, model=model, dir_name=dir_name, max_epoch=max_epoch)

    elif opt.train_test == "test":

        train_folds_all, valid_folds_all, test_folds_all = make_folds_originalversion()

        for fold in range(25):
            test_folds = test_folds_all[fold]
            vm = Voxelmorph([32, 64, 64], [16, 32, 32, 32], [32, 32, 32, 32, 32, 16, 16], full_size=True).to(device)
            custom_load(vm, opt.VM_checkpoint, device)
            dir_name = 'f' + str(fold) + "/test/"
            model = TLNet_dvf_3S(in_channels=opt.input_type, train_step=3).to(device)
            custom_load(model, opt.checkpoint + "f" + str(fold) + "/s3/model_best.pth", device)
            test(test_folds, vm, model, dir_name)

    else:
        print("Unknown mode for train_test argument:{}".format(opt.train_test))


if __name__ == "__main__":
    main()
