# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#


# Basic libs
import torch
import torch.nn as nn
import numpy as np
from os import makedirs, listdir
from os.path import exists, join
import time
import json
from sklearn.neighbors import KDTree

# PLY reader
from pids_core.utils.ply import read_ply, write_ply

# Metrics
from pids_core.utils.metrics import IoU_from_confusions, fast_confusion
from sklearn.metrics import confusion_matrix

#from utils.visualizer import show_ModelNet_models
from nasflow.optim.ema import EMA

# ----------------------------------------------------------------------------------------------------------------------
#
#           Tester Class
#       \******************/
#

class ModelTester:

    # Initialization methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, net, chkp_path=None, on_gpu=True, load_ckpt=True):

        ############
        # Parameters
        ############

        # Choose to train on CPU or GPU
        if on_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        net.to(self.device)

        ##########################
        # Load previous checkpoint
        ##########################
        if chkp_path and  exists(chkp_path) and load_ckpt:
            checkpoint = torch.load(chkp_path)
            net.load_state_dict(checkpoint['model_state_dict'], strict=False)
            self.epoch = checkpoint['epoch']
        net.eval()
        print("Model and training state restored.")

        return

    # Test main methods
    # ------------------------------------------------------------------------------------------------------------------

    def classification_test(self, net, test_loader, config, debug=False, model_root=None,
                            use_vote_acc=True):

        ############
        # Initialize
        ############
        smooth_factor = .95

        # Choose test smoothing parameter (0 for no smothing, 0.99 for big smoothing)
        softmax = torch.nn.Softmax(1)

        # Number of classes including ignored labels
        nc_tot = test_loader.dataset.num_classes

        # Number of classes predicted by the model
        nc_model = config.num_classes

        # Initiate global prediction over test clouds
        self.test_probs = np.zeros((test_loader.dataset.num_models, nc_model))
        self.test_counts = np.zeros((test_loader.dataset.num_models, nc_model))

        t = [time.time()]
        mean_dt = np.zeros(1)
        last_display = time.time()
        while np.min(self.test_counts) < config.num_votes:
            # Run model on all test examples
            # ******************************
            # Initiate result containers
            probs = []
            targets = []
            obj_inds = []
            lat = 0
            num_b = 0
            # Start validation loop
            for batch in test_loader:
                num_b += 1

                if 'cuda' in self.device.type:
                    batch.to(self.device)

                # New time
                t = t[-1:]
                t += [time.time()]

                # Forward pass
                outputs = net(batch, config)

                # Average timing
                t += [time.time()]
                dt = np.array(t[1:]) - np.array(t[:-1])
                fra = 0.9
                warm_up = 5
                #if i < warm_up:
                #    mean_dt = dt
                #else:
                mean_dt = fra * mean_dt + (1-fra) * dt
                lat = 1000 * (mean_dt[1])

                # Get probs and labels
                probs += [softmax(outputs).cpu().detach().numpy()]
                targets += [batch.labels.cpu().numpy()]
                obj_inds += [batch.model_inds.cpu().numpy()]

                # Display
                if (t[-1] - last_display) > 1.0:
                    last_display = t[-1]
                    message = 'Test vote {:.0f} : {:.1f}% (timings : {:4.2f} {:4.2f})'
                    print(message.format(np.min(self.test_counts),
                                         100 * len(obj_inds) / config.validation_size,
                                         1000 * (mean_dt[0]),
                                         1000 * (mean_dt[1])))

            # Stack all validation predictions
            probs = np.vstack(probs)
            targets = np.hstack(targets)
            obj_inds = np.hstack(obj_inds)

            if np.any(test_loader.dataset.input_labels[obj_inds] != targets):
                raise ValueError('wrong object indices')

            # Compute incremental average (predictions are always ordered): Vote accuracy
            if use_vote_acc:
                self.test_counts[obj_inds] += 1
                self.test_probs[obj_inds] = self.test_probs[obj_inds] * smooth_factor + probs * (1 - smooth_factor)
                # self.test_probs[obj_inds] += (probs - self.test_probs[obj_inds]) / (self.test_counts[obj_inds])
            else:
                # Non-vote accuracy
                self.test_counts[obj_inds] += 1
                self.test_probs[obj_inds] = probs            
            # Save/Display temporary results
            # ******************************

            test_labels = np.array(test_loader.dataset.label_values)

            # Compute classification results
            C1 = fast_confusion(test_loader.dataset.input_labels,
                                np.argmax(self.test_probs, axis=1),
                                test_labels)
            ACC = 100 * np.sum(np.diag(C1)) / (np.sum(C1) + 1e-6)
            print("Confusion:", C1)
            print('Test Accuracy = {:.1f}%'.format(ACC))

        # return ACC, np.mean(LAT[warm_up:])
        return ACC

    def cloud_segmentation_test(self, net, test_loader, config, debug=False, model_root=None):
        # Use num_votes=100 for real testing scenario.
        """
        Test method for cloud segmentation models
        """

        ############
        # Initialize
        ############

        # Choose test smoothing parameter (0 for no smothing, 0.99 for big smoothing)
        test_smooth = 0.99
        test_radius_ratio = 0.7
        softmax = torch.nn.Softmax(1)

        # Number of classes including ignored labels
        nc_tot = test_loader.dataset.num_classes

        # Number of classes predicted by the model
        nc_model = config.num_classes

        # Initiate global prediction over test clouds
        self.test_probs = [np.zeros((l.shape[0], nc_model)) for l in test_loader.dataset.input_labels]

        # Test saving path
        if config.saving:
            if model_root is not None:
                test_path = join(model_root, 'test', config.saving_path.split('/')[-1])
            else:
                test_path = join('test', config.saving_path.split('/')[-1])
            if not exists(test_path):
                makedirs(test_path)
            if not exists(join(test_path, 'predictions')):
                makedirs(join(test_path, 'predictions'))
            if not exists(join(test_path, 'probs')):
                makedirs(join(test_path, 'probs'))
            if not exists(join(test_path, 'potentials')):
                makedirs(join(test_path, 'potentials'))
        else:
            test_path = None

        # If on validation directly compute score
        if test_loader.dataset.set == 'validation':
            val_proportions = np.zeros(nc_model, dtype=np.float32)
            i = 0
            for label_value in test_loader.dataset.label_values:
                if label_value not in test_loader.dataset.ignored_labels:
                    val_proportions[i] = np.sum([np.sum(labels == label_value)
                                                 for labels in test_loader.dataset.validation_labels])
                    i += 1
        else:
            val_proportions = None

        #####################
        # Network predictions
        #####################

        test_epoch = 0
        last_min = -0.5

        t = [time.time()]
        last_display = time.time()
        mean_dt = np.zeros(1)

        # Start test loop
        while True:
            print('Initialize workers')
            lat = 0
            num_b = 0
            for i, batch in enumerate(test_loader):
                num_b += 1

                if 'cuda' in self.device.type:
                    batch.to(self.device)

                # New time
                t = t[-1:]
                t += [time.time()]

                # Forward pass
                outputs = net(batch, config)

                if 'cuda' in self.device.type:
                    torch.cuda.synchronize(self.device)

                t += [time.time()]

                # Get probs and labels
                stacked_probs = softmax(outputs).cpu().detach().numpy()
                s_points = batch.points[0].cpu().numpy()
                lengths = batch.lengths[0].cpu().numpy()
                in_inds = batch.input_inds.cpu().numpy()
                cloud_inds = batch.cloud_inds.cpu().numpy()

                # if 'cuda' in self.device.type:
                #    torch.cuda.synchronize(self.device)

                # Get predictions and labels per instance
                # ***************************************

                i0 = 0
                for b_i, length in enumerate(lengths):
                    # Get prediction
                    points = s_points[i0:i0 + length]
                    probs = stacked_probs[i0:i0 + length]
                    inds = in_inds[i0:i0 + length]
                    c_i = cloud_inds[b_i]

                    if 0 < test_radius_ratio < 1:
                        mask = np.sum(points ** 2, axis=1) < (test_radius_ratio * config.in_radius) ** 2
                        inds = inds[mask]
                        probs = probs[mask]

                    # Update current probs in whole cloud
                    self.test_probs[c_i][inds] = test_smooth * self.test_probs[c_i][inds] + (1 - test_smooth) * probs
                    i0 += length

                # Average timing
                t += [time.time()]
                dt = np.array(t[1:]) - np.array(t[:-1])
                fra1 = 0.9
                fra2 = 0.1
                warm_up = 5
                if i < warm_up:
                    mean_dt = dt
                else:
                    mean_dt = fra1 * mean_dt + fra2 * dt

                # Display
                if (t[-1] - last_display) > 1.0:
                    last_display = t[-1]
                    message = 'e{:03d}-i{:04d} => {:.1f}% (timings : {:4.2f} {:4.2f} {:4.2f})'
                    print(message.format(test_epoch, i,
                                         100 * i / config.validation_size,
                                         1000 * (mean_dt[0]),
                                         1000 * (mean_dt[1]),
                                         1000 * (mean_dt[2])))

            # Update minimum od potentials
            new_min = torch.min(test_loader.dataset.min_potentials)
            print('Test epoch {:d}, end. Min potential = {:.1f}'.format(test_epoch, new_min))
            #print([np.mean(pots) for pots in test_loader.dataset.potentials])

            # Save predicted cloud
            if last_min + 1 < new_min:

                # Update last_min
                last_min += 1

                # Show vote results (On subcloud so it is not the good values here)
                if test_loader.dataset.set == 'validation':
                    print('\nConfusion on sub clouds')
                    Confs = []
                    for i, file_path in enumerate(test_loader.dataset.files):

                        # Insert false columns for ignored labels
                        probs = np.array(self.test_probs[i], copy=True)
                        for l_ind, label_value in enumerate(test_loader.dataset.label_values):
                            if label_value in test_loader.dataset.ignored_labels:
                                probs = np.insert(probs, l_ind, 0, axis=1)

                        # Predicted labels
                        preds = test_loader.dataset.label_values[np.argmax(probs, axis=1)].astype(np.int32)

                        # Targets
                        targets = test_loader.dataset.input_labels[i]

                        # Confs
                        Confs += [fast_confusion(targets, preds, test_loader.dataset.label_values)]

                    # Regroup confusions
                    C = np.sum(np.stack(Confs), axis=0).astype(np.float32)

                    # Remove ignored labels from confusions
                    for l_ind, label_value in reversed(list(enumerate(test_loader.dataset.label_values))):
                        if label_value in test_loader.dataset.ignored_labels:
                            C = np.delete(C, l_ind, axis=0)
                            C = np.delete(C, l_ind, axis=1)

                    # Rescale with the right number of point per class
                    C *= np.expand_dims(val_proportions / (np.sum(C, axis=1) + 1e-6), 1)

                    # Compute IoUs
                    IoUs = IoU_from_confusions(C)
                    mIoU = np.mean(IoUs)
                    s = '{:5.2f} | '.format(100 * mIoU)
                    for IoU in IoUs:
                        s += '{:5.2f} '.format(100 * IoU)
                    print(s + '\n')

                # Save real IoU once in a while
                if int(np.ceil(new_min)) % 2 == 0:

                    # Project predictions
                    print('\nReproject Vote #{:d}'.format(int(np.floor(new_min))))
                    t1 = time.time()
                    proj_probs = []
                    for i, file_path in enumerate(test_loader.dataset.files):

                        print(i, file_path, test_loader.dataset.test_proj[i].shape, self.test_probs[i].shape)

                        print(test_loader.dataset.test_proj[i].dtype, np.max(test_loader.dataset.test_proj[i]))
                        print(test_loader.dataset.test_proj[i][:5])

                        # Reproject probs on the evaluations points
                        probs = self.test_probs[i][test_loader.dataset.test_proj[i], :]
                        proj_probs += [probs]

                    t2 = time.time()
                    print('Done in {:.1f} s\n'.format(t2 - t1))

                    # Show vote results
                    if test_loader.dataset.set == 'validation':
                        print('Confusion on full clouds')
                        t1 = time.time()
                        Confs = []
                        for i, file_path in enumerate(test_loader.dataset.files):

                            # Insert false columns for ignored labels
                            for l_ind, label_value in enumerate(test_loader.dataset.label_values):
                                if label_value in test_loader.dataset.ignored_labels:
                                    proj_probs[i] = np.insert(proj_probs[i], l_ind, 0, axis=1)

                            # Get the predicted labels
                            preds = test_loader.dataset.label_values[np.argmax(proj_probs[i], axis=1)].astype(np.int32)

                            # Confusion
                            targets = test_loader.dataset.validation_labels[i]
                            Confs += [fast_confusion(targets, preds, test_loader.dataset.label_values)]

                        t2 = time.time()
                        print('Done in {:.1f} s\n'.format(t2 - t1))

                        # Regroup confusions
                        C = np.sum(np.stack(Confs), axis=0)

                        # Remove ignored labels from confusions
                        for l_ind, label_value in reversed(list(enumerate(test_loader.dataset.label_values))):
                            if label_value in test_loader.dataset.ignored_labels:
                                C = np.delete(C, l_ind, axis=0)
                                C = np.delete(C, l_ind, axis=1)

                        IoUs = IoU_from_confusions(C)
                        mIoU_full = np.mean(IoUs)
                        s = '{:5.2f} | '.format(100 * mIoU_full)
                        for IoU in IoUs:
                            s += '{:5.2f} '.format(100 * IoU)
                        print('-' * len(s))
                        print(s)
                        print('-' * len(s) + '\n')

                        if config.no_validation:
                            return mIoU_full
                    # Save predictions. Disabled currently to save time.
                    if 1:
                        print('Saving clouds')
                        t1 = time.time()
                        for i, file_path in enumerate(test_loader.dataset.files):

                            # Get file
                            points = test_loader.dataset.load_evaluation_points(file_path)

                            # Get the predicted labels
                            preds = test_loader.dataset.label_values[np.argmax(proj_probs[i], axis=1)].astype(np.int32)

                            # Save plys
                            cloud_name = file_path.split('/')[-1]
                            test_name = join(test_path, 'predictions', cloud_name)
                            write_ply(test_name,
                                      [points, preds],
                                      ['x', 'y', 'z', 'preds'])
                            test_name2 = join(test_path, 'probs', cloud_name)
                            prob_names = ['_'.join(test_loader.dataset.label_to_names[label].split())
                                          for label in test_loader.dataset.label_values]
                            write_ply(test_name2,
                                      [points, proj_probs[i]],
                                      ['x', 'y', 'z'] + prob_names)

                            # Save potentials
                            pot_points = np.array(test_loader.dataset.pot_trees[i].data, copy=False)
                            pot_name = join(test_path, 'potentials', cloud_name)
                            pots = test_loader.dataset.potentials[i].numpy().astype(np.float32)
                            write_ply(pot_name,
                                      [pot_points.astype(np.float32), pots],
                                      ['x', 'y', 'z', 'pots'])

                            # Save ascii preds
                            if test_loader.dataset.set == 'test':
                                if test_loader.dataset.name.startswith('Semantic3D'):
                                    ascii_name = join(test_path, 'predictions', test_loader.dataset.ascii_files[cloud_name])
                                else:
                                    ascii_name = join(test_path, 'predictions', cloud_name[:-4] + '.txt')
                                np.savetxt(ascii_name, preds, fmt='%d')

                        t2 = time.time()
                        print('Done in {:.1f} s\n'.format(t2 - t1))

            test_epoch += 1


            # Break when reaching number of desired votes
            if last_min > config.num_votes:
                break
            else:
                print("Last min: {:.3f}, Num votes:{:d}".format(last_min, config.num_votes))

        return mIoU_full

    def slam_segmentation_test(self, net, test_loader, config, debug=True, model_root=None):
        # Use num_votes=100 for real testing scenario.
        """
        Test method for slam segmentation models
        """
        ############
        # Initialize
        ############

        # Choose validation smoothing parameter (0 for no smothing, 0.99 for big smoothing)
        test_smooth = config.test_smooth
        # Override this.
        test_smooth = .99
        print("Test with test_smooth: {}".format(test_smooth))
        last_min = -0.5
        softmax = torch.nn.Softmax(1)

        # Number of classes including ignored labels
        nc_tot = test_loader.dataset.num_classes
        nc_model = net.C

        # Test saving path
        test_path = None
        report_path = None
        if config.saving:
            if model_root is not None:
                test_path = join(model_root, 'test', config.saving_path.split('/')[-1])
            else:
                test_path = join('test', config.saving_path.split('/')[-1])
            print("Test path: {}".format(test_path))
            if not exists(test_path):
                makedirs(test_path)
            report_path = join(test_path, 'reports')
            if not exists(report_path):
                makedirs(report_path)

        if test_loader.dataset.set == 'validation' or test_loader.dataset.set == "mini-val":
            for folder in ['val_predictions', 'val_probs']:
                if not exists(join(test_path, folder)):
                    makedirs(join(test_path, folder))
        else:
            for folder in ['predictions', 'probs']:
                if not exists(join(test_path, folder)):
                    makedirs(join(test_path, folder))

        # Init validation container
        all_f_preds = []
        all_f_labels = []
        if test_loader.dataset.set == 'validation' or test_loader.dataset.set == 'mini-val':
            for i, seq_frames in enumerate(test_loader.dataset.frames):
                all_f_preds.append([np.zeros((0,), dtype=np.int32) for _ in seq_frames])
                all_f_labels.append([np.zeros((0,), dtype=np.int32) for _ in seq_frames])

        #####################
        # Network predictions
        #####################

        predictions = []
        targets = []
        test_epoch = 0

        t = [time.time()]
        last_display = time.time()
        mean_dt = np.zeros(1)

        # Start test loop
        while True:
            print('Initialize workers')
            for i, batch in enumerate(test_loader):
                if 'cuda' in self.device.type:
                    batch.to(self.device)
                # New time
                t = t[-1:]
                t += [time.time()]
                # Forward pass
                outputs = net(batch, config)
                t += [time.time()]
                # Get probs and labels
                stk_probs = softmax(outputs).cpu().detach().numpy()
                lengths = batch.lengths[0].cpu().numpy()
                f_inds = batch.frame_inds.cpu().numpy()
                r_inds_list = batch.reproj_inds
                r_mask_list = batch.reproj_masks
                labels_list = batch.val_labels
                # Get predictions and labels per instance
                # ***************************************
                i0 = 0
                for b_i, length in enumerate(lengths):
                    # Get prediction
                    probs = stk_probs[i0:i0 + length]
                    proj_inds = r_inds_list[b_i]
                    proj_mask = r_mask_list[b_i]
                    frame_labels = labels_list[b_i]
                    s_ind = f_inds[b_i, 0]
                    f_ind = f_inds[b_i, 1]
                    # Project predictions on the frame points
                    proj_probs = probs[proj_inds]
                    # Safe check if only one point:
                    if proj_probs.ndim < 2:
                        proj_probs = np.expand_dims(proj_probs, 0)
                    # Save probs in a binary file (uint8 format for lighter weight)
                    seq_name = test_loader.dataset.sequences[s_ind]
                    if test_loader.dataset.set == 'validation' or test_loader.dataset.set == 'mini-val':
                        folder = 'val_probs'
                        pred_folder = 'val_predictions'
                    else:
                        folder = 'probs'
                        pred_folder = 'predictions'
                    filename = '{:s}_{:07d}.npy'.format(seq_name, f_ind)
                    filepath = join(test_path, folder, filename)
                    if exists(filepath):
                        try:
                            frame_probs_uint8 = np.load(filepath)
                        except Exception:
                            print("Ignoring file {} as it is corrupted.".format(filepath))
                            # For some special reasons (e.g., downsamping is large.),
                            # the file may corrupt. Thus simply ignore the npy and make no prediction.
                            frame_probs_uint8 = np.zeros((proj_mask.shape[0], nc_model), dtype=np.uint8)
                    else:
                        frame_probs_uint8 = np.zeros((proj_mask.shape[0], nc_model), dtype=np.uint8)
                    frame_probs = frame_probs_uint8[proj_mask, :].astype(np.float32) / 255
                    frame_probs = test_smooth * frame_probs + (1 - test_smooth) * proj_probs
                    frame_probs_uint8[proj_mask, :] = (frame_probs * 255).astype(np.uint8)
                    np.save(filepath, frame_probs_uint8)
                    # Save some prediction in ply format for visual
                    if test_loader.dataset.set == 'validation' or test_loader.dataset.set == 'mini-val':
                        # Insert false columns for ignored labels
                        frame_probs_uint8_bis = frame_probs_uint8.copy()
                        for l_ind, label_value in enumerate(test_loader.dataset.label_values):
                            if label_value in test_loader.dataset.ignored_labels:
                                frame_probs_uint8_bis = np.insert(frame_probs_uint8_bis, l_ind, 0, axis=1)

                        # Predicted labels
                        frame_preds = test_loader.dataset.label_values[np.argmax(frame_probs_uint8_bis,
                                                                                 axis=1)].astype(np.int32)

                        # Save some of the frame pots
                        if f_ind % 20 == 0:
                            seq_path = join(test_loader.dataset.path, 'sequences', test_loader.dataset.sequences[s_ind])
                            velo_file = join(seq_path, 'velodyne', test_loader.dataset.frames[s_ind][f_ind] + '.bin')
                            frame_points = np.fromfile(velo_file, dtype=np.float32)
                            frame_points = frame_points.reshape((-1, 4))
                            predpath = join(test_path, pred_folder, filename[:-4] + '.ply')
                            #pots = test_loader.dataset.f_potentials[s_ind][f_ind]
                            pots = np.zeros((0,))
                            if pots.shape[0] > 0:
                                write_ply(predpath,
                                          [frame_points[:, :3], frame_labels, frame_preds, pots],
                                          ['x', 'y', 'z', 'gt', 'pre', 'pots'])
                            else:
                                write_ply(predpath,
                                          [frame_points[:, :3], frame_labels, frame_preds],
                                          ['x', 'y', 'z', 'gt', 'pre'])

                            # Also Save lbl probabilities
                            probpath = join(test_path, folder, filename[:-4] + '_probs.ply')
                            lbl_names = [test_loader.dataset.label_to_names[l]
                                         for l in test_loader.dataset.label_values
                                         if l not in test_loader.dataset.ignored_labels]
                            write_ply(probpath,
                                      [frame_points[:, :3], frame_probs_uint8],
                                      ['x', 'y', 'z'] + lbl_names)

                        # keep frame preds in memory
                        all_f_preds[s_ind][f_ind] = frame_preds
                        all_f_labels[s_ind][f_ind] = frame_labels

                    else:
                        # Save some of the frame preds
                        if f_inds[b_i, 1] % 100 == 0:

                            # Insert false columns for ignored labels
                            for l_ind, label_value in enumerate(test_loader.dataset.label_values):
                                if label_value in test_loader.dataset.ignored_labels:
                                    frame_probs_uint8 = np.insert(frame_probs_uint8, l_ind, 0, axis=1)

                            # Predicted labels
                            frame_preds = test_loader.dataset.label_values[np.argmax(frame_probs_uint8,
                                                                                     axis=1)].astype(np.int32)

                            # Load points
                            seq_path = join(test_loader.dataset.path, 'sequences', test_loader.dataset.sequences[s_ind])
                            velo_file = join(seq_path, 'velodyne', test_loader.dataset.frames[s_ind][f_ind] + '.bin')
                            frame_points = np.fromfile(velo_file, dtype=np.float32)
                            frame_points = frame_points.reshape((-1, 4))
                            predpath = join(test_path, pred_folder, filename[:-4] + '.ply')
                            #pots = test_loader.dataset.f_potentials[s_ind][f_ind]
                            pots = np.zeros((0,))
                            if pots.shape[0] > 0:
                                write_ply(predpath,
                                          [frame_points[:, :3], frame_preds, pots],
                                          ['x', 'y', 'z', 'pre', 'pots'])
                            else:
                                write_ply(predpath,
                                          [frame_points[:, :3], frame_preds],
                                          ['x', 'y', 'z', 'pre'])

                    # Stack all prediction for this epoch
                    i0 += length

                # Average timing
                t += [time.time()]
                dt = np.array(t[1:]) - np.array(t[:-1])
                fra1 = 0.9
                fra2 = 0.1
                warm_up = 5
                if i < warm_up:
                    mean_dt = dt
                else:
                    mean_dt = fra1 * mean_dt + fra2 * dt

                # Display
                if (t[-1] - last_display) > 1.0:
                    last_display = t[-1]
                    message = 'e{:03d}-i{:04d} => {:.1f}% (timings : {:4.2f} {:4.2f} {:4.2f}) / pots {:d} => {:.1f}%'
                    min_pot = int(torch.floor(torch.min(test_loader.dataset.potentials)))
                    pot_num = torch.sum(test_loader.dataset.potentials > min_pot + 0.5).type(torch.int32).item()
                    current_num = pot_num + (i + 1 - config.validation_size) * config.val_batch_num
                    print(message.format(test_epoch, i,
                                         100 * i / config.validation_size,
                                         1000 * (mean_dt[0]),
                                         1000 * (mean_dt[1]),
                                         1000 * (mean_dt[2]),
                                         min_pot,
                                         100.0 * current_num / len(test_loader.dataset.potentials)))


            # Update minimum od potentials
            new_min = torch.min(test_loader.dataset.potentials)
            print('Test epoch {:d}, end. Min potential = {:.1f}'.format(test_epoch, new_min))
            print("Last min: {:.3f}, new min: {:.3f}".format(last_min, new_min))

            if last_min + 1 < new_min:
                # Update last_min
                last_min += 1
                if test_loader.dataset.set == 'validation' or test_loader.dataset.set == "mini-val":
                    #####################################
                    # Results on the whole validation set
                    #####################################
                    # Confusions for our subparts of validation set
                    Confs = np.zeros((len(predictions), nc_tot, nc_tot), dtype=np.int32)
                    for i, (preds, truth) in enumerate(zip(predictions, targets)):
                        # Confusions
                        Confs[i, :, :] = fast_confusion(truth, preds, test_loader.dataset.label_values).astype(np.int32)
                    # Show vote results
                    print('\nCompute confusion')
                    val_preds = []
                    val_labels = []
                    t1 = time.time()
                    for i, seq_frames in enumerate(test_loader.dataset.frames):
                        val_preds += [np.hstack(all_f_preds[i])]
                        val_labels += [np.hstack(all_f_labels[i])]
                    val_preds = np.hstack(val_preds)
                    val_labels = np.hstack(val_labels)
                    t2 = time.time()
                    C_tot = fast_confusion(val_labels, val_preds, test_loader.dataset.label_values)
                    t3 = time.time()
                    print('Stacking time : {:.1f}s'.format(t2 - t1))
                    print('Confusion time : {:.1f}s'.format(t3 - t2))
                    s1 = '\n'
                    for cc in C_tot:
                        for c in cc:
                            s1 += '{:7.0f} '.format(c)
                        s1 += '\n'
                    if debug:
                        print(s1)
                    # Remove ignored labels from confusions
                    for l_ind, label_value in reversed(list(enumerate(test_loader.dataset.label_values))):
                        if label_value in test_loader.dataset.ignored_labels:
                            C_tot = np.delete(C_tot, l_ind, axis=0)
                            C_tot = np.delete(C_tot, l_ind, axis=1)
                    # Objects IoU
                    val_IoUs = IoU_from_confusions(C_tot)
                    # Compute IoUs
                    mIoU = np.mean(val_IoUs)
                    s2 = '{:5.2f} | '.format(100 * mIoU)
                    for IoU in val_IoUs:
                        s2 += '{:5.2f} '.format(100 * IoU)
                    print(s2 + '\n')

                    # Save a report
                    report_file = join(report_path, 'report_{:04d}.txt'.format(int(np.floor(last_min))))
                    str = 'Report of the confusion and metrics\n'
                    str += '***********************************\n\n\n'
                    str += 'Confusion matrix:\n\n'
                    str += s1
                    str += '\nIoU values:\n\n'
                    str += s2
                    str += '\n\n'
                    print("Saving report to {}".format(report_file))
                    with open(report_file, 'w') as f:
                        f.write(str)

            test_epoch += 1

            # Break when reaching number of desired votes
            if last_min > config.num_votes:
                break
            else:
                print("Last min: {:.3f}, Num votes:{:d}".format(last_min, config.num_votes))

        return mIoU

    def classification_test_latency(self, net, test_loader, config, debug=False, model_root=None, max_tries=25,
                                    inner_loop_trials=7):
        ############
        # Initialize
        ############
        net.eval()

        LAT = []

        # Start validation loop
        for idx, batch in enumerate(test_loader):
            #print(batch.features.shape)

            if 'cuda' in self.device.type:
                batch.to(self.device)

            if 'cuda' in self.device.type:
                torch.cuda.synchronize(self.device)

            batch_lat = []
            for _ in range(inner_loop_trials):
                start_time = time.time()
                # Forward pass
                _ = net(batch, config)
                if 'cuda' in self.device.type:
                    torch.cuda.synchronize(self.device)

                end_time = time.time()
                batch_lat.append(end_time - start_time)

            # 5 warmup steps.
            if idx < 2:
                continue
            LAT.append(np.mean(batch_lat))

            if idx > max_tries + 2:
                break

        mean_lat = np.mean(LAT)
        std_lat = np.std(LAT)

        return mean_lat, std_lat

    def cloud_segmentation_test_latency(self, net, test_loader, config, debug=False, model_root=None, max_tries=25,
                                        inner_loop_trials=7, use_profiler=False):
        # Use num_votes=100 for real testing scenario.
        """
        Test method for cloud segmentation models
        """

        if use_profiler:
            for i, batch in enumerate(test_loader):
                if 'cuda' in self.device.type:
                    batch.to(self.device)
                    if 'cuda' in self.device.type:
                        torch.cuda.synchronize(self.device)
                    with torch.autograd.profiler.profile(use_cuda=True) as prof:
                        _ = net(batch, config)
                    result = prof.key_averages().table(sort_by="cuda_time_total")
                    print(result)
                    return 0
                break

        ############
        # Initialize
        ############
        LAT = []

        for i, batch in enumerate(test_loader):
            # if i == 0:
            #    print('Done in {:.1f}s'.format(t[1] - t[0]))
            if 'cuda' in self.device.type:
                batch.to(self.device)

            if 'cuda' in self.device.type:
                torch.cuda.synchronize(self.device)

            batch_lats = []
            for _ in range(inner_loop_trials):
                start_time = time.time()
                # Forward pass
                _ = net(batch, config)

                if 'cuda' in self.device.type:
                    torch.cuda.synchronize(self.device)

                end_time = time.time()
                batch_lats.append(end_time - start_time)

            if i >= 2:
                LAT.append(np.mean(batch_lats))
            if i >= max_tries + 2:
                break

        mean_lat = np.mean(LAT)
        std_lat = np.std(LAT)
        return mean_lat, std_lat

    def slam_segmentation_test_latency(self, net, test_loader, config, debug=True, model_root=None, max_tries=25,
                                       inner_loop_trials=7, use_profiler=False):
        # Use num_votes=100 for real testing scenario.
        """
        Test method for slam segmentation models
        """
        ############
        # Initialize
        ############
        LAT = []
        if use_profiler:
            for i, batch in enumerate(test_loader):
                if 'cuda' in self.device.type:
                    batch.to(self.device)
                    if 'cuda' in self.device.type:
                        torch.cuda.synchronize(self.device)
                    with torch.autograd.profiler.profile(use_cuda=True) as prof:
                        _ = net(batch, config)
                    result = prof.key_averages().table(sort_by="cuda_time_total")
                    print(result)
                    return 0
                break

        for i, batch in enumerate(test_loader):

            # if i == 0:
            #    print('Done in {:.1f}s'.format(t[1] - t[0]))

            if 'cuda' in self.device.type:
                batch.to(self.device)

            if 'cuda' in self.device.type:
                torch.cuda.synchronize(self.device)

            batch_lats = []

            for _ in range(inner_loop_trials):
                start = time.time()
                # Forward pass
                _ = net(batch, config)
                if 'cuda' in self.device.type:
                    torch.cuda.synchronize(self.device)
                end = time.time()
                batch_lats.append(end-start)

            if i >= 2:
                LAT.append(np.mean(batch_lats))
            if i >= max_tries + 2:
                break

        mean_lat = np.mean(LAT)
        std_lat = np.std(LAT)
        return mean_lat, std_lat
    
    def test_flops(self, net, test_loader, config, model_root=None, max_tries=100):
        FLOPs = []
        net.eval()
        batches = []
        with torch.no_grad():
            for idx, batch in enumerate(test_loader):
                batches.append(batch.features.size()[0])
                if 'cuda' in self.device.type:
                    batch.to(self.device)
                _, flops = net.get_flops(batch, config)
                FLOPs.append(flops)

                if idx > max_tries:
                    break

        mean_FLOPs = np.mean(FLOPs)
        std_FLOPS = np.std(FLOPs)
        mean_batch_size = np.mean(batches)
        std_batch_size = np.std(batches)
        return mean_FLOPs, std_FLOPS, mean_batch_size, std_batch_size

    def test_latency_repeat(self, net, test_loader, config, max_tries=40,warm_up=5, repeat=20):
        ############
        # Initialize
        ############

        LAT = []

        # Start validation loop
        for idx, batch in enumerate(test_loader):
            #print(batch.features.shape)

            if 'cuda' in self.device.type:
                batch.to(self.device)

            #if 'cuda' in self.device.type:
            #    torch.cuda.synchronize(self.device)

            lat_b = 0
            print("idx:",idx,"size:",batch.features.shape)
            for i in range(repeat):

                if 'cuda' in self.device.type:
                    torch.cuda.synchronize(self.device)

                start_time = time.time()

                # Forward pass
                _ = net(batch, config)

                if 'cuda' in self.device.type:
                    torch.cuda.synchronize(self.device)

                end_time = time.time()

                print(i,"lat_b:",1000*(end_time-start_time))
                if i > warm_up:
                    lat_b += 1000*(end_time-start_time)

            #if 'cuda' in self.device.type:
            #    torch.cuda.synchronize(self.device)
            # 5 warmup steps.
            #if idx < 5:
            #    continue
            LAT.append(lat_b/(repeat-warm_up-1))
            print("#################")
            print(LAT[-1])
            print("################")

            if idx > 8:
                break

        mean_lat = np.mean(LAT)
        return mean_lat









