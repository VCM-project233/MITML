from __future__ import print_function, absolute_import
import os.path as osp

import numpy as np



def decoder_pic_path(fname):
    base = fname[0:4]
    modality = fname[5]
    if modality == '1' :
        modality_str = 'ir'
    else:
        modality_str = 'rgb'
    T_pos = fname.find('T')
    D_pos = fname.find('D')
    F_pos = fname.find('F')
    camera = fname[D_pos:T_pos]
    picture = fname[F_pos+1:]
    path = base + '/' + modality_str + '/' + camera + '/' + picture
    return path


class VCM(object):
    root = '/data/VCM-HITSZ/'
    # training data
    train_name_path = osp.join(root,'info/train_name.txt')
    track_train_info_path = osp.join(root,'info/track_train_info.txt')
    # testing data
    test_name_path = osp.join(root,'info/test_name.txt')
    track_test_info_path = osp.join(root,'info/track_test_info.txt')
    query_IDX_path = osp.join(root,'info/query_IDX.txt')

    def __init__(self,min_seq_len=12):
        self._check_before_run()

        # prepare meta data
        train_names = self._get_names(self.train_name_path)
        track_train = self._get_tracks(self.track_train_info_path)

        # for test
        test_names = self._get_names(self.test_name_path)
        track_test = self._get_tracks(self.track_test_info_path)
        query_IDX =  self._get_query_idx(self.query_IDX_path)
        query_IDX -= 1

        track_query = track_test[query_IDX,:]
        print('query')
        print(track_query)
        gallery_IDX = [i for i in range(track_test.shape[0]) if i not in query_IDX]
        track_gallery = track_test[gallery_IDX,:]
        print('gallery')
        print(track_gallery)

        #---------visible to infrared-----------
        gallery_IDX_1 = self._get_query_idx(self.query_IDX_path)
        gallery_IDX_1 -= 1
        track_gallery_1 = track_test[gallery_IDX_1,:]

        query_IDX_1 = [j for j in range(track_test.shape[0]) if j not in gallery_IDX_1]
        track_query_1 = track_test[query_IDX_1,:]
        #-----------------------------------------

        train_ir, num_train_tracklets_ir,num_train_imgs_ir,train_rgb, num_train_tracklets_rgb,num_train_imgs_rgb,num_train_pids,ir_label,rgb_label = \
          self._process_data_train(train_names,track_train,relabel=True,min_seq_len=min_seq_len)


        query, num_query_tracklets, num_query_pids, num_query_imgs = \
          self._process_data_test(test_names, track_query, relabel=False, min_seq_len=min_seq_len)

        gallery, num_gallery_tracklets, num_gallery_pids, num_gallery_imgs = \
          self._process_data_test(test_names, track_gallery, relabel=False, min_seq_len=min_seq_len)


        #--------visible to infrared-----------
        query_1, num_query_tracklets_1, num_query_pids_1, num_query_imgs_1 = \
          self._process_data_test(test_names, track_query_1, relabel=False, min_seq_len=min_seq_len)

        gallery_1, num_gallery_tracklets_1, num_gallery_pids_1, num_gallery_imgs_1 = \
          self._process_data_test(test_names, track_gallery_1, relabel=False, min_seq_len=min_seq_len)
        #---------------------------------------

        print("=> VCM loaded")
        print("Dataset statistics:")
        print("---------------------------------")
        print("subset      | # ids | # tracklets")
        print("---------------------------------")
        print("train_ir    | {:5d} | {:8d}".format(num_train_pids,num_train_tracklets_ir))
        print("train_rgb   | {:5d} | {:8d}".format(num_train_pids,num_train_tracklets_rgb))
        print("query       | {:5d} | {:8d}".format(num_query_pids, num_query_tracklets))
        print("gallery     | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_tracklets))
        print("---------------------------------")
        print("ir_label    | {}".format(np.unique(ir_label)))
        print("rgb_label   | {}".format(np.unique(rgb_label)))

        self.train_ir = train_ir
        self.train_rgb = train_rgb
        self.ir_label = ir_label
        self.rgb_label = rgb_label

        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids
        self.num_query_tracklets = num_query_tracklets
        self.num_gallery_tracklets = num_gallery_tracklets

        #------- visible to infrared------------
        self.query_1 = query_1
        self.gallery_1 = gallery_1

        self.num_query_pids_1 = num_query_pids_1
        self.num_gallery_pids_1 = num_gallery_pids_1
        self.num_query_tracklets_1 = num_query_tracklets_1
        self.num_gallery_tracklets_1 = num_gallery_tracklets_1
        #---------------------------------------


    def _check_before_run(self):
        """check befor run """
        if not osp.exists(self.root):
            raise RuntimeError("'{}' is not available".format(self.root))
        if not osp.exists(self.train_name_path):
            raise RuntimeError("'{}' is not available".format(self.train_name_path))
        if not osp.exists(self.track_train_info_path):
            raise RuntimeError("'{}' is not available".format(self.track_train_info_path))
        if not osp.exists(self.query_IDX_path):
            raise RuntimeError("'{}' is not available".format(self.query_IDX_path))
        if not osp.exists(self.test_name_path):
            raise RuntimeError("'{}' is not available".format(self.test_name_path))
        if not osp.exists(self.track_test_info_path):
            raise RuntimeError("'{}' is not available".format(self.track_test_info_path))

    def _get_names(self,fpath):
        """get image name, retuen name list"""
        names = []
        with open(fpath,'r') as f:
            for line in f:
                new_line = line.rstrip()
                names.append(new_line)
        return names

    def _get_tracks(self,fpath):
        """get tracks file"""
        names = []
        with open(fpath,'r') as f:
            for line in f:
                new_line = line.rstrip()
                new_line.split(' ')

                tmp = new_line.split(' ')[0:]

                tmp = list(map(int, tmp))
                names.append(tmp)
        names = np.array(names)
        return names


    def _get_query_idx(self, fpath):
        with open(fpath, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                new_line.split(' ')

                tmp = new_line.split(' ')[0:]


                tmp = list(map(int, tmp))
                idxs = tmp
        idxs = np.array(idxs)
        print(idxs)
        return idxs

    def _process_data_train(self,names,meta_data,relabel=False,min_seq_len=0):
        num_tracklets = meta_data.shape[0]
        pid_list = list(set(meta_data[:,3].tolist()))
        num_pids = len(pid_list)

        # dict {pid : label}
        if relabel: pid2label = {pid: label for label, pid in enumerate(pid_list)}
        print('pid_list')
        print(pid_list)
        print(pid2label)
        tracklets_ir = []
        num_imgs_per_tracklet_ir = []
        ir_label = []

        tracklets_rgb = []
        num_imgs_per_tracklet_rgb = []
        rgb_label = []

        for tracklet_idx in range(num_tracklets):
            data = meta_data[tracklet_idx,...]
            m,start_index,end_index,pid,camid = data
            if relabel: pid = pid2label[pid]

            if m == 1:
                img_names = names[start_index-1:end_index]
                img_ir_paths = [osp.join(self.root,decoder_pic_path(img_name)) for img_name in img_names]
                if len(img_ir_paths) >= min_seq_len:
                    img_ir_paths = tuple(img_ir_paths)
                    ir_label.append(pid)
                    tracklets_ir.append((img_ir_paths,pid,camid))
                    # same id
                    num_imgs_per_tracklet_ir.append(len(img_ir_paths))
            else:
                img_names = names[start_index-1:end_index]
                img_rgb_paths = [osp.join(self.root,decoder_pic_path(img_name)) for img_name in img_names]
                if len(img_rgb_paths) >= min_seq_len:
                    img_rgb_paths = tuple(img_rgb_paths)
                    rgb_label.append(pid)
                    tracklets_rgb.append((img_rgb_paths,pid,camid))
                    #same id
                    num_imgs_per_tracklet_rgb.append(len(img_rgb_paths))

        num_tracklets_ir = len(tracklets_ir)
        num_tracklets_rgb = len(tracklets_rgb)
        num_tracklets = num_tracklets_rgb  + num_tracklets_ir

        return tracklets_ir, num_tracklets_ir,num_imgs_per_tracklet_ir,tracklets_rgb,num_tracklets_rgb,num_imgs_per_tracklet_rgb,num_pids,ir_label,rgb_label


    def _process_data_test(self,names,meta_data,relabel=False,min_seq_len=0):
        num_tracklets = meta_data.shape[0]
        pid_list = list(set(meta_data[:,3].tolist()))
        num_pids = len(pid_list)

        # dict {pid : label}
        if relabel: pid2label = {pid: label for label, pid in enumerate(pid_list)}
        tracklets = []
        num_imgs_per_tracklet = []

        for tracklet_idx in range(num_tracklets):
            data = meta_data[tracklet_idx,...]
            m,start_index,end_index,pid,camid = data
            if relabel: pid = pid2label[pid]

            img_names = names[start_index-1:end_index]
            img_paths = [osp.join(self.root,decoder_pic_path(img_name)) for img_name in img_names]
            if len(img_paths) >= min_seq_len:
                img_paths = tuple(img_paths)
                tracklets.append((img_paths, pid, camid))
                num_imgs_per_tracklet.append(len(img_paths))

        num_tracklets = len(tracklets)

        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet

if __name__ == '__main__':
    dataset = VCM()
