## Learning Modal-Invariant and Temporal-Memory for Video-based Visible-Infrared Person Re-Identification
This is the official implementation of our [paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Lin_Learning_Modal-Invariant_and_Temporal-Memory_for_Video-Based_Visible-Infrared_Person_Re-Identification_CVPR_2022_paper.pdf) 'Learning Modal-Invariant and Temporal-Memory for Video-based Visible-Infrared Person Re-Identification'.

### Usage
- Usage of this code is free for research purposes only. 
- This project is based on DDAG[1] ([paper](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123620222.pdf) and [official code](https://github.com/mangye16/DDAG)).
- Download and prepare data [VCM-HITSZ](https://github.com/VCM-project233/VCM-HITSZ-data).
- Download [weights](https://github.com/VCM-project233/VCM-HITSZ-data) of MITML.
- To begin testing.(See the code for more details).  
		```
		python test.py
		```
 - Please cite our paper for usage.
 ```
@inproceedings{lin2022learning,
  title={Learning Modal-Invariant and Temporal-Memory for Video-Based Visible-Infrared Person Re-Identification},
  author={Lin, Xinyu and Li, Jinxing and Ma, Zeyu and Li, Huafeng and Li, Shuang and Xu, Kaixiong and Lu, Guangming and Zhang, David},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={20973--20982},
  year={2022}
}
 ```

- Reference
	```
	[1]Ye M, Shen J, J. Crandall D, et al. Dynamic dual-attentive aggregation learning for visible-infrared person re-identification[C]//Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XVII 16. Springer International Publishing, 2020: 229-247.
	```
